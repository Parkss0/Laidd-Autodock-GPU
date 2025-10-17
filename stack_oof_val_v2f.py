
#!/usr/bin/env python3
import argparse, os, json, re, warnings
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression as LRCal
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:
    LGBMClassifier = None
    _HAS_LGBM = False

try:
    from scipy.special import logit as _scipy_logit
    def _to_logit(p): 
        p = np.asarray(p, float)
        p = np.clip(p, 1e-6, 1-1e-6)
        return _scipy_logit(p)
except Exception:
    def _to_logit(p): 
        p = np.asarray(p, float)
        p = np.clip(p, 1e-6, 1-1e-6)
        return np.log(p/(1-p))

def _zscore(X: pd.DataFrame) -> pd.DataFrame:
    mu = X.mean()
    sd = X.std(ddof=0).replace(0, 1.0)
    return (X - mu) / sd

CANONICAL = [
    "NR-AhR","NR-AR","NR-AR-LBD","NR-Aromatase","NR-ER","NR-ER-LBD","NR-PPAR-gamma",
    "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"
]
META_COL_CANDIDATES = {"smiles","SMILES","Smile","fold","Fold","oof","OOF","split","Split"}

def autodetect_id_col(df: pd.DataFrame) -> str:
    for c in ["molecule_id","Sample ID","sample_id","ID","id","ncgc_id","NCGC","index","compound_id","compound id"]:
        if c in df.columns: return c
    return df.columns[0]

def to_str_ids(s): return s.astype(str).map(lambda x: x.strip())

def clip01(p):
    p = np.asarray(p, dtype=float)
    return np.clip(p, 1e-6, 1-1e-6)

def safe_auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    if len(np.unique(y)) < 2: return np.nan
    try: return roc_auc_score(y, p)
    except: return np.nan

def safe_ap(y, p):
    y = np.asarray(y); p = np.asarray(p)
    if len(np.unique(y)) < 2: return np.nan
    try: return average_precision_score(y, p)
    except: return np.nan

def best_f1_threshold(y, p):
    y = np.asarray(y); p = np.asarray(p)
    ts = np.linspace(0.05, 0.95, 19)
    best_t, best_f = 0.5, -1
    for t in ts:
        f = f1_score(y, (p>=t).astype(int), zero_division=0)
        if f > best_f: best_f, best_t = f, t
    return best_t, best_f

def evaluate_model(name, y, p, task) -> Dict:
    p = clip01(p)
    res = {"model":name, "task":task}
    res["auc"]   = safe_auc(y,p)
    res["auprc"] = safe_ap(y,p)
    try: res["logloss"] = log_loss(y, p)
    except: res["logloss"] = np.nan
    try: res["brier"]   = brier_score_loss(y, p)
    except: res["brier"] = np.nan
    res["f1@0.5"] = f1_score(y, (p>=0.5).astype(int), zero_division=0)
    t,f = best_f1_threshold(y,p)
    res["best_thr"], res["f1@best"] = t, f
    return res

# --- header harmonization ---
def _norm_key(s: str) -> str:
    if not isinstance(s, str): s = str(s)
    s = s.strip()
    s = re.sub(r"[\s_]+", "-", s)
    s = s.lower()
    s = re.sub(r"^(prob|proba|probability|pred|prediction|oof|out|outputs?)[-_]", "", s)
    s = re.sub(r"[-_](pred|proba|prob(?:ability)?|oof|p|pr|prc|pro)$", "", s)
    s = s.replace("ppar-g", "ppar-gamma")
    s = s.replace("aromat", "aromatase")
    s = re.sub(r"-{2,}", "-", s)
    return s

CANON_MAP = { _norm_key(c): c for c in CANONICAL }

def harmonize_oof_columns(df: pd.DataFrame) -> Dict[str, str]:
    id_like = {autodetect_id_col(df), "SMILES", "smiles", "Smiles", "SMILE"}
    mapping: Dict[str, str] = {}
    for c in df.columns:
        if c in id_like: 
            continue
        key = _norm_key(c)
        can = CANON_MAP.get(key)
        if can is None:
            for k, v in CANON_MAP.items():
                if key == k or key.startswith(k) or k.startswith(key):
                    can = v; break
        if can is not None and can not in mapping:
            mapping[can] = c
    return mapping

def load_oof_file(path: str, id_col: str=None) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    cands = []
    if id_col: cands.append(id_col)
    cands += ["molecule_id","Sample ID","sample_id","ID","id","ncgc_id","NCGC","index","compound_id","compound id"]
    idc = next((c for c in cands if c in df.columns), None)
    if idc is None:
        raise ValueError(f"[{path}] cannot find id column. tried={cands}, cols={df.columns.tolist()}")
    return df, idc

# --- stacking helpers (fixed) ---
def stacked_oof_lr(Y: np.ndarray, feats: pd.DataFrame, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(Y), dtype=float)
    coefs = []
    for tr, va in skf.split(feats, Y):
        X_tr, y_tr = feats.iloc[tr], Y[tr]
        X_va       = feats.iloc[va]
        best_model, best_loss = None, 1e9
        for C in [0.1, 1.0, 3.0]:
            mdl = LogisticRegression(max_iter=2000, class_weight="balanced", solver="saga", C=C)
            mdl.fit(X_tr, y_tr)
            p_tr = mdl.predict_proba(X_tr)[:,1]
            try: ll = log_loss(y_tr, clip01(p_tr))
            except: ll = 1e9
            if ll < best_loss: best_loss, best_model = ll, mdl
        mdl = best_model
        oof[va] = mdl.predict_proba(X_va)[:,1]
        if hasattr(mdl, "coef_"): coefs.append(mdl.coef_.ravel())
    coef_mean = np.nanmean(np.vstack(coefs), axis=0) if coefs else None
    return clip01(oof), {"n_splits":n_splits, "seed":seed, "coef_mean":(coef_mean.tolist() if coef_mean is not None else None)}

def stacked_oof_lgbm(Y: np.ndarray, feats: pd.DataFrame, n_splits=5, seed=42, calibrate=True):
    if not _HAS_LGBM:
        raise RuntimeError("lightgbm not installed. pip install lightgbm")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(Y), dtype=float)
    imp_accum = np.zeros(feats.shape[1], dtype=float)
    for tr, va in skf.split(feats, Y):
        X_tr, y_tr = feats.iloc[tr], Y[tr]
        X_va       = feats.iloc[va]
        grid = [
            dict(num_leaves=5, min_child_samples=20, learning_rate=0.1, n_estimators=200),
            dict(num_leaves=7, min_child_samples=30, learning_rate=0.05, n_estimators=300),
            dict(num_leaves=3, min_child_samples=10, learning_rate=0.1, n_estimators=150),
        ]
        best_model, best_loss = None, 1e9
        for g in grid:
            mdl = LGBMClassifier(objective="binary", boosting_type="gbdt",
                                 n_jobs=-1, reg_lambda=1.0,
                                 subsample=1.0, colsample_bytree=1.0, **g)
            mdl.fit(X_tr, y_tr)
            p_tr = mdl.predict_proba(X_tr)[:,1]
            try: ll = log_loss(y_tr, clip01(p_tr))
            except: ll = 1e9
            if ll < best_loss: best_loss, best_model = ll, mdl
        # best_model로 다시 계산
        p_tr = best_model.predict_proba(X_tr)[:,1]
        p_va = best_model.predict_proba(X_va)[:,1]
        if calibrate and len(np.unique(y_tr))>=2:
            cal = LRCal(max_iter=2000); cal.fit(p_tr.reshape(-1,1), y_tr)
            p_va = cal.predict_proba(p_va.reshape(-1,1))[:,1]
        oof[va] = p_va
        if hasattr(best_model, "feature_importances_"): imp_accum += best_model.feature_importances_
    imp_mean = (imp_accum / n_splits).tolist()
    return clip01(oof), {"n_splits":n_splits, "seed":seed, "calibrated":bool(calibrate), "feat_importance_mean": imp_mean}

def main():
    ap = argparse.ArgumentParser(description="OOF stacking/blending (LR+LGBM) with fixes + logit stacking + topK weighted blend")
    ap.add_argument("--oofs", nargs="+", required=True, help="List of OOF CSV files (>=4)")
    ap.add_argument("--labels", required=True, help="Validation labels CSV")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--nfolds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=2.5, help="power for weighted blend (w ∝ (AUC-0.5)^alpha)")
    ap.add_argument("--topk", type=int, default=3, help="use top-K models per task for weighted blend (<= #models)")
    ap.add_argument("--no-logit", action="store_true", help="disable logit transform for stacking")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # labels
    ydf = pd.read_csv(args.labels)
    y_id = args.id_col or autodetect_id_col(ydf)
    if y_id not in ydf.columns:
        raise SystemExit(f"ID column not found in labels: {y_id}")
    task_cols = [c for c in CANONICAL if c in ydf.columns]
    if not task_cols:
        task_cols = [c for c in ydf.columns if c != y_id and c not in META_COL_CANDIDATES]
    ydf[y_id] = to_str_ids(ydf[y_id])
    ydf = ydf[[y_id]+task_cols].copy()

    # OOFs + mapping
    def harmonize(df: pd.DataFrame) -> Dict[str,str]:
        id_like = {autodetect_id_col(df), "SMILES", "smiles", "Smiles", "SMILE"}
        mapping = {}
        for c in df.columns:
            if c in id_like: continue
            key = _norm_key(c)
            can = CANON_MAP.get(key)
            if can is None:
                for k,v in CANON_MAP.items():
                    if key == k or key.startswith(k) or k.startswith(key):
                        can = v; break
            if can is not None and can not in mapping:
                mapping[can] = c
        return mapping

    model_tables, model_names, colmaps = [], [], []
    for p in args.oofs:
        df, oid = load_oof_file(p, args.id_col)
        if oid != y_id: df = df.rename(columns={oid: y_id})
        df[y_id] = to_str_ids(df[y_id])
        model_tables.append(df)
        model_names.append(Path(p).stem.replace(" ", "_"))
        colmaps.append(harmonize(df))

    # intersect IDs across labels & all OOFs
    ids = set(ydf[y_id])
    for df in model_tables: ids &= set(df[y_id])
    ids = list(sorted(ids))
    if not ids: raise SystemExit("No overlapping IDs across labels and OOF files.")
    ydf = ydf[ydf[y_id].isin(ids)].reset_index(drop=True)

    # mapping report
    map_rows = []
    for name, mp in zip(model_names, colmaps):
        for t in CANONICAL:
            map_rows.append({"model": name, "task": t, "mapped_col": mp.get(t, None)})
    pd.DataFrame(map_rows).to_csv(os.path.join(args.out_dir, "oof_column_mapping.csv"), index=False)

    # collect preds for all ids (may contain NaN)
    model_preds = {}
    usable_tasks = [t for t in task_cols if any(t in m for m in colmaps)]
    for name, df, mp in zip(model_names, model_tables, colmaps):
        row = pd.DataFrame({y_id: ids})
        df_idx = df.set_index(y_id)
        for t in usable_tasks:
            col = mp.get(t)
            row[t] = pd.to_numeric(df_idx.reindex(ids)[col].values, errors="coerce") if col else np.nan
        model_preds[name] = row

    tasks = [t for t in usable_tasks if t in ydf.columns]
    if not tasks: raise SystemExit("No overlapping tasks between labels and OOFs.")

    # helper: get valid mask per task (labels available)
    def task_mask(t):
        y_series = pd.to_numeric(ydf.set_index(y_id).loc[ids, t], errors="coerce")
        return ~y_series.isna(), y_series  # mask, float labels

    # ---- Base metrics (mask applied) ----
    eval_rows = []
    for t in tasks:
        mask, y_float = task_mask(t)
        yv = y_float[mask].astype(int).values
        for name in model_names:
            p = model_preds[name][t].values
            pv = p[mask.values]
            if np.all(np.isnan(pv)): continue
            eval_rows.append(evaluate_model(name, yv, pv, t))
    pd.DataFrame(eval_rows).to_csv(os.path.join(args.out_dir, "metrics_per_model.csv"), index=False)

    # ---- Blending (avg / weighted / power top-k) ----
    blend_avg = pd.DataFrame({y_id: ids})
    blend_wavg = pd.DataFrame({y_id: ids})
    blend_pow_topk = pd.DataFrame({y_id: ids})
    for t in tasks:
        mask, y_float = task_mask(t)
        yv = y_float[mask].astype(int).values
        Ps_all, aucs, names = [], [], []
        for name in model_names:
            p = model_preds[name][t].values
            if np.all(np.isnan(p)): continue
            Ps_all.append(p); names.append(name)
            aucs.append(safe_auc(yv, p[mask.values]))
        Ps_all = np.stack(Ps_all, axis=1) if Ps_all else np.full((len(ids),0), np.nan)
        pa = np.nanmean(Ps_all, axis=1) if Ps_all.size else np.full(len(ids), np.nan)
        # simple wavg (legacy)
        w = np.array([max(a if not np.isnan(a) else 0.5, 0.5001)-0.5 for a in aucs], dtype=float)
        if w.size == 0 or w.sum() <= 1e-12: w = np.ones_like(w if w.size else np.array([1.0]))
        w = w / w.sum()
        pw = (Ps_all * w[None,:]).sum(axis=1) if Ps_all.size else np.full(len(ids), np.nan)
        # power top-k
        order = np.argsort(-np.nan_to_num(aucs, nan=0.5))
        keep = order[:max(1, min(args.topk, len(order)))]
        auc_k = np.nan_to_num(np.array(aucs)[keep], nan=0.5)
        w2 = np.maximum(auc_k - 0.5, 1e-6) ** float(args.alpha)
        w2 = w2 / w2.sum()
        pt = (Ps_all[:, keep] * w2[None,:]).sum(axis=1) if Ps_all.size else np.full(len(ids), np.nan)
        blend_avg[t], blend_wavg[t], blend_pow_topk[t] = pa, pw, pt
    blend_avg.to_csv(os.path.join(args.out_dir, "oof_blend_avg.csv"), index=False)
    blend_wavg.to_csv(os.path.join(args.out_dir, "oof_blend_wavg.csv"), index=False)
    blend_pow_topk.to_csv(os.path.join(args.out_dir, "oof_blend_pow_topk.csv"), index=False)

    # ---- Stacking (logit + zscore) ----
    stack_lr  = pd.DataFrame({y_id: ids})
    stack_lgb = pd.DataFrame({y_id: ids})
    lgb_imp_rows = []
    for t in tasks:
        mask, y_float = task_mask(t)
        yv = y_float[mask].astype(int).values
        # features (valid only)
        X = pd.DataFrame({f"{name}": model_preds[name][t].values for name in model_names})
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
        if not args.no_logit:
            X = X.apply(_to_logit)
        Xv = X.iloc[mask.values]
        Xv = _zscore(Xv)
        # LR
        if len(yv) >= 2 and Xv.shape[1] >= 1:
            oof_lr_v, meta_lr = stacked_oof_lr(yv, Xv, n_splits=args.nfolds, seed=args.seed)
            arr = np.full(len(ids), np.nan); arr[mask.values] = oof_lr_v
            stack_lr[t] = arr
        else:
            stack_lr[t] = np.full(len(ids), np.nan)
        # LGBM
        if _HAS_LGBM and len(yv) >= 2 and Xv.shape[1] >= 1:
            oof_lg_v, meta_lg = stacked_oof_lgbm(yv, Xv, n_splits=args.nfolds, seed=args.seed, calibrate=True)
            arr = np.full(len(ids), np.nan); arr[mask.values] = oof_lg_v
            stack_lgb[t] = arr
            if meta_lg.get("feat_importance_mean") is not None:
                row = dict(task=t, **{model_names[i]: meta_lg["feat_importance_mean"][i] for i in range(len(model_names))})
                lgb_imp_rows.append(row)
        else:
            stack_lgb[t] = np.full(len(ids), np.nan)

    stack_lr.to_csv (os.path.join(args.out_dir, "oof_stack_lr.csv"),   index=False)
    stack_lgb.to_csv(os.path.join(args.out_dir, "oof_stack_lgbm.csv"), index=False)
    if lgb_imp_rows:
        pd.DataFrame(lgb_imp_rows).to_csv(os.path.join(args.out_dir, "lgbm_meta_importances.csv"), index=False)

    # ---- Summary (metrics computed on valid rows only) ----
    summary_rows = []
    methods = [("blend_avg", blend_avg), ("blend_wavg", blend_wavg), ("blend_pow_topk", blend_pow_topk), ("stack_lr", stack_lr)]
    if _HAS_LGBM: methods.append(("stack_lgbm", stack_lgb))
    for t in tasks:
        mask, y_float = task_mask(t)
        if mask.sum() < 2:  # not enough labels
            continue
        yv = y_float[mask].astype(int).values
        for name in model_names:
            p = model_preds[name][t].values[mask.values]
            if np.all(np.isnan(p)): continue
            summary_rows.append(evaluate_model(name, yv, p, t))
        for mname, mdf in methods:
            pv = pd.to_numeric(mdf[t].values[mask.values], errors="coerce")
            if np.all(np.isnan(pv)): continue
            summary_rows.append(evaluate_model(mname, yv, pv, t))
    sm = pd.DataFrame(summary_rows)
    sm.to_csv(os.path.join(args.out_dir, "summary_per_task.csv"), index=False)

    if not sm.empty:
        overall = (sm.groupby("model")[["auc","auprc","logloss","brier","f1@0.5","f1@best"]]
                     .mean(numeric_only=True).reset_index())
        overall.to_csv(os.path.join(args.out_dir, "summary_overall.csv"), index=False)

        best_rows = []
        for t in tasks:
            sdf = sm[sm["task"]==t]
            if sdf.empty: continue
            idx = sdf["auc"].idxmax()
            if pd.notna(idx):
                best_rows.append(sdf.loc[idx].to_dict())
        if best_rows:
            pd.DataFrame(best_rows).to_csv(os.path.join(args.out_dir, "best_method_per_task_by_auc.csv"), index=False)

    # meta info
    with open(os.path.join(args.out_dir, "run_meta.json"), "w") as f:
        json.dump({
            "id_col": y_id, "n_ids": len(ids),
            "models": model_names, "nfolds": args.nfolds,
            "seed": args.seed, "has_lightgbm": _HAS_LGBM,
            "tasks_present": tasks, "alpha": args.alpha, "topk": args.topk, "logit": (not args.no_logit)
        }, f, indent=2)

    print("[OK] tasks:", tasks)
    for fn in ["oof_column_mapping.csv","metrics_per_model.csv","oof_blend_avg.csv",
               "oof_blend_wavg.csv","oof_blend_pow_topk.csv",
               "oof_stack_lr.csv","oof_stack_lgbm.csv",
               "summary_per_task.csv","summary_overall.csv","best_method_per_task_by_auc.csv"]:
        print("[SAVE]", os.path.join(args.out_dir, fn))

if __name__ == "__main__":
    main()
