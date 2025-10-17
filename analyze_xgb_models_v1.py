
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os, re, sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import xgboost as xgb

LABEL_COLUMNS = ["NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase","NR-ER","NR-ER-LBD","NR-PPAR-gamma","SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"]

def _slug(s:str)->str:
    s=str(s).strip()
    s=re.sub(r"[^\w\-\.]+","_",s)
    s=re.sub(r"_+","_",s).strip("_")
    return s

def _norm_id(s:pd.Series)->pd.Series:
    return s.astype(str).str.strip().str.replace(r"\s+","",regex=True).str.upper()

def _ensure_dirs(base:Path):
    for d in ["metrics","plots","shap","shap/by_task","shap/receptor","gain_receptor","tables"]:
        (base/d).mkdir(parents=True, exist_ok=True)

def _micro_metrics(y_true_list:List[np.ndarray], y_score_list:List[np.ndarray]) -> Dict[str,float]:
    y = np.concatenate(y_true_list); p = np.concatenate(y_score_list)
    # ROC AUC
    auc = roc_auc_score(y, p)
    # PR AUC (average precision)
    ap = average_precision_score(y, p)
    # Curves
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    return {"micro_auc": float(auc), "micro_auprc": float(ap),
            "fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec}

def _bar(ax, labels, values, title, ylim=None, annotate=True, rotation=45):
    order = np.argsort(values)[::-1]
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    ax.bar(labels, values)
    if annotate:
        for i,v in enumerate(values):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    if ylim: ax.set_ylim(ylim)
    ax.set_xticklabels(labels, rotation=rotation, ha="right")

def _build_task_matrix(X_merged:pd.DataFrame, feat_list:List[str]) -> np.ndarray:
    X = X_merged.copy()
    # 누락 열 추가
    for c in feat_list:
        if c not in X.columns:
            X[c] = np.nan
    X = X.loc[:, feat_list]
    # 숫자형 강제 + NaN 채우기(전부 NaN이면 0, 아니면 중앙값)
    arr = X.values.astype(float)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        if np.isnan(col).all():
            arr[:, j] = 0.0
        else:
            med = np.nanmedian(col)
            col[np.isnan(col)] = med
            arr[:, j] = col
    return arr

def _prefix_of(feat:str) -> str:
    # receptor prefix 는 "<RECEPTOR>__feature" 구조
    return feat.split("__", 1)[0] if "__" in feat else "UNKNOWN"

def compute_metrics(models_dir:Path, out_dir:Path) -> Tuple[pd.DataFrame, Dict]:
    """OOF + labels 로 per-assay AUC/AUPRC 및 micro 계산, 플롯 저장"""
    oof_path = models_dir/"oof"/"oof_predictions.csv"
    lab_path = models_dir/"cache"/"labels.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"Missing OOF predictions: {oof_path}")
    if not lab_path.exists():
        raise FileNotFoundError(f"Missing labels: {lab_path}")

    oof = pd.read_csv(oof_path)
    labs = pd.read_csv(lab_path)

    # ID 정규화/조인
    def guess_id(df):
        low={c.lower():c for c in df.columns}
        for k in ["compound_id","ligand_id","molid","sample id","sample_id","id","ncgc id","ncgcid"]:
            if k in low: return low[k]
        return df.columns[0]
    ic1 = guess_id(oof); ic2 = guess_id(labs)
    if ic1!="compound_id": oof=oof.rename(columns={ic1:"compound_id"})
    if ic2!="compound_id": labs=labs.rename(columns={ic2:"compound_id"})
    oof["compound_id"]=_norm_id(oof["compound_id"]); labs["compound_id"]=_norm_id(labs["compound_id"])

    df = pd.merge(labs, oof, on="compound_id", how="inner", suffixes=("_y","_pred"))
    # 사용 가능한 task
    tasks = [t for t in LABEL_COLUMNS if (t in df.columns) and (t in oof.columns)]
    rows=[]
    y_all, p_all = [], []
    for t in tasks:
        sub = df[["compound_id", t, t]].copy()
        sub.columns = ["compound_id","y","p"]
        sub = sub.dropna(subset=["y","p"])
        if len(sub)==0:
            rows.append({"task":t, "n":0, "pos":0, "neg":0, "auc":np.nan, "auprc":np.nan})
            continue
        y = sub["y"].astype(float).values
        p = sub["p"].astype(float).values
        # 일부 레이블이 {-1,0,1}일 수 있으니 -1 제거
        mask = (y==0) | (y==1)
        y, p = y[mask], p[mask]
        if len(np.unique(y))<2:
            auc, ap = np.nan, np.nan
        else:
            auc = roc_auc_score(y, p)
            ap  = average_precision_score(y, p)
            y_all.append(y); p_all.append(p)
        rows.append({"task":t, "n":int(len(y)), "pos":int((y==1).sum()), "neg":int((y==0).sum()), "auc":float(auc), "auprc":float(ap)})

    per_task = pd.DataFrame(rows)
    per_task.to_csv(out_dir/"metrics"/"assay_metrics_from_oof.csv", index=False)

    micro = {}
    if len(y_all)>0:
        micro = _micro_metrics(y_all, p_all)
        # 곡선 플롯
        fig = plt.figure(figsize=(5,4))
        plt.plot(micro["fpr"], micro["tpr"], label=f"AUC={micro['micro_auc']:.3f}")
        plt.plot([0,1],[0,1],'--',lw=1)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("Micro-averaged ROC (12 assays)")
        plt.legend()
        fig.tight_layout()
        fig.savefig(out_dir/"plots"/"micro_roc.png", dpi=200)
        plt.close(fig)

        fig = plt.figure(figsize=(5,4))
        plt.plot(micro["rec"], micro["prec"], label=f"AP={micro['micro_auprc']:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Micro-averaged PR (12 assays)")
        plt.legend()
        fig.tight_layout()
        fig.savefig(out_dir/"plots"/"micro_pr.png", dpi=200)
        plt.close(fig)

    # per-assay bar charts
    if not per_task.empty:
        fig, ax = plt.subplots(figsize=(8,4))
        _bar(ax, per_task["task"].tolist(), per_task["auc"].fillna(0).tolist(), "Per-assay AUROC")
        fig.tight_layout(); fig.savefig(out_dir/"plots"/"assay_auc_bar.png", dpi=200); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8,4))
        _bar(ax, per_task["task"].tolist(), per_task["auprc"].fillna(0).tolist(), "Per-assay AUPRC")
        fig.tight_layout(); fig.savefig(out_dir/"plots"/"assay_auprc_bar.png", dpi=200); plt.close(fig)

    # 요약 저장
    summary = {"micro_auc": micro.get("micro_auc", None), "micro_auprc": micro.get("micro_auprc", None)}
    (out_dir/"metrics"/"overall_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return per_task, summary

def compute_gain_by_receptor(models_dir:Path, out_dir:Path) -> pd.DataFrame:
    """훈련 시 저장된 Gain 기반 feature importance를 receptor 단위로 집계"""
    imp_dir = models_dir/"importances"
    rows=[]
    for p in sorted(imp_dir.glob("*_feature_importance.csv")):
        task = p.name.replace("_feature_importance.csv","")
        df = pd.read_csv(p)
        if df.empty: 
            continue
        df["receptor"] = df["feature"].astype(str).map(_prefix_of)
        g = df.groupby("receptor")["gain"].sum().reset_index()
        g["task"] = task
        # task별 정규화 비중(합=1)
        total = g["gain"].sum()
        g["gain_share"] = g["gain"] / (total if total>0 else 1.0)
        rows.append(g)
    if not rows:
        return pd.DataFrame()
    allg = pd.concat(rows, ignore_index=True)
    allg.to_csv(out_dir/"gain_receptor"/"per_task_gain_by_receptor.csv", index=False)

    # overall: task별 share를 평균
    overall = allg.groupby("receptor")["gain_share"].mean().reset_index().sort_values("gain_share", ascending=False)
    overall.to_csv(out_dir/"gain_receptor"/"overall_gain_receptor_share.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(7,4))
    _bar(ax, overall["receptor"].tolist(), overall["gain_share"].tolist(), "Receptor importance (Gain, mean share)", ylim=(0, overall["gain_share"].max()*1.2 if len(overall)>0 else None))
    fig.tight_layout(); fig.savefig(out_dir/"plots"/"receptor_gain_share_bar.png", dpi=200); plt.close(fig)
    return overall

def compute_shap(models_dir:Path, out_dir:Path, topk:int=30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    XGBoost pred_contribs (SHAP values)로
    - 피처 중요도(mean |SHAP|) per task
    - receptor 중요도(mean |SHAP| 집계) per task + overall
    """
    X_path = models_dir/"cache"/"X_merged.csv"
    if not X_path.exists():
        raise FileNotFoundError(f"Missing merged features: {X_path}")
    X_merged = pd.read_csv(X_path)

    model_files = [p for p in sorted((models_dir/"models").glob("*.json")) if not p.name.endswith("_features.json")]
    if not model_files:
        raise FileNotFoundError(f"No model *.json under {models_dir/'models'}")

    feat_meta = {}
    for p in model_files:
        task = p.stem
        meta = (models_dir/"models"/f"{task}_features.json")
        if meta.exists():
            feat_meta[task] = json.loads(meta.read_text(encoding="utf-8")).get("feature_columns", [])
        else:
            # 폴백: 숫자열 전부
            cols = [c for c in X_merged.columns if c not in ("compound_id","smiles")]
            feat_meta[task] = [c for c in cols if np.issubdtype(X_merged[c].dtype, np.number)]

    per_task_receptor = []
    all_receptor_norm = []

    for p in model_files:
        task = p.stem
        feat_list = feat_meta.get(task, [])
        if not feat_list: 
            continue

        # DMatrix 구성 (feature_names 필수)
        Xnum = _build_task_matrix(X_merged, feat_list)
        dmat = xgb.DMatrix(Xnum, feature_names=feat_list)

        # Booster 로드 + SHAP
        booster = xgb.Booster()
        booster.load_model(p.as_posix())
        shap = booster.predict(dmat, pred_contribs=True)  # (n_samples, n_features+1), last is bias
        if shap.ndim!=2 or shap.shape[1] != len(feat_list)+1:
            # 호환성 문제 시 sklearn wrapper 시도
            clf = xgb.XGBClassifier()
            clf.load_model(p.as_posix())
            shap = clf.get_booster().predict(dmat, pred_contribs=True)

        shap_feat = np.abs(shap[:, :-1]).mean(axis=0)  # mean |SHAP| per feature
        feat_imp = pd.DataFrame({"feature": feat_list, "mean_abs_shap": shap_feat}).sort_values("mean_abs_shap", ascending=False)
        feat_imp.to_csv(out_dir/"shap"/"by_task"/f"{_slug(task)}__shap_feature_importance.csv", index=False)

        # Top-K Feature bar
        top = feat_imp.head(topk)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.25*len(top))))
        _bar(ax, top["feature"].tolist(), top["mean_abs_shap"].tolist(), f"{task} - Top{topk} features (mean |SHAP|)", annotate=False, rotation=75)
        fig.tight_layout(); fig.savefig(out_dir/"plots"/f"{_slug(task)}__top{topk}_features_shap.png", dpi=200); plt.close(fig)

        # Receptor 집계
        feat_imp["receptor"] = feat_imp["feature"].map(_prefix_of)
        rec = feat_imp.groupby("receptor")["mean_abs_shap"].sum().reset_index().sort_values("mean_abs_shap", ascending=False)
        rec["task"] = task
        # task 내 정규화(합=1)
        s = rec["mean_abs_shap"].sum()
        rec["share"] = rec["mean_abs_shap"] / (s if s>0 else 1.0)
        per_task_receptor.append(rec)

        # per task receptor bar
        fig, ax = plt.subplots(figsize=(7,4))
        _bar(ax, rec["receptor"].tolist(), rec["share"].tolist(), f"{task} - Receptor importance (SHAP share)", ylim=(0, max(0.5, rec["share"].max()*1.2)))
        fig.tight_layout(); fig.savefig(out_dir/"plots"/f"{_slug(task)}__receptor_shap_share.png", dpi=200); plt.close(fig)

    if not per_task_receptor:
        return pd.DataFrame(), pd.DataFrame()

    per_task_rec_df = pd.concat(per_task_receptor, ignore_index=True)
    per_task_rec_df.to_csv(out_dir/"shap"/"receptor"/"per_task_receptor_shap_share.csv", index=False)

    # overall: task별 share 평균(동일 가중)
    overall = per_task_rec_df.groupby("receptor")["share"].mean().reset_index().sort_values("share", ascending=False)
    overall.to_csv(out_dir/"shap"/"receptor"/"overall_receptor_shap_share.csv", index=False)

    # overall plot
    fig, ax = plt.subplots(figsize=(7,4))
    _bar(ax, overall["receptor"].tolist(), overall["share"].tolist(), "Receptor importance (mean SHAP share)")
    fig.tight_layout(); fig.savefig(out_dir/"plots"/"receptor_shap_share_bar.png", dpi=200); plt.close(fig)

    return per_task_rec_df, overall

def main():
    ap = argparse.ArgumentParser(description="Analyze XGB Tox21 models: AUROC/AUPRC + SHAP + receptor importance")
    ap.add_argument("--models-dir", required=True, help="e.g., /home/ssm-user/example/models_xgb_ver5")
    ap.add_argument("--out-dir", required=False, help="Result folder (default: <models-dir>/analysis)")
    ap.add_argument("--topk-features", type=int, default=30)
    ap.add_argument("--no-shap", action="store_true", help="Skip SHAP (faster)")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (models_dir/"analysis")
    _ensure_dirs(out_dir)

    print(f"[+] Compute AUROC/AUPRC from OOF: {models_dir}")
    per_task, micro = compute_metrics(models_dir, out_dir)
    print(f"    - Saved per-assay metrics to {out_dir/'metrics'/'assay_metrics_from_oof.csv'}")
    print(f"    - Micro ROC/PR plots: {out_dir/'plots'}")

    # Gain 기반 receptor 영향력(훈련 시 저장값 사용)
    print("[+] Aggregate Gain-based receptor importance ...")
    gain_overall = compute_gain_by_receptor(models_dir, out_dir)
    if not gain_overall.empty:
        print(f"    - Gain overall receptor ranking saved to {out_dir/'gain_receptor'/'overall_gain_receptor_share.csv'}")

    # SHAP (선호: 모델-데이터 일치 기준)
    if not args.no_shap:
        print("[+] Compute SHAP attributions (pred_contribs=True) ...")
        per_task_rec_df, shap_overall = compute_shap(models_dir, out_dir, topk=args.topk_features)
        if not shap_overall.empty:
            print(f"    - SHAP overall receptor ranking saved to {out_dir/'shap'/'receptor'/'overall_receptor_shap_share.csv'}")
    else:
        print("[!] Skipped SHAP (use --no-shap to skip, default is compute)")

    # 테이블 모아두기(보고용)
    tables_dir = out_dir/"tables"
    tables_dir.mkdir(exist_ok=True)
    # per-assay top-line만 별도 테이블
    if not per_task.empty:
        slim = per_task[["task","n","pos","neg","auc","auprc"]].sort_values("auc", ascending=False)
        slim.to_csv(tables_dir/"assay_topline.csv", index=False)

    print("\n[Done] Results saved under:", out_dir.as_posix())

if __name__ == "__main__":
    main()
