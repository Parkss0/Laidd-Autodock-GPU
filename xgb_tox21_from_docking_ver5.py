#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost multi-endpoint trainer from receptor-docking/PLIP features (ver5/run1safe)
- Merges 11 per-receptor feature CSVs (features__*.csv) by compound ID.
- Trains one XGBoost classifier per endpoint (multi-label binary classification).
- Handles missing values, constant columns, and saves:
    * models/<TASK>.json
    * metrics/<TASK>_cv_metrics.csv and summary.json
    * importances/<TASK>_feature_importance.csv
    * oof/oof_predictions.csv (stacking/metalearner-friendly)
- Provides `predict` mode to score new feature directories with the saved models.
- Publication-friendly: logs config, seed, class imbalance, AUROC/AUPRC and CV protocol.

Assumptions
- Each features__<RECEPTOR>.csv contains an identifier column (e.g., ligand_id / compound_id / Sample ID).
- Label file (optional) has one identifier column matching the features' ID and N endpoint columns (0/1/NaN/-1/Active/Inactive).
- If labels are embedded in the merged features (rare), they will be auto-detected unless --labels forces an external file.

Author: ChatGPT (GPT-5 Thinking)
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             brier_score_loss)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state

try:
    import xgboost as xgb
except Exception as e:
    print("[ERROR] xgboost is required. Try: pip install xgboost", file=sys.stderr)
    raise

SEED = 42

# ------------------------------
# Utilities
# ------------------------------

ID_CANDIDATES = [
    "compound_id", "ligand_id", "ligand", "molid",
    "molecule_id", "sample id", "sample_id", "ncgc id", "ncgcid",
    "id"
]

LABEL_NAME_HINTS = [
    # Typical Tox21 task name fragments (script will auto-detect any columns
    # not part of features and treat them as labels when plausible)
    "NR-", "SR-", "tox", "label", "endpoint"
]

def slug(s: str) -> str:
    """Filesystem-safe slug."""
    s = str(s).strip()
    s = re.sub(r"[^\w\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def guess_id_column(df: pd.DataFrame) -> Optional[str]:
    cols = [c for c in df.columns]
    lower_map = {c.lower(): c for c in cols}
    for key in ID_CANDIDATES:
        if key in lower_map:
            return lower_map[key]
    # Heuristic fallbacks
    for c in cols:
        if c.lower().endswith("_id"):
            return c
    # If first column looks like an ID (object/string and mostly unique)
    c0 = cols[0]
    if df[c0].dtype == object and df[c0].nunique(dropna=True) > df.shape[0] * 0.5:
        return c0
    return None

def read_feature_file(p: Path) -> Tuple[pd.DataFrame, str]:
    """Read a feature CSV (auto sep) and return (df, receptor_name)."""
    df = pd.read_csv(p, sep=None, engine="python", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    # receptor name from file
    m = re.search(r"features__(.+?)\.csv$", p.name)
    receptor = m.group(1) if m else p.stem.replace("features__", "")
    receptor = slug(receptor)
    return df, receptor

def extract_id(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    id_col = guess_id_column(df)
    if id_col is None:
        raise ValueError(f"Cannot find ID column in columns={df.columns.tolist()}")
    # normalize ID column name
    if id_col != "compound_id":
        df = df.rename(columns={id_col: "compound_id"})
    return df, "compound_id"

def split_label_columns(df: pd.DataFrame, force_label_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split label columns from df. If force_label_cols provided, use those explicitly.
    Else, auto-detect label-like columns (non-numeric excluded)."""
    if force_label_cols:
        lab_cols = [c for c in force_label_cols if c in df.columns]
    else:
        lab_cols = []
        for c in df.columns:
            lc = c.lower()
            # ignore known non-label meta
            if c == "compound_id" or lc in ("smiles", "receptor", "run"):
                continue
            # consider as potential label if it contains hints and relatively low cardinality (<= 3 unique non-nan values)
            if any(h.lower() in lc for h in LABEL_NAME_HINTS):
                nun = df[c].dropna().nunique()
                if nun <= 5:
                    lab_cols.append(c)
    lab_cols = sorted(set(lab_cols))
    labels = df[["compound_id"] + lab_cols].copy() if lab_cols else pd.DataFrame(columns=["compound_id"])
    feats = df.drop(columns=lab_cols, errors="ignore")
    return feats, labels

def prefix_columns(df: pd.DataFrame, receptor: str, skip: List[str]) -> pd.DataFrame:
    new_cols = {}
    for c in df.columns:
        if c in skip:
            continue
        new_cols[c] = f"{receptor}__{c}"
    return df.rename(columns=new_cols)

def coerce_numeric(df: pd.DataFrame, skip: List[str]) -> pd.DataFrame:
    for c in df.columns:
        if c in skip:
            continue
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif str(df[c].dtype).startswith("bool"):
            df[c] = df[c].astype(int)
    return df

def drop_useless_columns(df: pd.DataFrame, skip: List[str]) -> pd.DataFrame:
    drop_cols = []
    for c in df.columns:
        if c in skip:
            continue
        s = df[c]
        if s.isna().all():
            drop_cols.append(c)
        else:
            nun = s.nunique(dropna=True)
            if nun <= 1:
                drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df

def merge_feature_dir(features_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge all features__*.csv in a directory. Return (merged_features, embedded_labels [may be empty])."""
    files = sorted(features_dir.glob("features__*.csv"))
    if not files:
        raise FileNotFoundError(f"No features__*.csv files in {features_dir}")

    merged = None
    labels_all = []

    for p in files:
        df, receptor = read_feature_file(p)
        df, idcol = extract_id(df)
        # Attempt to split any embedded labels
        df, labs = split_label_columns(df)
        if not labs.empty:
            labs = labs.copy()
            # keep only unique per compound
            labs = labs.groupby("compound_id").first().reset_index()
            labels_all.append(labs)

        skip = ["compound_id", "smiles"]
        df = coerce_numeric(df, skip=skip)
        df = drop_useless_columns(df, skip=skip)
        df = prefix_columns(df, receptor, skip=skip)

        if merged is None:
            merged = df
        else:
            merged = pd.merge(merged, df, on="compound_id", how="outer")

    # combine all embedded labels (outer join on compound_id)
    embedded_labels = None
    if labels_all:
        embedded_labels = labels_all[0]
        for labs in labels_all[1:]:
            embedded_labels = pd.merge(embedded_labels, labs, on="compound_id", how="outer")
    else:
        embedded_labels = pd.DataFrame(columns=["compound_id"])

    # Final cleanups
    merged = merged.drop_duplicates(subset=["compound_id"], keep="first")
    return merged, embedded_labels

def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize labels to {0,1,NaN} and enforce column names."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["compound_id"])
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    # normalize textual labels
    map_dict = {
        "active": 1, "inactive": 0,
        "act": 1, "inact": 0,
        "pos": 1, "neg": 0
    }
    for c in out.columns:
        if c == "compound_id":
            continue
        # coerce to str to match mapping; then back to numeric
        out[c] = out[c].replace(map_dict).replace({-1: np.nan}).astype("float64")
    return out

def pick_tree_method(use_gpu: bool) -> str:
    if use_gpu:
        try:
            booster = xgb.XGBClassifier(tree_method="gpu_hist", n_estimators=1, max_depth=1)
            # A light fit test may fail in some envs; avoid fitting — just return the method.
            return "gpu_hist"
        except Exception:
            pass
    return "hist"

@dataclass
class TrainConfig:
    features_dir: str
    out_dir: str
    labels_path: Optional[str] = None
    id_column: str = "compound_id"
    use_gpu: bool = True
    n_splits: int = 5
    random_state: int = SEED
    fast: bool = True  # fast search vs thorough
    early_stopping_rounds: int = 100

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

# ------------------------------
# Training core
# ------------------------------

def build_feature_matrix(features_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X, embedded_labels = merge_feature_dir(features_dir)
    # Drop non-numeric leftover columns except ID/SMILES
    keep = ["compound_id", "smiles"]
    for c in list(X.columns):
        if c in keep:
            continue
        if not (np.issubdtype(X[c].dtype, np.number)):
            # try coercion; else drop
            X[c] = pd.to_numeric(X[c], errors="coerce")
    # Final useless col drop
    X = drop_useless_columns(X, skip=keep)
    return X, embedded_labels

def prepare_labels(labels_path: Optional[Path], embedded: pd.DataFrame) -> pd.DataFrame:
    if labels_path is not None and labels_path.exists():
        lab = pd.read_csv(labels_path, sep=None, engine="python", skipinitialspace=True)
        lab.columns = [c.strip() for c in lab.columns]
        idcol = guess_id_column(lab)
        if idcol and idcol != "compound_id":
            lab = lab.rename(columns={idcol: "compound_id"})
        lab = lab.drop_duplicates(subset=["compound_id"], keep="first")
    else:
        lab = embedded.copy()
    lab = clean_labels(lab)
    if lab.empty or lab.shape[1] <= 1:
        raise ValueError("No label columns found. Provide --labels pointing to a CSV with endpoints.")
    return lab

def infer_tasks(labels_df: pd.DataFrame) -> List[str]:
    tasks = [c for c in labels_df.columns if c != "compound_id"]
    if not tasks:
        raise ValueError("No endpoint columns detected in labels.")
    return tasks

def compute_scale_pos_weight(y: np.ndarray) -> float:
    # Avoid div by zero; clamp
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos == 0.0:
        return 1.0
    return max(1.0, neg / pos)

def cv_param_space(fast: bool) -> Dict[str, List]:
    if fast:
        return {
            "n_estimators": [600, 900],
            "max_depth": [5, 7],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.7, 0.9],
            "min_child_weight": [1, 5],
            "reg_lambda": [1.0, 5.0],
        }
    else:
        return {
            "n_estimators": [600, 900, 1200],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_lambda": [1.0, 3.0, 6.0, 10.0],
        }

def product_dict(d: Dict[str, List]) -> List[Dict]:
    """Cartesian product of parameter dict (manual grid to avoid extra deps)."""
    from itertools import product
    keys = list(d.keys())
    combos = []
    for values in product(*[d[k] for k in keys]):
        combos.append({k: v for k, v in zip(keys, values)})
    return combos

def train_single_task(task: str,
                      X: pd.DataFrame,
                      y: pd.Series,
                      out_dir: Path,
                      tree_method: str,
                      n_splits: int,
                      rs: int,
                      fast: bool,
                      early_stopping_rounds: int) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    """Train one endpoint with CV and return (oof_df, summary_metrics, importances_df)."""
    rng = check_random_state(rs)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)

    # Preprocess pipeline: median imputation for numerics
    feature_cols = [c for c in X.columns if c != "compound_id"]
    X_num = X[feature_cols].copy()
    # Ensure numpy array for speed
    X_np = X_num.values

    oof_pred = np.full(shape=(X.shape[0],), fill_value=np.nan, dtype=float)
    fold_metrics = []
    scales = []

    param_grid = cv_param_space(fast)
    combos = product_dict(param_grid)

    # Limit combos if too many (fast safety)
    max_combos = 16 if fast else 48
    if len(combos) > max_combos:
        combos = [combos[i] for i in rng.choice(len(combos), size=max_combos, replace=False)]

    best_auc = -np.inf
    best_params = None

    # Simple median imputer
    imputer = SimpleImputer(strategy="median")
    X_np = imputer.fit_transform(X_np)

    for params in combos:
        cv_scores = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_np, y.values)):
            X_tr, X_va = X_np[tr_idx], X_np[va_idx]
            y_tr, y_va = y.values[tr_idx], y.values[va_idx]

            spw = compute_scale_pos_weight(y_tr)
            clf = xgb.XGBClassifier(
                objective="binary:logistic",
                tree_method=tree_method,
                eval_metric=["auc", "aucpr"],
                random_state=rs,
                n_jobs=0,
                scale_pos_weight=spw,
                **params
            )
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
                early_stopping_rounds=early_stopping_rounds
            )
            va_pred = clf.predict_proba(X_va)[:, 1]
            auc = roc_auc_score(y_va, va_pred)
            cv_scores.append(auc)

        mean_auc = float(np.mean(cv_scores))
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params

    # Refit with best params and fill OOF
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_np, y.values)):
        X_tr, X_va = X_np[tr_idx], X_np[va_idx]
        y_tr, y_va = y.values[tr_idx], y.values[va_idx]

        spw = compute_scale_pos_weight(y_tr)
        clf = xgb.XGBClassifier(
            objective="binary:logistic",
            tree_method=tree_method,
            eval_metric=["auc", "aucpr"],
            random_state=rs + fold,
            n_jobs=0,
            scale_pos_weight=spw,
            **best_params
        )
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
            early_stopping_rounds=early_stopping_rounds
        )

        va_pred = clf.predict_proba(X_va)[:, 1]
        oof_pred[va_idx] = va_pred

        # Collect metrics
        m = {
            "fold": fold,
            "pos_train": int((y_tr == 1).sum()),
            "neg_train": int((y_tr == 0).sum()),
            "scale_pos_weight": float(compute_scale_pos_weight(y_tr)),
            "roc_auc": float(roc_auc_score(y_va, va_pred)),
            "auprc": float(average_precision_score(y_va, va_pred)),
            "brier": float(brier_score_loss(y_va, va_pred)),
        }
        # Thresholded metrics at 0.5
        y_hat = (va_pred >= 0.5).astype(int)
        m["f1"] = float(f1_score(y_va, y_hat))
        m["precision"] = float(precision_score(y_va, y_hat, zero_division=0))
        m["recall"] = float(recall_score(y_va, y_hat, zero_division=0))
        fold_metrics.append(m)

    metrics_df = pd.DataFrame(fold_metrics)
    summary = {
        "task": task,
        "n_samples": int(len(y)),
        "prevalence": float((y == 1).mean()),
        "cv_mean_auc": float(metrics_df["roc_auc"].mean()),
        "cv_std_auc": float(metrics_df["roc_auc"].std()),
        "cv_mean_auprc": float(metrics_df["auprc"].mean()),
        "best_params": best_params,
        "tree_method": tree_method,
        "n_splits": n_splits,
        "seed": rs,
    }

    # Train final model on full data (with best_params)
    spw_full = compute_scale_pos_weight(y.values)
    final_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        tree_method=tree_method,
        eval_metric=["auc", "aucpr"],
        random_state=rs,
        n_jobs=0,
        scale_pos_weight=spw_full,
        **best_params
    )
    final_clf.fit(X_np, y.values, verbose=False)

    # Save artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)
    (out_dir / "metrics").mkdir(exist_ok=True)
    (out_dir / "importances").mkdir(exist_ok=True)
    (out_dir / "oof").mkdir(exist_ok=True)

    model_path = out_dir / "models" / f"{slug(task)}.json"
    final_clf.save_model(model_path.as_posix())

    metrics_df.to_csv(out_dir / "metrics" / f"{slug(task)}_cv_metrics.csv", index=False)
    with open(out_dir / "metrics" / f"{slug(task)}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Feature importance (gain)
    booster = final_clf.get_booster()
    fmap = {i: col for i, col in enumerate(X_num.columns)}
    imp_gain = booster.get_score(importance_type="gain")
    rows = []
    for k, v in imp_gain.items():
        if k.startswith("f"):
            idx = int(k[1:])
            rows.append((fmap.get(idx, f"f{idx}"), float(v)))
        else:
            rows.append((k, float(v)))
    imp_df = pd.DataFrame(rows, columns=["feature", "gain"]).sort_values("gain", ascending=False)
    imp_df.to_csv(out_dir / "importances" / f"{slug(task)}_feature_importance.csv", index=False)

    # OOF dataframe
    oof_df = pd.DataFrame({
        "compound_id": X["compound_id"].values,
        task: oof_pred
    })

    # Save imputer & feature list (for predict compatibility)
    aux = {
        "feature_columns": feature_cols,
        "imputer_strategy": "median"
    }
    with open(out_dir / "models" / f"{slug(task)}_features.json", "w", encoding="utf-8") as f:
        json.dump(aux, f, indent=2, ensure_ascii=False)

    return oof_df, summary, imp_df

# ------------------------------
# CLI
# ------------------------------

def train_main(cfg: TrainConfig):
    features_dir = Path(cfg.features_dir)
    out_dir = Path(cfg.out_dir)
    labels_path = Path(cfg.labels_path) if cfg.labels_path else None

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build X and (embedded) labels
    print(f"[LOAD] Merging feature CSVs from: {features_dir}")
    X, embedded_labels = build_feature_matrix(features_dir)
    print(f"[OK] Merged feature shape: {X.shape} (rows=compounds)")

    # Prepare labels
    labels_df = prepare_labels(labels_path, embedded_labels)
    tasks = infer_tasks(labels_df)
    print(f"[LABELS] Tasks detected: {tasks}")

    # Merge features and labels on compound_id
    df = pd.merge(X, labels_df, on="compound_id", how="inner")
    print(f"[MERGE] Joined features+labels shape: {df.shape}")

    # Split X / per-task y (drop rows with NaN label per task)
    tree_method = pick_tree_method(cfg.use_gpu)
    print(f"[XGB] tree_method={tree_method}  n_splits={cfg.n_splits}  fast={cfg.fast}")

    # Persist merged matrices
    (out_dir / "cache").mkdir(exist_ok=True)
    X.to_csv(out_dir / "cache" / "X_merged.csv", index=False)
    labels_df.to_csv(out_dir / "cache" / "labels.csv", index=False)

    oof_all = df[["compound_id"]].copy()
    summaries = {}

    for task in tasks:
        sub = df[["compound_id"] + [c for c in X.columns if c != "compound_id"] + [task]].copy()
        sub = sub.dropna(subset=[task])
        print(f"[TASK] {task}: n={len(sub)}  pos={int(sub[task].sum())}  neg={int((1-sub[task]).sum())}")
        oof_df, summary, _imp = train_single_task(
            task=task,
            X=sub[X.columns],
            y=sub[task].astype(int),
            out_dir=out_dir,
            tree_method=tree_method,
            n_splits=cfg.n_splits,
            rs=cfg.random_state,
            fast=cfg.fast,
            early_stopping_rounds=cfg.early_stopping_rounds
        )
        summaries[task] = summary
        # merge OOF
        oof_all = pd.merge(oof_all, oof_df, on="compound_id", how="left")

    oof_all.to_csv(out_dir / "oof" / "oof_predictions.csv", index=False)
    with open(out_dir / "metrics" / "all_tasks_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    # Save config
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        f.write(cfg.to_json())

    print(f"[DONE] Artifacts saved under: {out_dir.as_posix()}")
    print(" - models/*.json (per-task)")
    print(" - metrics/*summary.json, *_cv_metrics.csv")
    print(" - importances/*_feature_importance.csv")
    print(" - oof/oof_predictions.csv")
    print(" - cache/X_merged.csv, cache/labels.csv")

def predict_main(features_dir: Path, models_dir: Path, out_path: Path, use_gpu: bool = True):
    # Rebuild feature matrix (must correspond to training schema; missing columns are imputed)
    X, _ = build_feature_matrix(features_dir)
    feature_cols_union = set()
    tasks = []

    # Discover tasks by model files
    for p in sorted((models_dir).glob("*.json")):
        if p.name.endswith("_features.json"):
            continue
        task = p.stem
        tasks.append(task)
        # load feature list per task
        auxp = models_dir / f"{task}_features.json"
        if auxp.exists():
            aux = json.loads(auxp.read_text(encoding="utf-8"))
            feature_cols_union.update(aux["feature_columns"])

    feature_cols_union = [c for c in X.columns if c in feature_cols_union]
    if not feature_cols_union:
        raise RuntimeError("No feature columns shared with trained models. Check your feature directory.")

    tree_method = pick_tree_method(use_gpu)

    # Prepare matrix
    Xnum = X[feature_cols_union].copy()
    Xnum = SimpleImputer(strategy="median").fit_transform(Xnum.values)

    pred_df = pd.DataFrame({"compound_id": X["compound_id"].values})
    for task in tasks:
        model_path = models_dir / f"{task}.json"
        clf = xgb.XGBClassifier(objective="binary:logistic", tree_method=tree_method)
        clf.load_model(model_path.as_posix())
        yhat = clf.predict_proba(Xnum)[:, 1]
        pred_df[task] = yhat

    pred_df.to_csv(out_path, index=False)
    print(f"[PREDICT] Wrote predictions: {out_path}")

def parse_args():
    ap = argparse.ArgumentParser(description="XGBoost multi-endpoint trainer from receptor features (ver5/run1safe)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_train = sub.add_parser("train", help="Train models (one-vs-rest per endpoint)")
    ap_train.add_argument("--features-dir", type=str, required=True, help="Directory with features__*.csv (11 files)")
    ap_train.add_argument("--out-dir", type=str, required=True, help="Output directory for models/metrics/oof")
    ap_train.add_argument("--labels", type=str, default=None, help="CSV with labels (first ID col + N endpoints). If omitted, try embedded labels.")
    ap_train.add_argument("--no-gpu", action="store_true", help="Force CPU (hist)")
    ap_train.add_argument("--splits", type=int, default=5, help="StratifiedKFold splits")
    ap_train.add_argument("--seed", type=int, default=SEED, help="Random seed")
    ap_train.add_argument("--thorough", action="store_true", help="Use a broader hyperparam grid (slower)")
    ap_train.add_argument("--early-stop", type=int, default=100, help="early_stopping_rounds")

    ap_pred = sub.add_parser("predict", help="Predict with saved models")
    ap_pred.add_argument("--features-dir", type=str, required=True, help="Directory with features__*.csv for scoring")
    ap_pred.add_argument("--models-dir", type=str, required=True, help="Directory containing trained models/*.json and *_features.json")
    ap_pred.add_argument("--out", type=str, required=True, help="Output CSV path for predictions")
    ap_pred.add_argument("--no-gpu", action="store_true", help="Force CPU (hist)")

    return ap.parse_args()

def main():
    args = parse_args()
    if args.cmd == "train":
        cfg = TrainConfig(
            features_dir=args.features_dir,
            out_dir=args.out_dir,
            labels_path=args.labels,
            use_gpu=(not args.no_gpu),
            n_splits=int(args.splits),
            random_state=int(args.seed),
            fast=(not args.thorough),
            early_stopping_rounds=int(args.early_stop),
        )
        print("[CONFIG]")
        print(cfg.to_json())
        train_main(cfg)
    elif args.cmd == "predict":
        predict_main(
            features_dir=Path(args.features_dir),
            models_dir=Path(args.models_dir),
            out_path=Path(args.out),
            use_gpu=(not args.no_gpu)
        )
    else:
        raise SystemExit("Unknown command")

if __name__ == "__main__":
    main()

# ------------------------------
# Quick Usage (examples)
# ------------------------------
#
# 1) Train on AWS (all 10k as training; labels CSV required unless embedded)
#    python xgb_tox21_from_docking_ver5.py train \
#      --features-dir "/home/ssm-user/example/features_ver5_plip_run1safe" \
#      --out-dir "/home/ssm-user/example/models_xgb_ver5" \
#      --labels "/home/ssm-user/example/labels/tox21_labels_12.csv" \
#      --splits 5 --seed 42 --early-stop 100
#
#    (Omit --labels to auto-use embedded labels if present.)
#    Add --thorough for a wider param search; add --no-gpu to force CPU.
#
# 2) Predict later on a new batch (same 11 receptors, same feature set schema)
#    python xgb_tox21_from_docking_ver5.py predict \
#      --features-dir "/home/ssm-user/example/features_ver5_plip_run1safe" \
#      --models-dir "/home/ssm-user/example/models_xgb_ver5/models" \
#      --out "/home/ssm-user/example/models_xgb_ver5/preds_train.csv"
#
# Artifacts you’ll get:
#   models/*.json, models/*_features.json
#   metrics/*_cv_metrics.csv, *summary.json, metrics/all_tasks_summary.json
#   importances/*_feature_importance.csv
#   oof/oof_predictions.csv
#
# Notes:
# - The script auto-detects the ID column. If your ID is unusual, rename to 'compound_id' first.
# - Labels are normalized (Active/Inactive → 1/0; -1 → NaN and dropped per-task).
# - The merging is OUTER across receptors to "maximize usable rows"; remaining NaNs imputed with median.
# - Scale_pos_weight is applied per fold to counter class imbalance; metrics include AUROC/AUPRC.
# - Final model retrains on all rows after CV with best params.
# - For meta-learning, use oof/oof_predictions.csv (per-task probabilities).

