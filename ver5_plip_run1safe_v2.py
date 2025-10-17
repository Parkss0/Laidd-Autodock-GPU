#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, gzip
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

# ====== CONFIG ======
BASE        = "/home/ssm-user/example"
SMILES_CSV  = f"{BASE}/tox21_647_labels_clean_minus_intersection.csv"
REC_TAG     = "PTGS2_COX2_3LN1_A_receptor"   # ← 리셉터 바꿀 때 여기만 수정
REC_DIR     = f"{BASE}/atdk_gpu/docking/090909_val2/{REC_TAG}"
GRID_DIR    = f"{BASE}/atdk_gpu/grids/p2rank_selected_23/{REC_TAG}"
OUT_DIR     = f"{BASE}/features_ver5_plip_run1safe_val3_v2"
SCHEMA_VER  = "v5_plip_run1safe_v2"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV     = os.path.join(OUT_DIR, f"features__{REC_TAG}.csv")

# ====== helpers ======
real = os.path.realpath
def list_files(patts):
    fs=[]; [fs.extend(glob.glob(p)) for p in patts]
    return sorted(set(real(x) for x in fs))

def load_smiles_and_labels(smiles_csv):
    df = pd.read_csv(smiles_csv, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}
    sid = low.get("sample id") or low.get("sample_id") or df.columns[0]
    smi = low.get("smiles")
    if smi is None:
        raise ValueError("SMILES CSV에 'SMILES' 컬럼이 필요합니다.")
    label_cols=[]
    for c in df.columns:
        if c in [sid, smi]: continue
        v = pd.to_numeric(df[c], errors="coerce").dropna().unique()
        if len(v)>0 and set(v).issubset({0,1}):
            label_cols.append(c)
    mp={}
    for _, r in df.iterrows():
        rid = str(r[sid]).strip()
        sm  = str(r[smi]).strip() if pd.notna(r[smi]) else None
        labels = {c:int(r[c]) if pd.notna(r[c]) else np.nan for c in label_cols}
        mp[rid] = (sm, labels)
    return mp, label_cols

def parse_dlg(dlg_path):
    e_pat  = re.compile(r"Estimated Free Energy of Binding\s*=\s*(-?\d+\.\d+)\s*kcal/mol", re.I)
    t_pat1 = re.compile(r"Number of torsions\s*:\s*(\d+)", re.I)
    t_pat2 = re.compile(r"Number of rotatable bonds\s*:\s*(\d+)", re.I)
    t_pat3 = re.compile(r"\bTORSDOF\s+(\d+)", re.I)
    sm_pat = re.compile(r"REMARK SMILES\s+(\S+)", re.I)
    Es, nt, smiles_in_dlg = [], None, None
    op = gzip.open if dlg_path.endswith(".gz") else open
    with op(dlg_path, "rt", errors="ignore") as f:
        for line in f:
            m = e_pat.search(line)
            if m:
                try: Es.append(float(m.group(1)))
                except: pass
            if nt is None:
                m1 = t_pat1.search(line) or t_pat2.search(line) or t_pat3.search(line)
                if m1:
                    try: nt = int(m1.group(1))
                    except: pass
            if smiles_in_dlg is None:
                m2 = sm_pat.search(line)
                if m2:
                    smiles_in_dlg = m2.group(1).strip()
    return Es, nt, smiles_in_dlg

def read_predictions_or_gpf(grid_dir):
    preds = sorted(glob.glob(os.path.join(grid_dir, "*predictions*.csv")))
    if preds:
        df = pd.read_csv(preds[0], sep=None, engine="python")
        norm = {c.lower().strip().replace(".", "_"): c for c in df.columns}
        need = ["rank","center_x","center_y","center_z"]
        miss = [k for k in need if k not in norm]
        if miss:
            raise ValueError(f"predictions csv missing {miss}; columns={list(df.columns)}")
        row = df.sort_values(norm["rank"]).iloc[0]
        cx = float(pd.to_numeric(row[norm["center_x"]], errors="coerce"))
        cy = float(pd.to_numeric(row[norm["center_y"]], errors="coerce"))
        cz = float(pd.to_numeric(row[norm["center_z"]], errors="coerce"))
        return np.array([cx,cy,cz], float), "predictions", preds[0]
    gpf = os.path.join(grid_dir, "receptor.gpf")
    if os.path.exists(gpf):
        center=None
        with open(gpf, "rt", errors="ignore") as f:
            for line in f:
                if line.strip().lower().startswith("gridcenter"):
                    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
                    if len(nums) >= 3:
                        center = np.array([float(nums[0]), float(nums[1]), float(nums[2])], float)
                        break
        if center is not None:
            return center, "gpf", gpf
    raise FileNotFoundError("No predictions.csv nor receptor.gpf")

def rdkit_props(smiles, cache):
    if smiles in cache: return cache[smiles]
    if not smiles:
        res={}
    else:
        m=Chem.MolFromSmiles(smiles)
        if m is None: res={}
        else:
            res={
                "n_heavy": m.GetNumHeavyAtoms(),
                "MW": Descriptors.MolWt(m),
                "logP": Descriptors.MolLogP(m),
                "TPSA": rdMolDescriptors.CalcTPSA(m),
                "HBD": Lipinski.NumHDonors(m),
                "HBA": Lipinski.NumHAcceptors(m),
                "RotBonds": Lipinski.NumRotatableBonds(m),
                "FormalCharge": int(sum(a.GetFormalCharge() for a in m.GetAtoms())),
                "RingCount": Lipinski.RingCount(m),
                "AromaticRingCount": rdMolDescriptors.CalcNumAromaticRings(m),
                "ECFP4_bitcount": int(rdMolDescriptors.GetMorganFingerprintAsBitVect(m,2,2048).GetNumOnBits()),
            }
    cache[smiles]=res; return res

def sdf_centroid_firstconf(sdf_path):
    if not sdf_path or not os.path.exists(sdf_path): return None
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None: return None
    conf = mol.GetConformer()
    pts=[]
    for a in mol.GetAtoms():
        if a.GetAtomicNum()==1: continue
        p=conf.GetAtomPosition(a.GetIdx())
        pts.append([p.x,p.y,p.z])
    return np.mean(np.array(pts), axis=0) if pts else None

def plip_read(path):
    return pd.read_csv(path, sep=None, engine="python") if path and os.path.exists(path) else None

def plip_centroid(df):
    if df is None or "LIGCOO" not in df.columns: return None
    nums = lambda s: re.findall(r"[-+]?\d+\.?\d*", str(s))
    pts=[]
    for v in df["LIGCOO"].dropna():
        arr = nums(v)
        if len(arr)>=3:
            pts.append([float(arr[0]), float(arr[1]), float(arr[2])])
    return np.mean(np.array(pts), axis=0) if pts else None

def residue_class(res):
    acidic={"ASP","GLU"}; basic={"ARG","LYS","HIS"}; polar={"SER","THR","ASN","GLN","TYR"}
    arom={"PHE","TYR","TRP","HIS"}; hydro={"ALA","VAL","LEU","ILE","PRO","MET"}
    r=str(res).upper()
    if r in acidic: return "acidic"
    if r in basic:  return "basic"
    if r in polar:  return "polar"
    if r in arom:   return "aromatic"
    if r in hydro:  return "hydrophobic"
    return "other"

def _norm_interaction_label(s):
    t = re.sub(r"[^A-Za-z]", "", str(s)).upper()
    repl = {
        "PISTACKING":"PISTACKING",
        "PISTACK":"PISTACKING",
        "PIINTERACTION":"PISTACKING",
        "PICATION":"PICATION",
        "SALTSBRIDGE":"SALTBRIDGE",
        "SALTBRIDGE":"SALTBRIDGE",
        "HYDROPHOBIC":"HYDROPHOBIC",
        "WATERBRIDGE":"WATERBRIDGE",
        "HALOGEN":"HALOGEN",
        "METAL":"METAL",
        "HBOND":"HBOND",
    }
    return repl.get(t, t)

def plip_features(df, n_heavy):
    F={}
    if df is None or len(df)==0:
        for k in ["n_hbond","n_hbond_don","n_hbond_acc","n_hydrophob","n_pistack","n_halogen",
                  "n_saltbridge","n_waterbridge","n_metal","total_contacts",
                  "hbond_dist_mean","hbond_dist_min","hbond_angle_mean",
                  "contact_per_heavy","frac_aromatic_contacts",
                  "contacts_to__acidic","contacts_to__basic","contacts_to__polar",
                  "contacts_to__aromatic","contacts_to__hydrophobic","contacts_to__other",
                  "plip_rows"]:
            F[k]=0 if k in {"n_hbond","n_hbond_don","n_hbond_acc","n_hydrophob","n_pistack","n_halogen","n_saltbridge","n_waterbridge","n_metal","total_contacts","contacts_to__acidic","contacts_to__basic","contacts_to__polar","contacts_to__aromatic","contacts_to__hydrophobic","contacts_to__other","plip_rows"} else np.nan
        return F
    hb_dcol = "DIST_H-A" if "DIST_H-A" in df.columns else None
    hb_acol = "DON_ANGLE" if "DON_ANGLE" in df.columns else None
    n_hbond = 0
    if hb_dcol:
        dist = pd.to_numeric(df[hb_dcol], errors="coerce")
        mask = dist.notna() & (dist <= 3.5)
        ang=None
        if hb_acol in df.columns:
            ang = pd.to_numeric(df[hb_acol], errors="coerce")
            mask = mask & (ang >= 120)
        n_hbond = int(mask.sum())
        F["hbond_dist_mean"] = float(dist[mask].mean()) if n_hbond else np.nan
        F["hbond_dist_min"]  = float(dist[mask].min())  if n_hbond else np.nan
        F["hbond_angle_mean"]= float(ang[mask].mean())  if (ang is not None and n_hbond) else np.nan
        pdon = "PROTISDON" if "PROTISDON" in df.columns else None
        if pdon:
            to_bool = lambda x: str(x).strip().lower() in {"1","true","t","y","yes"}
            n_acc = int(df.loc[mask, pdon].map(to_bool).sum())
            F["n_hbond_acc"] = n_acc
            F["n_hbond_don"] = int(n_hbond - n_acc)
        else:
            F["n_hbond_acc"] = 0
            F["n_hbond_don"] = 0
    else:
        F["hbond_dist_mean"]=np.nan; F["hbond_dist_min"]=np.nan; F["hbond_angle_mean"]=np.nan
        F["n_hbond_acc"]=0; F["n_hbond_don"]=0
    F["n_hbond"]=int(n_hbond)
    s = pd.Series(dtype=int)
    if "TYPE" in df.columns:
        s = df["TYPE"].astype(str).map(_norm_interaction_label).value_counts()
    elif "BOND" in df.columns:
        s = df["BOND"].astype(str).map(_norm_interaction_label).value_counts()
    F["n_hydrophob"]  = int(s.get("HYDROPHOBIC", 0))
    F["n_pistack"]    = int(s.get("PISTACKING", 0) + s.get("PICATION", 0))
    F["n_halogen"]    = int(s.get("HALOGEN", 0))
    F["n_saltbridge"] = int(s.get("SALTBRIDGE", 0))
    F["n_waterbridge"]= int(s.get("WATERBRIDGE", 0))
    F["n_metal"]      = int(s.get("METAL", 0))
    rcol = "RESTYPE" if "RESTYPE" in df.columns else ("RESNAME" if "RESNAME" in df.columns else None)
    cls = df[rcol].map(residue_class).value_counts() if rcol else pd.Series(dtype=int)
    for c in ["acidic","basic","polar","aromatic","hydrophobic","other"]:
        F[f"contacts_to__{c}"] = int(cls.get(c,0))
    total = int(len(df))
    F["total_contacts"]        = total
    F["contact_per_heavy"]     = (total / n_heavy) if (n_heavy and total>0) else np.nan
    F["frac_aromatic_contacts"]= (F["n_pistack"]/total) if total>0 else np.nan
    F["plip_rows"]             = total
    return F

def preflight(center, src, srcfile):
    plip_n = len(list_files([
        os.path.join(REC_DIR, "plip_batch",  "*", "intrxn*.csv"),
        os.path.join(REC_DIR, "plip_batch",  "*", "intrxn*.csv.gz"),
        os.path.join(REC_DIR, "plip_batch",  "*", "interactions*.csv"),
        os.path.join(REC_DIR, "plip_batch",  "*", "interactions*.csv.gz"),
        os.path.join(REC_DIR, "plip_batch2", "*", "intrxn*.csv"),
        os.path.join(REC_DIR, "plip_batch2", "*", "interactions*.csv"),
    ]))
    dlg_n = len(list_files([os.path.join(REC_DIR,"poses","*.dlg"),
                            os.path.join(REC_DIR,"poses","*.dlg.gz")]))
    sdf_n = len(list_files([os.path.join(REC_DIR,"sdf","*.sdf")]))
    print("=== PRECHECK ===")
    print("REC_DIR :", REC_DIR, "exists=", os.path.exists(REC_DIR))
    print("GRID_DIR:", GRID_DIR, "exists=", os.path.exists(GRID_DIR))
    print("SMILES  :", SMILES_CSV, "exists=", os.path.exists(SMILES_CSV))
    print(f"DLG files={dlg_n}  SDF files={sdf_n}  PLIP files={plip_n}")
    print("center :", center, "source:", src, "file:", srcfile)

def audit_and_save_missing(dlg_ids, plip_ids):
    missing = sorted(dlg_ids - plip_ids)
    if not missing: return []
    audit_csv = os.path.join(OUT_DIR, f"plip_missing__{REC_TAG}.csv")
    pd.DataFrame({"ligand_id": missing}).to_csv(audit_csv, index=False)
    print(f"[AUDIT] PLIP missing {len(missing)} ligands → {audit_csv}")
    return missing

def main():
    pocket_center, center_src, center_file = read_predictions_or_gpf(GRID_DIR)
    preflight(pocket_center, center_src, center_file)

    smi_map, label_cols = load_smiles_and_labels(SMILES_CSV)
    dlg_files = list_files([os.path.join(REC_DIR,"poses","*.dlg"),
                            os.path.join(REC_DIR,"poses","*.dlg.gz")])
    sdf_files = {Path(p).stem: p for p in list_files([os.path.join(REC_DIR,"sdf","*.sdf")])}

    # --- v2: PLIP globs 확장 ---
    plip_globs = [
        os.path.join(REC_DIR, "plip_batch",  "*", "intrxn*.csv"),
        os.path.join(REC_DIR, "plip_batch",  "*", "intrxn*.csv.gz"),
        os.path.join(REC_DIR, "plip_batch",  "*", "interactions*.csv"),
        os.path.join(REC_DIR, "plip_batch",  "*", "interactions*.csv.gz"),
        os.path.join(REC_DIR, "plip_batch2", "*", "intrxn*.csv"),
        os.path.join(REC_DIR, "plip_batch2", "*", "interactions*.csv"),
    ]
    plip_files={}
    for p in list_files(plip_globs):
        key = Path(p).parent.name  # 폴더명이 ligand_id
        plip_files.setdefault(key, p)

    rdkit_cache={}
    rows=[]
    dlg_ids = set()
    plip_ids = set(plip_files.keys())

    for i, dlg in enumerate(dlg_files, 1):
        lig = Path(dlg).stem.strip()
        dlg_ids.add(lig)

        Es, n_tors, dlg_smiles = parse_dlg(dlg)
        Es_sorted = sorted(Es) if Es else []
        dg_min = Es_sorted[0] if Es_sorted else np.nan

        smiles, labels = smi_map.get(lig, (None, {c: np.nan for c in label_cols}))
        if (not smiles) and dlg_smiles:
            smiles = dlg_smiles
        props = rdkit_props(smiles, rdkit_cache) if smiles else {}
        n_heavy = props.get("n_heavy", np.nan)
        dg_per_heavy = (dg_min/n_heavy) if (n_heavy and not np.isnan(dg_min)) else np.nan

        # ΔG → Ki(nM) 변환 (가정: kcal/mol, 298K)
        R=1.987e-3; T=298.0
        ki_nM = (1e9*np.exp(dg_min/(R*T))) if not np.isnan(dg_min) else np.nan

        sdf = sdf_files.get(lig)
        plip_df = pd.read_csv(plip_files[lig], sep=None, engine="python") if lig in plip_files else None

        # 리간드 중심: PLIP(LIGCOO) → SDF first conformer → None
        def plip_centroid_local(df):
            if df is None or "LIGCOO" not in df.columns: return None
            nums = lambda s: re.findall(r"[-+]?\d+\.?\d*", str(s))
            pts=[]
            for v in df["LIGCOO"].dropna():
                arr = nums(v)
                if len(arr)>=3:
                    pts.append([float(arr[0]), float(arr[1]), float(arr[2])])
            return np.mean(np.array(pts), axis=0) if pts else None
        lig_cent = plip_centroid_local(plip_df)
        if lig_cent is None:
            lig_cent = sdf_centroid_firstconf(sdf)

        centroid_dist = float(np.linalg.norm(lig_cent - pocket_center)) if lig_cent is not None else np.nan
        B = plip_features(plip_df, n_heavy)

        row = {
            "schema_ver": SCHEMA_VER, "receptor": REC_TAG, "ligand_id": lig, "center_source": center_src,
            "dg_min": dg_min, "dg_per_heavy": dg_per_heavy, "ki_nM": ki_nM,
            "n_tors": n_tors, "pose_count": len(Es_sorted),
            "centroid_dist": centroid_dist,
            **B,
            "smiles": smiles, "n_heavy": n_heavy,
            "MW": props.get("MW", np.nan), "logP": props.get("logP", np.nan),
            "TPSA": props.get("TPSA", np.nan), "HBD": props.get("HBD", np.nan),
            "HBA": props.get("HBA", np.nan), "RotBonds": props.get("RotBonds", np.nan),
            "FormalCharge": props.get("FormalCharge", np.nan),
            "RingCount": props.get("RingCount", np.nan),
            "AromaticRingCount": props.get("AromaticRingCount", np.nan),
            "ECFP4_bitcount": props.get("ECFP4_bitcount", np.nan),
            "dock_missing": int(len(Es_sorted)==0),
            "sdf_missing": int(sdf is None),
            "plip_missing": int(plip_df is None),
        }
        row.update(labels)
        rows.append(row)
        if i % 1000 == 0:
            print(f"[{REC_TAG}] processed {i}/{len(dlg_files)}")

    # 누락 리포트(정보용) — 학습/예측은 그대로 진행
    missing_list = audit_and_save_missing(dlg_ids, plip_ids)

    df = pd.DataFrame(rows)
    meta_cols=["schema_ver","receptor","ligand_id","center_source"]
    A_cols=["dg_min","dg_per_heavy","ki_nM","n_tors","pose_count","centroid_dist"]
    B_cols=["n_hbond","n_hbond_don","n_hbond_acc","n_hydrophob","n_pistack","n_halogen",
            "n_saltbridge","n_waterbridge","n_metal","total_contacts",
            "hbond_dist_mean","hbond_dist_min","hbond_angle_mean",
            "contact_per_heavy","frac_aromatic_contacts",
            "contacts_to__acidic","contacts_to__basic","contacts_to__polar",
            "contacts_to__aromatic","contacts_to__hydrophobic","contacts_to__other",
            "plip_rows"]
    C_cols=["smiles","n_heavy","MW","logP","TPSA","HBD","HBA","RotBonds",
            "FormalCharge","RingCount","AromaticRingCount","ECFP4_bitcount"]
    avail_cols=["dock_missing","sdf_missing","plip_missing"]
    label_cols=[c for c in df.columns if c not in set(meta_cols+A_cols+B_cols+C_cols+avail_cols)]
    order=meta_cols+A_cols+B_cols+C_cols+avail_cols+label_cols
    order=[c for c in order if c in df.columns]+[c for c in df.columns if c not in order]
    df=df.reindex(columns=order)
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] wrote {OUT_CSV} (rows={len(df)}, cols={len(df.columns)})")
    print(f"[SUM] dock_missing={int(df['dock_missing'].sum())} | sdf_missing={int(df['sdf_missing'].sum())} | plip_missing={int(df['plip_missing'].sum())}")
    if missing_list:
        print(f"[NOTE] PLIP 누락 {len(missing_list)}건은 의도적으로 제외(v2).")

if __name__=="__main__":
    main()
