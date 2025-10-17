#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, re, csv, time, argparse, subprocess, shutil
from pathlib import Path
from multiprocessing import Pool
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter

# rdMolStandardize가 없을 수도 있으므로 안전하게 임포트
try:
    from rdkit.Chem.MolStandardize import rdMolStandardize as sdz  # 신형 경로
except Exception:
    try:
        from rdkit.Chem import rdMolStandardize as sdz  # 구형 경로
    except Exception:
        sdz = None  # 폴백

DEFAULT_WORKERS = max(1, min(8, os.cpu_count() // 2 if os.cpu_count() else 1))
RDKit_MAX_ITERS = 2000

def safe_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^\w.\-+]", "_", s)
    return s[:200] if len(s) > 200 else s

def is_metal_z(z: int) -> bool:
    ranges = [(3,5), (11,13), (19,31), (37,50), (55,83)]
    return any(a <= z <= b for (a,b) in ranges)

def remove_metals_descending(m: Chem.Mol) -> Chem.Mol:
    rw = Chem.RWMol(m)
    idxs = [a.GetIdx() for a in rw.GetAtoms() if is_metal_z(a.GetAtomicNum())]
    for i in sorted(idxs, reverse=True):
        rw.RemoveAtom(i)
    return rw.GetMol()

def torstorf_ok(pdbqt: Path) -> bool:
    try:
        with open(pdbqt, "r", errors="ignore") as f:
            for line in f:
                if "TORSDOF" in line:
                    return True
    except Exception:
        pass
    return False

def heavy_nonH_nonMetal_count(m: Chem.Mol) -> int:
    return sum(1 for a in m.GetAtoms() if (a.GetAtomicNum() > 1) and (not is_metal_z(a.GetAtomicNum())))

def standardize_disconnect(m: Chem.Mol) -> Chem.Mol:
    # sdz 있으면 전체 파이프라인, 없으면 금속 제거 + 최대 유기 프래그먼트 선택만
    if sdz is not None:
        try: m = sdz.Cleanup(m)
        except Exception: pass
        try: m = sdz.FragmentParent(m)
        except Exception: pass
        try: m = sdz.Uncharger().uncharge(m)
        except Exception: pass
        try: m = sdz.LargestFragmentChooser(preferOrganic=True).choose(m)
        except Exception: pass
        try: m = sdz.MetalDisconnector().Disconnect(m)
        except Exception: pass
        return m
    # ---- 폴백 ----
    m = remove_metals_descending(m)
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    if not frags:
        return m
    frags = [Chem.Mol(f) for f in frags if f.GetNumAtoms() > 0]
    frags.sort(key=lambda f: heavy_nonH_nonMetal_count(f), reverse=True)
    return frags[0]

def soft_sanitize(m: Chem.Mol) -> Chem.Mol:
    try:
        Chem.SanitizeMol(m)
        return m
    except Exception:
        pass
    try:
        Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    except Exception:
        Chem.SanitizeMol(
            m,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES
            | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
            | Chem.SanitizeFlags.SANITIZE_CLEANUP
        )
    return m

def _try_embed_and_optimize(core: Chem.Mol) -> None:
    p = AllChem.ETKDGv3(); p.randomSeed = 17
    if AllChem.EmbedMolecule(core, p) != 0:
        p2 = AllChem.ETKDGv2(); p2.randomSeed = 17
        if AllChem.EmbedMolecule(core, p2) != 0:
            if AllChem.EmbedMolecule(core, useRandomCoords=True, randomSeed=17, maxAttempts=50) != 0:
                raise RuntimeError("3D embedding failed")

    did_opt = False
    try:
        if AllChem.UFFHasAllMoleculeParams(core):
            AllChem.UFFOptimizeMolecule(core, maxIters=RDKit_MAX_ITERS)
            did_opt = True
    except Exception:
        pass
    if not did_opt:
        try:
            if AllChem.MMFFHasAllMoleculeParams(core):
                AllChem.MMFFOptimizeMolecule(core, maxIters=RDKit_MAX_ITERS)
        except Exception:
            pass

def _obabel_fallback(smiles: str, sdf_path: Path, timeout=60) -> bool:
    obabel = shutil.which("obabel")
    if not obabel:
        return False
    cmd = [obabel, f'-:"{smiles}"', "-osdf", "-O", str(sdf_path), "--gen3d", "-h"]
    try:
        p = subprocess.run(" ".join(cmd), shell=True, capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0 and sdf_path.exists() and sdf_path.stat().st_size > 0
    except Exception:
        return False

def build_sdf_from_smiles(smiles: str, mol_id: str, sdf_path: Path, min_heavy:int=2) -> None:
    smi = (smiles or "").strip()
    if not smi:
        raise RuntimeError("Empty SMILES")

    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        if _obabel_fallback(smi, sdf_path):
            return
        raise RuntimeError(f"MolFromSmiles failed: {smi[:60]}...")

    mol = standardize_disconnect(mol)

    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        raise RuntimeError("No fragments after cleaning")
    frags = [Chem.Mol(f) for f in frags if f.GetNumAtoms() > 0]
    frags.sort(key=lambda f: heavy_nonH_nonMetal_count(f), reverse=True)
    core = frags[0]

    core = soft_sanitize(core)
    core = Chem.AddHs(core)

    if heavy_nonH_nonMetal_count(core) < min_heavy:
        raise RuntimeError(f"Too small after cleaning (<{min_heavy} heavy non-metal atoms)")

    _try_embed_and_optimize(core)
    core.SetProp("_Name", mol_id)

    sdf_path.parent.mkdir(parents=True, exist_ok=True)
    with SDWriter(str(sdf_path)) as w:
        w.write(core)

def run_meeko(meeko_cli: Path, sdf_path: Path, out_pdbqt: Path, try_pH=True, pH=7.4, timeout=300) -> (bool, str, str):
    pyexe = sys.executable
    base_mod  = [pyexe, "-m", "meeko.cli.mk_prepare_ligand", "-i", str(sdf_path), "-o", str(out_pdbqt)]
    base_file = [pyexe, str(meeko_cli), "-i", str(sdf_path), "-o", str(out_pdbqt)]
    cmds = []
    if try_pH:
        cmds += [ base_mod + ["--pH", str(pH)], base_mod, base_file + ["--pH", str(pH)], base_file ]
    else:
        cmds += [ base_mod, base_mod + ["--pH", str(pH)], base_file, base_file + ["--pH", str(pH)] ]
    last_out, last_err = "", ""
    for cmd in cmds:
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = "/home/ssm-user/apps/Meeko:" + env.get("PYTHONPATH", "")
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
            last_out, last_err = p.stdout, p.stderr
            if p.returncode == 0 and out_pdbqt.exists():
                return True, last_out, last_err
        except subprocess.TimeoutExpired as e:
            last_out, last_err = e.stdout or "", (e.stderr or "") + "\n[ERROR] Meeko timeout"
        except Exception as e:
            last_out, last_err = "", f"[ERROR] Meeko exception: {e}"
    return False, last_out, last_err

def write_log(log_path: Path, content: str):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(content)

def read_table(in_file: Path) -> pd.DataFrame:
    return pd.read_csv(in_file, sep=None, engine="python", dtype=str, keep_default_na=False)

def _smiles_score_series(s: pd.Series, n=200) -> float:
    cnt = 0; tot = 0
    for v in s.dropna().astype(str).head(n):
        tot += 1
        if Chem.MolFromSmiles(v) is not None: cnt += 1
    return (cnt / tot) if tot else 0.0

def detect_cols(df: pd.DataFrame, user_smiles=None, user_id=None):
    cols_lc = {c.lower().strip(): c for c in df.columns}
    if user_smiles and user_smiles in df.columns:
        smi_col = user_smiles
    else:
        for key in ["smiles","canonical_smiles","smiles_clean"]:
            if key in cols_lc:
                smi_col = cols_lc[key]; break
        else:
            scores = {c: _smiles_score_series(df[c]) for c in df.columns}
            smi_col = max(scores, key=scores.get)
            if scores[smi_col] < 0.5:
                raise RuntimeError(f"Could not confidently detect SMILES column. Top candidate '{smi_col}' score={scores[smi_col]:.2f}")
    if user_id and user_id in df.columns:
        id_col = user_id
    else:
        for key in ["sample id","sample_id","id","ligand_id","compound_id"]:
            if key in cols_lc:
                id_col = cols_lc[key]; break
        else:
            candidates = [c for c in df.columns if c != smi_col]
            uniq_scores = {c: (df[c].astype(str).nunique() / max(1,len(df))) for c in candidates}
            id_col = max(uniq_scores, key=uniq_scores.get)
    return smi_col, id_col

def worker(args):
    smiles, id_raw, meeko_cli, sdf_dir, pdbqt_dir, log_dir, try_pH, min_heavy = args
    os.environ["OMP_NUM_THREADS"] = "1"
    mol_id = safe_name(id_raw)
    sdf_path = sdf_dir / f"{mol_id}.sdf"
    out_pdbqt = pdbqt_dir / f"{mol_id}.pdbqt"
    log_path = log_dir / f"{mol_id}.log"

    if out_pdbqt.exists() and torstorf_ok(out_pdbqt):
        return {"id": mol_id, "status": "skip_exist", "reason": "", "pdbqt": str(out_pdbqt)}

    rdkit_log = []
    try:
        build_sdf_from_smiles(smiles, mol_id, sdf_path, min_heavy=min_heavy)
        rdkit_log.append(f"[OK] SDF: {sdf_path}")
    except Exception as e:
        reason = f"RDKit_failed: {e}"
        write_log(log_path, "\n".join(rdkit_log + [reason]))
        return {"id": mol_id, "status": "fail", "reason": reason, "pdbqt": ""}

    ok, meeko_out, meeko_err = run_meeko(meeko_cli, sdf_path, out_pdbqt, try_pH=try_pH)
    log_txt = "\n".join(rdkit_log + ["[MEEKO OUT]\n" + (meeko_out or ""), "[MEEKO ERR]\n" + (meeko_err or "")])
    write_log(log_path, log_txt)

    if not ok:
        return {"id": mol_id, "status": "fail", "reason": "Meeko_failed", "pdbqt": ""}

    if torstorf_ok(out_pdbqt):
        return {"id": mol_id, "status": "ok", "reason": "", "pdbqt": str(out_pdbqt)}
    else:
        return {"id": mol_id, "status": "warn", "reason": "no_TORSDOF", "pdbqt": str(out_pdbqt)}

def main():
    ap = argparse.ArgumentParser(description="Batch RDKit->SDF->Meeko PDBQT (v4a; rdMolStandardize optional)")
    ap.add_argument("--in", dest="in_file", required=True, help="입력 CSV/TSV (SMILES, ID)")
    ap.add_argument("--out", dest="outdir", required=True, help="출력 루트 디렉토리")
    ap.add_argument("--meeko", dest="meeko_cli", default=str(Path.home() / "apps/Meeko/meeko/cli/mk_prepare_ligand.py"), help="mk_prepare_ligand.py 경로")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"동시 작업 수 (기본 {DEFAULT_WORKERS})")
    ap.add_argument("--no-ph-first", action="store_true", help="Meeko 실행 시 pH 옵션을 먼저 시도하지 않음")
    ap.add_argument("--smiles-col", dest="smiles_col", default=None, help="SMILES 컬럼명 강제 지정")
    ap.add_argument("--id-col", dest="id_col", default=None, help="ID 컬럼명 강제 지정")
    ap.add_argument("--min-heavy", dest="min_heavy", type=int, default=2, help="금속 제외 비수소 원자 최소 수(기본=2; 1로 더 공격적으로)")

    args = ap.parse_args()

    in_file = Path(args.in_file).expanduser().resolve()
    out_root = Path(args.outdir).expanduser().resolve()
    meeko_cli = Path(args.meeko_cli).expanduser().resolve()
    workers = max(1, args.workers)

    sdf_dir = out_root / "sdf"
    pdbqt_dir = out_root / "pdbqt"
    log_dir = out_root / "logs_meeko"
    out_root.mkdir(parents=True, exist_ok=True)
    sdf_dir.mkdir(parents=True, exist_ok=True)
    pdbqt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(in_file)
    smi_col, id_col = detect_cols(df, user_smiles=args.smiles_col, user_id=args.id_col)
    sub = df[[smi_col, id_col]].dropna().drop_duplicates()
    sub.columns = ["SMILES", "ID"]

    print(f"[INFO] 입력 {len(sub)}개 | SMILES='{smi_col}' | ID='{id_col}' | min_heavy={args.min_heavy} | rdMolStandardize={'on' if sdz else 'off'}")
    print("[INFO] 예시 3개:", sub.head(3).to_dict("records"))

    tasks = [
        (row.SMILES, row.ID, meeko_cli, sdf_dir, pdbqt_dir, log_dir, (not args.no_ph_first), int(args.min_heavy))
        for row in sub.itertuples(index=False)
    ]

    t0 = time.time()
    ok = warn = fail = skip = 0
    results = []
    with Pool(processes=workers) as pool:
        for res in pool.imap_unordered(worker, tasks, chunksize=4):
            results.append(res)
            if res["status"] == "ok": ok += 1
            elif res["status"] == "warn": warn += 1
            elif res["status"] == "skip_exist": skip += 1
            else: fail += 1
            if (ok + warn + skip + fail) % 25 == 0:
                print(f"[PROGRESS] ok={ok} warn={warn} skip={skip} fail={fail}")

    dt = time.time() - t0
    print(f"\n[DONE] total={len(results)}  ok={ok}  warn={warn}  skip={skip}  fail={fail}  ({dt:.1f}s)")

    summary_tsv = out_root / "meeko_batch_summary.tsv"
    with open(summary_tsv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "status", "reason", "pdbqt_path"])
        for r in results:
            w.writerow([r["id"], r["status"], r["reason"], r["pdbqt"]])

    fail_tsv = out_root / "meeko_fail_summary.tsv"
    with open(fail_tsv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "reason"])
        for r in results:
            if r["status"] == "fail":
                w.writerow([r["id"], r["reason"]])

    print(f"[WRITE] 요약 TSV: {summary_tsv}")
    print(f"[WRITE] 실패 TSV: {fail_tsv}")
    print(f"[OUT ] SDF:   {sdf_dir}")
    print(f"[OUT ] PDBQT: {pdbqt_dir}")
    print(f"[LOG ] {log_dir}")

if __name__ == "__main__":
    main()
