#!/usr/bin/env bash
# sweep_p2rank23_gpu_parallel_v10b.sh
# - Multi-ligand sets via LIGDIRS (colon-separated) + optional LIGTAGS
# - Unique ligand_id: <tag>__<basename> to avoid collisions
# - Keeps v10 speed-focused batch(-B), summary, resume logic
# - No bash arrays/mapfile; POSIX-friendly constructs
set -euo pipefail
export LC_ALL=C

# === 기본 경로/옵션 ===
ROOT="${ROOT:-/home/ssm-user/example/atdk_gpu}"
DOCKBIN="${DOCKBIN:-$HOME/apps/AutoDock-GPU/bin/autodock_gpu_128wi}"
REC_PARENT="${REC_PARENT:-$ROOT/grids/p2rank_selected_23}"

# 여러 리간드 세트: 콜론(:)으로 구분
# 예) LIGDIRS="/home/.../tox21_ligprep/pdbqt:/home/.../tox_val_ligprep/pdbqt"
LIGDIRS="${LIGDIRS:-$ROOT/ligand_batch_v4a/pdbqt}"
# 각 세트 태그(선택): 콜론(:)으로 구분; 개수 안 맞으면 해당 디렉토리 basename 사용
# 예) LIGTAGS="res:val"
LIGTAGS="${LIGTAGS:-}"

NRUN="${NRUN:-1}"
REC_JOBS="${REC_JOBS:-1}"         # 리셉터 병렬(외부)
JOBS_PER_REC="${JOBS_PER_REC:-4}" # 청크 병렬(내부)
MAXLIG="${MAXLIG:-0}"             # 0=전체, >0=상위 N개만(머지 후 적용)
SHUFFLE="${SHUFFLE:-0}"           # 1=셔플
POCKET_MODE="${POCKET_MODE:-link}"# map link|copy

RUN_TAG="${RUN_TAG:-gpu_p2rank23_v10b_nrun${NRUN}_$(date +%Y%m%d_%H%M%S)}"
OUTROOT="$ROOT/docking/$RUN_TAG"
mkdir -p "$OUTROOT"

# === 점검 ===
[[ -x "$DOCKBIN" ]] || { echo "[ERR] DOCKBIN not found: $DOCKBIN"; exit 1; }
[[ -d "$REC_PARENT" ]] || { echo "[ERR] REC_PARENT not found: $REC_PARENT"; exit 1; }

# LIGDIRS 유효성 체크 & 표시
echo "[INFO] ROOT=$ROOT"
echo "[INFO] OUTROOT=$OUTROOT"
echo "[INFO] REC_PARENT=$REC_PARENT"
echo "[INFO] DOCKBIN=$DOCKBIN"
echo "[INFO] NRUN=$NRUN  REC_JOBS=$REC_JOBS  JOBS_PER_REC=$JOBS_PER_REC  MAXLIG=$MAXLIG  SHUFFLE=$SHUFFLE  POCKET_MODE=$POCKET_MODE  RUN_TAG=$RUN_TAG"
echo "[INFO] LIGDIRS=$LIGDIRS"
echo "[INFO] LIGTAGS=$LIGTAGS"
echo

# === 리간드 마스터(TSV: path \t lid) 생성 ===
LIG_ALL_TSV="$OUTROOT/ligands.master.tsv"
: > "$LIG_ALL_TSV"

# 내부: i번째 태그 얻기(없으면 빈칸)
_get_tag_by_index() {
  local idx="$1" tags="$2"
  if [[ -z "$tags" ]]; then printf ""; return 0; fi
  awk -v FS=":" -v i="$idx" '{
    if (NF>=i) {
      for (k=1;k<i;k++) $k="";
      n=split($0,a,":");
      if (i<=n) print a[i];
    }
  }' <(printf "%s" "$tags") | head -n1
}

i=0
OLDIFS="$IFS"; IFS=':'; set -- $LIGDIRS; IFS="$OLDIFS"
for DIR in "$@"; do
  [[ -d "$DIR" ]] || { echo "[ERR] ligand dir not found: $DIR"; exit 1; }
  i=$((i+1))
  TAG="$(_get_tag_by_index "$i" "$LIGTAGS")"
  [[ -z "$TAG" ]] && TAG="$(basename "$DIR")"

  # DIR 내 pdbqt 수집 -> path \t <TAG>__<basename>
  find "$DIR" -maxdepth 1 -type f -name '*.pdbqt' -print0 \
  | while IFS= read -r -d '' f; do
      b="$(basename "${f%.pdbqt}")"
      printf "%s\t%s__%s\n" "$f" "$TAG" "$b" >> "$LIG_ALL_TSV"
    done
done

# 정렬/셔플/상위 N
TMP="$OUTROOT/.lig.tmp"
if (( SHUFFLE )); then
  shuf "$LIG_ALL_TSV" > "$TMP"
else
  sort -t $'\t' -k2,2 "$LIG_ALL_TSV" > "$TMP"
fi
if (( MAXLIG > 0 )); then
  head -n "$MAXLIG" "$TMP" > "$LIG_ALL_TSV"
else
  mv -f "$TMP" "$LIG_ALL_TSV"
fi
LCOUNT=$(wc -l < "$LIG_ALL_TSV" || echo 0)
echo "[INFO] ligands selected (merged) = $LCOUNT"

# === receptor.maps.fld 수집 ===
FLD_LIST="$OUTROOT/flds.list.txt"
find -L "$REC_PARENT" \( -type d -name 'No' -prune \) -o \( -type f -name 'receptor.maps.fld' -print \) | sort > "$FLD_LIST"
RCOUNT=$(wc -l < "$FLD_LIST" || echo 0)
echo "[INFO] receptors(found fld) = $RCOUNT"
(( RCOUNT > 0 )) || { echo "[ERR] no receptor.maps.fld under $REC_PARENT"; exit 2; }
echo

# === 유틸 ===
parse_best_kcalmol() {
  awk '
    BEGIN{min=1e9; ok=0}
    tolower($0) ~ /estimated free energy of binding/ {
      for(i=1;i<=NF;i++){
        if ($i ~ /^-?[0-9]*\.[0-9]+$/) { v=$i+0; if(v<min){min=v; ok=1} }
      }
    }
    END{ if(ok) printf("%.3f", min); }
  ' "$1"
}

prepare_pocket() {
  local fld="$1" work="$2" fld_dir
  fld_dir="$(dirname "$fld")"
  mkdir -p "$work/pocket" "$work/logs" "$work/poses"
  cp -f "$fld" "$work/pocket/receptor.maps.fld"
  sed -i -E 's#\.\./pocket/##g; s#^\./##g; s#(^|[[:space:]])pocket/#\1#g' "$work/pocket/receptor.maps.fld"
  ls "$fld_dir"/*.map >/dev/null 2>&1 || { echo "[ERR] no *.map next to $fld"; return 3; }
  if [[ "$POCKET_MODE" == "copy" ]]; then
    cp -f "$fld_dir"/*.map "$work/pocket/"
  else
    for m in "$fld_dir"/*.map; do ln -sf "$m" "$work/pocket/"; done
  fi
}

# chunk.tsv(path \t lid) -> batch( fld \n path \n lid )*
make_batch_from_chunk() {
  local fld_rel="$1" chunk_tsv="$2" batch_txt="$3"
  : > "$batch_txt"
  while IFS=$'\t' read -r lig lid; do
    [[ -n "$lig" && -n "$lid" ]] || continue
    printf "%s\n%s\n%s\n" "$fld_rel" "$lig" "$lid" >> "$batch_txt"
  done < "$chunk_tsv"
}

dock_one_receptor() {
  local FLD="$1" D BASE WORK start
  start=$(date +%s)
  D="$(dirname "$FLD")"; BASE="$(basename "$D")"
  WORK="$OUTROOT/$BASE"; mkdir -p "$WORK"

  echo "====================================================="
  echo "[INFO] Receptor: $BASE"
  echo "[INFO] FLD     : $FLD"
  echo "====================================================="

  prepare_pocket "$FLD" "$WORK"

  cp -f "$LIG_ALL_TSV" "$WORK/ligands.all.tsv"

  # 이미 존재하는 DLG 스킵: lig \t lid TSV 기준
  awk -F'\t' -v poses="$WORK/poses" '{
    lid=$2; dlg=poses "/" lid ".dlg";
    cmd="test -s \"" dlg "\"";
    if (system(cmd)!=0) print $0;
  }' "$WORK/ligands.all.tsv" > "$WORK/ligands.todo.tsv" || true

  TODO_COUNT=$(wc -l < "$WORK/ligands.todo.tsv" || echo 0)
  echo "[INFO] todo ligands = $TODO_COUNT"

  if (( TODO_COUNT > 0 )); then
    rm -f "$WORK"/chunk_* "$WORK"/chunk_*.tsv 2>/dev/null || true
    if (( JOBS_PER_REC <= 1 )); then
      cp -f "$WORK/ligands.todo.tsv" "$WORK/chunk_00.tsv"
    else
      split -n l/"$JOBS_PER_REC" -d -a 2 "$WORK/ligands.todo.tsv" "$WORK/chunk_"
      for f in "$WORK"/chunk_*; do
        [[ -e "$f" ]] || continue
        [[ "$f" == *.tsv ]] || mv -f "$f" "$f.tsv"
      done
    fi

    # 병렬 실행
    find "$WORK" -maxdepth 1 -name 'chunk_*.tsv' -print0 \
    | xargs -0 -I{} -P "$JOBS_PER_REC" bash -lc '
        set -euo pipefail
        CHUNK="{}"; WORK="'"$WORK"'"; DOCKBIN="'"$DOCKBIN"'"; NRUN="'"$NRUN"'"
        [[ -s "$CHUNK" ]] || exit 0
        BATCH="$WORK/pocket/$(basename "$CHUNK" .tsv).batch"
        LOG="$WORK/logs/$(basename "$CHUNK" .tsv).log"
        : > "$LOG"

        make_batch_from_chunk "receptor.maps.fld" "$CHUNK" "$BATCH"

        (
          cd "$WORK/pocket"
          echo "[BATCH] lines=$(wc -l < "$BATCH") file=$(basename "$BATCH")" >> "$LOG"
          if command -v /usr/bin/time >/dev/null 2>&1; then
            /usr/bin/time -f "%e" -o "${LOG}.time" \
              "$DOCKBIN" -B "$BATCH" --nrun "$NRUN" --dlgoutput 1 --xmloutput 0 \
              >> "$LOG" 2>&1 || true
          else
            t0=$(date +%s)
            "$DOCKBIN" -B "$BATCH" --nrun "$NRUN" --dlgoutput 1 --xmloutput 0 \
              >> "$LOG" 2>&1 || true
            t1=$(date +%s); echo "$((t1-t0))" > "${LOG}.time"
          fi
          # 라벨(lid)만 poses로 이동
          awk "NR%3==0{print}" "$BATCH" | while read -r lid; do
            [[ -s "${lid}.dlg" ]] && mv -f "${lid}.dlg" "$WORK/poses/" || true
          done
          # per-lig seconds(청크 평균)
          if [[ -s "${LOG}.time" ]]; then
            n=$(awk "END{print int((NR/3)+0.5)}" "$BATCH")
            if (( n > 0 )); then
              avg=$(awk -v t="$(cat "${LOG}.time")" -v n="$n" "BEGIN{printf \"%.3f\", t/n}")
              awk "NR%3==0{print}" "$BATCH" | awk -v s="$avg" "{print \$0\"\t\"s}" >> "$WORK/logs/per_ligand_secs.tsv"
            fi
          fi
        )
      '
  fi

  # 남은 DLG 수거(혹시 pocket에 남은 경우)
  cut -f2 "$WORK/ligands.all.tsv" 2>/dev/null \
    | while read -r lid; do
        [[ -s "$WORK/pocket/${lid}.dlg" ]] && mv -f "$WORK/pocket/${lid}.dlg" "$WORK/poses/" || true
      done

  # === summary 생성 ===
  SUMMARY="$WORK/summary.tsv"
  printf "ligand_id\tstatus\tbest_kcalmol\tseconds\tlogfile\n" > "$SUMMARY"

  OK_FILE="$WORK/logs/ok_lids.txt"
  find "$WORK/poses" -maxdepth 1 -type f -name "*.dlg" -printf "%f\n" \
    | sed 's/\.dlg$//' | sort -u > "$OK_FILE" || true

  SECS_FILE=""
  [[ -s "$WORK/logs/per_ligand_secs.tsv" ]] && SECS_FILE="$WORK/logs/per_ligand_secs.tsv"

  # 성공/에너지 파싱
  if [[ -s "$OK_FILE" ]]; then
    while read -r lid; do
      dlg="$WORK/poses/${lid}.dlg"
      best="$(parse_best_kcalmol "$dlg" || true)"
      secs=""
      if [[ -n "$SECS_FILE" ]]; then
        secs="$(awk -F'\t' -v L="$lid" '($1==L){print $2; exit}' "$SECS_FILE" || true)"
      fi
      status="ok"; [[ -z "$best" ]] && status="no_energy"
      printf "%s\t%s\t%s\t%s\t%s\n" "$lid" "$status" "$best" "$secs" "$dlg" >> "$SUMMARY"
    done < "$OK_FILE"
  fi

  # 실패(no_dlg)
  TRY_FILE="$WORK/logs/try_lids.txt"
  cut -f2 "$WORK/ligands.all.tsv" 2>/tmp/null | sort -u > "$TRY_FILE" || true
  if [[ -s "$TRY_FILE" ]]; then
    if [[ -s "$OK_FILE" ]]; then
      comm -23 "$TRY_FILE" "$OK_FILE" | while read -r lid; do
        printf "%s\t%s\t%s\t%s\t%s\n" "$lid" "fail:no_dlg" "" "" "$WORK/logs" >> "$SUMMARY"
      done
    else
      while read -r lid; do
        printf "%s\t%s\t%s\t%s\t%s\n" "$lid" "fail:no_dlg" "" "" "$WORK/logs" >> "$SUMMARY"
      done < "$TRY_FILE"
    fi
  fi

  dur=$(( $(date +%s) - start ))
  n_all=$(( $(wc -l < "$SUMMARY") - 1 ))
  n_ok=$(grep -E -c $'\tok\t|\tno_energy\t' "$SUMMARY" || true)
  echo "[DONE] $BASE | total=$n_all ok_or_no_energy=$n_ok | summary=$SUMMARY | ${dur}s"
  echo
}

export -f parse_best_kcalmol
export -f prepare_pocket
export -f make_batch_from_chunk
export -f dock_one_receptor
export ROOT DOCKBIN REC_PARENT LIGDIRS LIGTAGS NRUN REC_JOBS JOBS_PER_REC MAXLIG SHUFFLE POCKET_MODE OUTROOT LIG_ALL_TSV RUN_TAG

# === 리셉터 병렬 실행 ===
cat "$FLD_LIST" | xargs -I{} -P "$REC_JOBS" bash -lc 'dock_one_receptor "$1"' _ "{}"

# === 매니페스트 ===
MAN="$OUTROOT/run_manifest.tsv"
printf "receptor\tn_pocket_files\tn_pose_files\n" > "$MAN"
for rd in "$OUTROOT"/*_receptor; do
  [[ -d "$rd" ]] || continue
  pf=$(find "$rd/pocket" -maxdepth 1 -type f | wc -l | awk "{print \$1}")
  ps=$(find "$rd/poses"  -maxdepth 1 -type f -name "*.dlg" | wc -l | awk "{print \$1}")
  printf "%s\t%s\t%s\n" "$(basename "$rd")" "$pf" "$ps" >> "$MAN"
done

echo
echo "[ALL DONE] OUTROOT=$OUTROOT"
echo " - Per-receptor summaries: $OUTROOT/*_receptor/summary.tsv"
echo " - Manifest: $MAN"
