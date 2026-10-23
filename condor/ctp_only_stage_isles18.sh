#!/bin/bash
# Invoked by ctp_only_stage_isles18.sub, on the H100 box. Copies ISLES 2018 (all 94 scans,
# both labels) from the /radraid2 stage to the box's own disk, for the test job's
# testset = isles18 (ctp_only_eval_plan.md §6), checks the copy, and writes STAGED_OK_isles18
# last. It only adds stage/isles2018/ and the marker, so it can run while training jobs do.
#
# Before it, once, on 14958 (whose /raid the H100 can't see): the split and the /radraid2
# copy, by ctp_only_isles18_maps.py split and ctp_only_stage_data.py (the handoff's paste).
set -u

export HOME="${_CONDOR_SCRATCH_DIR:-/tmp}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

NAME=mctp_dataset_msu_f13_isles2018_all94
SPLIT=/radraid2/colivares/mrdpm_ctp_only/isles2018/split.json
SRC=/radraid2/colivares/mrdpm_ctp_only/stage/isles2018
DST=/home/colivares/mrdpm_ctp_only/stage/isles2018
MARK=/home/colivares/mrdpm_ctp_only/stage/STAGED_OK_isles18
STAGER="${_CONDOR_SCRATCH_DIR:-$PWD}/ctp_only_stage_data.py"

fail() { echo "FATAL: $*"; exit 1; }

echo "host: $(hostname)   date: $(date -Is)"
df -h /home | tail -1
[ -f "$SPLIT" ] || fail "no $SPLIT: run ctp_only_isles18_maps.py split on 14958 first"
for t in rcbf tmax; do
    [ -f "$SRC/$NAME/${t}_index.parquet" ] || fail "no $SRC/$NAME/${t}_index.parquet: copy ISLES 2018 to /radraid2 first"
done
rm -f "$MARK"                   # until this run has checked its copy
mkdir -p "$DST" || fail "cannot create $DST"
echo "==================================================================="

echo "1. the 94 scans, both labels"
for t in rcbf tmax; do
    python "$STAGER" "$SRC/$NAME" "$DST/$NAME" --split "$SPLIT" --side val --label-type "$t" \
        || fail "staging $t"
done
cp "$SPLIT" "$DST/split.json" || fail "copying split.json"

echo "==================================================================="
echo "2. checks"
python - "$SRC/$NAME" "$DST/$NAME" "$SPLIT" "$DST/split.json" <<'EOF' || fail "the copy"
import hashlib, json, sys
from pathlib import Path
import pandas as pd
src, dst, split, split_copy = (Path(a) for a in sys.argv[1:])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
cases = set(json.loads(split.read_text())["val_case_ids"])
bad = [] if sha(split) == sha(split_copy) else ["split.json differs"]
keys = {}
for t in ("rcbf", "tmax"):
    index = f"{t}_index.parquet"
    if sha(src / index) != sha(dst / index):
        bad.append(f"{index} differs")
    rows = pd.read_parquet(dst / index)
    rows = rows[rows["case_id"].astype(str).isin(cases)]
    names = sorted(set(rows["input_relpath"]) | set(rows["label_relpath"]))
    wrong = [n for n in names if not (dst / n).is_file() or (dst / n).stat().st_size != (src / n).stat().st_size]
    keys[t] = set(zip(rows["case_id"].astype(str), rows["slice_idx"].astype(int)))
    print(f"{t}: {rows['case_id'].nunique()} scans, {len(rows)} slices, {len(names)} files; "
          f"{len(wrong)} missing or of another size")
    if wrong or rows["case_id"].nunique() != 94 or len(rows) != 502:
        bad.append(f"{t}: {len(wrong)} files wrong, {rows['case_id'].nunique()} scans, {len(rows)} slices")
if keys["rcbf"] != keys["tmax"]:
    bad.append("the rCBF and Tmax indexes hold other slices")
cache = [f for f in (src / "mask_cache").rglob("*") if f.is_file()] if (src / "mask_cache").is_dir() else []
missing = [f for f in cache if not (dst / f.relative_to(src)).is_file()]
print(f"mask cache: {len(cache)} entries, {len(missing)} not copied")
bad += [f"{len(missing)} mask-cache entries not copied"] if missing else []
print("\n".join(f"PROBLEM: {b}" for b in bad) or "copy checked")
sys.exit(1 if bad else 0)
EOF
du -sh "$DST"

{ echo "staged ISLES 2018 $(date -Is) on $(hostname) from $SRC and $SPLIT"; } > "$MARK"
echo "wrote $MARK"
