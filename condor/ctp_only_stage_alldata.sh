#!/bin/bash
# Invoked by ctp_only_stage_alldata.sub, on the H100 box. Copies gan_bpm's all-data lambda_adv = 0
# runs (all/split.json, all/checkpoints/epoch_*.pt) for Tmax and rCBF to the box's own disk, for
# the all-data MRDPM (ctp_only_eval_plan.md §6), checks the copy, and writes STAGED_OK_all last.
#
# The data need no copy: every case of both targets is already there (ctp_only_stage_home.sub,
# target tmax and rcbf). This only adds gan_run_all_<target>/ and touches nothing else, so it can
# run while training jobs do.
set -u

export HOME="${_CONDOR_SCRATCH_DIR:-/tmp}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

MATRIX=/radraid2/colivares/matrix
DST=/home/colivares/mrdpm_ctp_only/stage
MARK="$DST/STAGED_OK_all"

fail() { echo "FATAL: $*"; exit 1; }

echo "host: $(hostname)   date: $(date -Is)"
for m in STAGED_OK STAGED_OK_rcbf; do
    [ -f "$DST/$m" ] || fail "no $DST/$m: stage both targets' data first (ctp_only_stage_home.sub)"
done
rm -f "$MARK"                   # until this run has checked its copy
echo "==================================================================="

for target in tmax rcbf; do
    case "$target" in
        tmax) RUN="$MATRIX/gan_bpm_tmax_f13uclav2_t12_adv0_all" ;;
        rcbf) RUN="$MATRIX/gan_bpm_rcbf_f13uclav2_adv0_all" ;;
    esac
    OUT="$DST/gan_run_all_$target"
    echo "$target: $RUN"
    [ -f "$RUN/all_data_summary.json" ] || fail "$RUN has no all_data_summary.json: not finished"
    mkdir -p "$OUT/all/checkpoints"
    cp "$RUN/all_data_summary.json" "$OUT/" || fail "copying $target's all_data_summary.json"
    cp "$RUN/all/split.json" "$OUT/all/" || fail "copying $target's split.json"
    n=0
    for ck in "$RUN"/all/checkpoints/epoch_*.pt; do
        [ -f "$ck" ] || continue
        cp "$ck" "$OUT/all/checkpoints/" || fail "copying $ck"
        a=$(sha256sum < "$ck" | cut -c1-16)
        b=$(sha256sum < "$OUT/all/checkpoints/$(basename "$ck")" | cut -c1-16)
        [ "$a" = "$b" ] || fail "$(basename "$ck") differs after the copy ($a vs $b)"
        echo "   $(basename "$ck") sha256 $b"
        n=$((n + 1))
    done
    [ "$n" -gt 0 ] || fail "$RUN/all/checkpoints has no epoch_*.pt"
    python - "$DST/mctp_dataset_f13_isles24_ucla_v2" "$OUT/all/split.json" "$target" <<'EOF' || fail "the $target split"
import json, sys
from pathlib import Path
import pandas as pd
root, split, target = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), sys.argv[3]
train, val = set(split["train_case_ids"]), set(split["val_case_ids"])
rows = pd.read_parquet(root / f"{target}_index.parquet")
missing = sorted(train - set(rows["case_id"]))
rows = rows[rows["case_id"].isin(train)]
files = [n for n in list(rows["input_relpath"]) + list(rows["label_relpath"]) if not (root / n).is_file()]
print(f"   split: {len(train)} training cases, {len(val)} validation; {len(missing)} not in the staged "
      f"{target} index, {len(files)} of its files missing")
sys.exit(1 if val or missing or files or len(train) != 443 else 0)
EOF
done
du -sh "$DST"/gan_run_all_*

{ echo "staged all-data generators $(date -Is) on $(hostname) from $MATRIX"; } > "$MARK"
echo "wrote $MARK"
