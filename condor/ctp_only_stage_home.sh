#!/bin/bash
# Invoked by ctp_only_stage_home.sub, on the H100 box. Copies from the /radraid2 stage to the
# box's own disk, checks the copy, and writes STAGED_OK last.
set -u

export HOME="${_CONDOR_SCRATCH_DIR:-/tmp}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

SRC=/radraid2/colivares/mrdpm_ctp_only/stage
GAN_RUN=/radraid2/colivares/matrix/gan_bpm_tmax_f13uclav2_t12_adv0_k5
DST=/home/colivares/mrdpm_ctp_only/stage
STAGER="${_CONDOR_SCRATCH_DIR:-$PWD}/ctp_only_stage_data.py"

fail() { echo "FATAL: $*"; exit 1; }

echo "host: $(hostname)   date: $(date -Is)"
df -h /home | tail -1
rm -f "$DST/STAGED_OK"          # until this run has checked its copy
mkdir -p "$DST" || fail "cannot create $DST"
echo "==================================================================="

echo "1. the dataset: every case of the 5-fold split, Tmax only"
python "$STAGER" "$SRC/mctp_dataset_f13_isles24_ucla_v2" "$DST/mctp_dataset_f13_isles24_ucla_v2" \
    --split "$GAN_RUN/fold_0/split.json" --side all || fail "staging the dataset"

echo "2. gan_bpm's code"
rm -rf "$DST/gan_bpm"
cp -r "$SRC/gan_bpm" "$DST/gan_bpm" || fail "copying gan_bpm"

echo "3. gan_bpm's 5 folds: split.json and checkpoints/best.pt"
for k in 0 1 2 3 4; do
    mkdir -p "$DST/gan_run/fold_$k/checkpoints"
    cp "$GAN_RUN/fold_$k/split.json" "$DST/gan_run/fold_$k/" || fail "copying fold $k's split.json"
    cp "$GAN_RUN/fold_$k/checkpoints/best.pt" "$DST/gan_run/fold_$k/checkpoints/" || fail "copying fold $k's best.pt"
done

echo "==================================================================="
echo "4. checks"
for k in 0 1 2 3 4; do
    a=$(sha256sum < "$GAN_RUN/fold_$k/checkpoints/best.pt" | cut -c1-16)
    b=$(sha256sum < "$DST/gan_run/fold_$k/checkpoints/best.pt" | cut -c1-16)
    [ "$a" = "$b" ] || fail "fold $k's best.pt differs after the copy ($a vs $b)"
    echo "fold $k best.pt sha256 $b"
done
python - "$DST/mctp_dataset_f13_isles24_ucla_v2" "$DST/gan_run/fold_0/split.json" <<'EOF' || fail "a file of a split case is missing"
import json, sys
from pathlib import Path
import pandas as pd
root, split = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
cases = set(split["train_case_ids"]) | set(split["val_case_ids"])
rows = pd.read_parquet(root / "tmax_index.parquet")
rows = rows[rows["case_id"].isin(cases)]
names = list(rows["input_relpath"]) + list(rows["label_relpath"])
missing = [n for n in names if not (root / n).is_file()]
print(f"{len(cases)} cases in the split, {len(rows)} slices; {len(missing)} of their files missing")
sys.exit(1 if missing else 0)
EOF
du -sh "$DST"/*

{ echo "staged $(date -Is) on $(hostname) from $SRC and $GAN_RUN"; } > "$DST/STAGED_OK"
echo "wrote $DST/STAGED_OK"
