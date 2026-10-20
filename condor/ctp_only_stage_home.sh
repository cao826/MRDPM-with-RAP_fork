#!/bin/bash
# Invoked by ctp_only_stage_home.sub, on the H100 box. Copies from the /radraid2 stage to the
# box's own disk, checks the copy, and writes the target's done-marker last.
#
# MRDPM_TARGET (the .sub's `target`) picks what is staged:
#   tmax (default)  the Tmax files, gan_bpm's code, the Tmax generators in gan_run/; STAGED_OK
#   rcbf            the rCBF files into the same root, the rCBF generators in gan_run_rcbf/;
#                   STAGED_OK_rcbf. It reuses Tmax's copy of gan_bpm's code and mask cache, and
#                   touches nothing of Tmax's, so Tmax jobs can run while it copies.
# Don't rerun the tmax staging while a training job runs: it replaces gan_bpm's code under it.
set -u

export HOME="${_CONDOR_SCRATCH_DIR:-/tmp}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

TARGET="${MRDPM_TARGET:-tmax}"
SRC=/radraid2/colivares/mrdpm_ctp_only/stage
DST=/home/colivares/mrdpm_ctp_only/stage
STAGER="${_CONDOR_SCRATCH_DIR:-$PWD}/ctp_only_stage_data.py"

fail() { echo "FATAL: $*"; exit 1; }

case "$TARGET" in
    tmax) GAN_RUN=/radraid2/colivares/matrix/gan_bpm_tmax_f13uclav2_t12_adv0_k5
          RUN_DIR="$DST/gan_run"; MARK="$DST/STAGED_OK" ;;
    rcbf) GAN_RUN=/radraid2/colivares/matrix/gan_bpm_rcbf_f13uclav2_adv0_k5
          RUN_DIR="$DST/gan_run_rcbf"; MARK="$DST/STAGED_OK_rcbf" ;;
    *) fail "target must be tmax or rcbf, not '$TARGET'" ;;
esac

echo "host: $(hostname)   date: $(date -Is)   target $TARGET"
df -h /home | tail -1
if [ "$TARGET" != tmax ]; then
    [ -f "$DST/STAGED_OK" ] || fail "stage tmax first: $TARGET reuses its copy of gan_bpm's code"
fi
rm -f "$MARK"                   # until this run has checked its copy
mkdir -p "$DST" || fail "cannot create $DST"
echo "==================================================================="

echo "1. the dataset: every case of the 5-fold split, $TARGET only"
python "$STAGER" "$SRC/mctp_dataset_f13_isles24_ucla_v2" "$DST/mctp_dataset_f13_isles24_ucla_v2" \
    --split "$GAN_RUN/fold_0/split.json" --side all --label-type "$TARGET" || fail "staging the dataset"

if [ "$TARGET" = tmax ]; then
    echo "2. gan_bpm's code"
    rm -rf "$DST/gan_bpm"
    cp -r "$SRC/gan_bpm" "$DST/gan_bpm" || fail "copying gan_bpm"
else
    echo "2. gan_bpm's code: Tmax's copy, left as it is ($(cat "$DST/STAGED_OK"))"
fi

echo "3. gan_bpm's 5 folds: split.json and checkpoints/best.pt"
for k in 0 1 2 3 4; do
    mkdir -p "$RUN_DIR/fold_$k/checkpoints"
    cp "$GAN_RUN/fold_$k/split.json" "$RUN_DIR/fold_$k/" || fail "copying fold $k's split.json"
    cp "$GAN_RUN/fold_$k/checkpoints/best.pt" "$RUN_DIR/fold_$k/checkpoints/" || fail "copying fold $k's best.pt"
done

echo "==================================================================="
echo "4. checks"
for k in 0 1 2 3 4; do
    a=$(sha256sum < "$GAN_RUN/fold_$k/checkpoints/best.pt" | cut -c1-16)
    b=$(sha256sum < "$RUN_DIR/fold_$k/checkpoints/best.pt" | cut -c1-16)
    [ "$a" = "$b" ] || fail "fold $k's best.pt differs after the copy ($a vs $b)"
    echo "fold $k best.pt sha256 $b"
done
python - "$DST/mctp_dataset_f13_isles24_ucla_v2" "$RUN_DIR/fold_0/split.json" "$TARGET" <<'EOF' || fail "a file of a split case is missing"
import json, sys
from pathlib import Path
import pandas as pd
root, split, target = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), sys.argv[3]
cases = set(split["train_case_ids"]) | set(split["val_case_ids"])
rows = pd.read_parquet(root / f"{target}_index.parquet")
rows = rows[rows["case_id"].isin(cases)]
names = list(rows["input_relpath"]) + list(rows["label_relpath"])
missing = [n for n in names if not (root / n).is_file()]
print(f"{target}: {len(cases)} cases in the split, {len(rows)} slices; {len(missing)} of their files missing")
sys.exit(1 if missing else 0)
EOF
du -sh "$DST"/*

{ echo "staged $TARGET $(date -Is) on $(hostname) from $SRC and $GAN_RUN"; } > "$MARK"
echo "wrote $MARK"
