#!/bin/bash
# Invoked by ctp_only_gpu_checks.sub. Reads the box's checkout of this repo (git pull first)
# and writes only summary_gpu.json under $OUT. MRDPM_TARGET (the .sub's `target`, tmax by
# default) picks config/ctp_only_f13_<target>_fold0.json; rcbf's summary goes to $OUT/rcbf.
set -u

# The execute node's $HOME is not writable inside the container (gan_bpm condor/run_cv.sh).
export HOME="${_CONDOR_SCRATCH_DIR:-/tmp}"
export MPLCONFIGDIR="$HOME/.matplotlib"
export PYTHONDONTWRITEBYTECODE=1    # this repo tracks __pycache__; don't add to it

REPO="/raid/colivares/github_repos/MRDPM-with-RAP_fork"
OUT="/radraid2/colivares/mrdpm_ctp_only/box_checks"
TARGET="${MRDPM_TARGET:-tmax}"
case "$TARGET" in
    tmax) ;;
    rcbf) OUT="$OUT/rcbf" ;;
    *) echo "FATAL: target must be tmax or rcbf, not '$TARGET'"; exit 1 ;;
esac

echo "host: $(hostname)   date: $(date -Is)   target $TARGET   CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || echo "(nvidia-smi -L failed)"
cd "$REPO" || { echo "FATAL: cannot cd to $REPO"; exit 1; }
echo "MRDPM checkout: $(git rev-parse --short HEAD 2>/dev/null || echo '<not a git checkout>')"
echo "==================================================================="

python scripts/ctp_only_box_checks.py --part gpu --workers 4 --out "$OUT" \
    --config "config/ctp_only_f13_${TARGET}_fold0.json"
status=$?
echo "==================================================================="
echo "gpu checks exited $status"
exit "$status"
