#!/bin/bash
# Invoked by ctp_only_gpu_checks.sub. Reads the box's checkout of this repo (git pull first)
# and writes only summary_gpu.json under $OUT.
set -u

# The execute node's $HOME is not writable inside the container (gan_bpm condor/run_cv.sh).
export HOME="${_CONDOR_SCRATCH_DIR:-/tmp}"
export MPLCONFIGDIR="$HOME/.matplotlib"
export PYTHONDONTWRITEBYTECODE=1    # this repo tracks __pycache__; don't add to it

REPO="/raid/colivares/github_repos/MRDPM-with-RAP_fork"
OUT="/radraid2/colivares/mrdpm_ctp_only/box_checks"

echo "host: $(hostname)   date: $(date -Is)   CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || echo "(nvidia-smi -L failed)"
cd "$REPO" || { echo "FATAL: cannot cd to $REPO"; exit 1; }
echo "MRDPM checkout: $(git rev-parse --short HEAD 2>/dev/null || echo '<not a git checkout>')"
echo "==================================================================="

python scripts/ctp_only_box_checks.py --part gpu --workers 4 --out "$OUT"
status=$?
echo "==================================================================="
echo "gpu checks exited $status"
exit "$status"
