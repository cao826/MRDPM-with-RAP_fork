#!/bin/bash
# Invoked by ctp_only_probe_remote.sub, on a machine without 14958's paths. Make the bundle it
# reads first, on 14958 (CPU, a few minutes):
#
#   cd /raid/colivares/github_repos/MRDPM-with-RAP_fork && git pull
#   B=/radraid2/colivares/mrdpm_ctp_only/probe_bundle && rm -rf "$B" && mkdir -p "$B/mrdpm"
#   git archive HEAD | tar -x -C "$B/mrdpm" && git rev-parse --short HEAD > "$B/COMMIT"
#   docker run --rm --user "$(id -u):$(id -g)" -e NVIDIA_VISIBLE_DEVICES=void -e PYTHONDONTWRITEBYTECODE=1 \
#     -v /raid:/raid -v /radraid2:/radraid2 -v "$PWD:/repo" -w /repo registry.cvib.ucla.edu/colivares:mrdpm-env-v1 \
#     python scripts/ctp_only_gpu_probe.py --make-bundle "$B"
set -u

export HOME="${_CONDOR_SCRATCH_DIR:-/tmp}"
export MPLCONFIGDIR="$HOME/.matplotlib"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

B="${_CONDOR_SCRATCH_DIR:-$PWD}/probe_bundle"

echo "host: $(hostname)   date: $(date -Is)   CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "(nvidia-smi failed)"
[ -f "$B/batch.pt" ] || { echo "FATAL: no bundle at $B; make it on 14958 first (this script's header)"; exit 1; }
echo "MRDPM code: $(cat "$B/COMMIT" 2>/dev/null || echo '<unknown>')"
echo "==================================================================="

# Batches up to what 95 GB might hold: training took 12.8 GiB at batch 1 and 24.3 at 2 on 14958.
# The output stays in the bundle, inside the scratch dir: everything needed is printed here.
cd "$B/mrdpm" || exit 1
python scripts/ctp_only_gpu_probe.py --bundle "$B" --out "$B/out" --train-batches 1,2,4,6,8 --sample-batches 1,4,8,16,32
status=$?
echo "==================================================================="
echo "probe exited $status"
exit "$status"
