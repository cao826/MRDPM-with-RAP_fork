#!/bin/bash
# Invoked by ctp_only_isles18_ganbpm_maps.sub, on 14958: gan_bpm's fold models (best.pt of
# each fold; λ_adv = 0, LSGAN, WGAN-GP; rCBF and Tmax) on all 94 ISLES 2018 scans, written
# as stage-1 maps to /radraid2/colivares/mrdpm_ctp_only/isles2018/maps/
# (scripts/ctp_only_isles18_maps.py gan-bpm, which first checks itself against stage 1's
# stored maps). Arguments go to the script, e.g. --target rcbf, --overwrite.
#
# Reads this box's /raid (the ISLES 2018 root, gan_bpm's checkout) in place, as gan_bpm's
# own stage 1 (condor/external_maps.sh) did; gan_bpm is imported read-only.
set -u
set -o pipefail

CHECKOUT=/raid/colivares/github_repos/MRDPM-with-RAP_fork

export HOME="${_CONDOR_SCRATCH_DIR:-/tmp}"
export MPLCONFIGDIR="$HOME/.matplotlib"
export PYTHONDONTWRITEBYTECODE=1    # neither repo should gain .pyc files
export PYTHONUNBUFFERED=1

echo "host: $(hostname)   date: $(date -Is)   arguments: ${*:-<none>}"
nvidia-smi -L || echo "(nvidia-smi -L failed)"
cd "$CHECKOUT" || { echo "FATAL: cannot cd to $CHECKOUT"; exit 1; }
echo "MRDPM: $(git rev-parse --short HEAD 2>/dev/null || echo '<git unavailable>')"
python scripts/ctp_only_isles18_maps.py gan-bpm --device cuda "$@"
status=$?
echo "ctp_only_isles18_maps.py exited $status   (0 done, 2 refused)"
exit "$status"
