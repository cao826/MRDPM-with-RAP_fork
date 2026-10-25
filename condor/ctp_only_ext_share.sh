#!/bin/bash
# Invoked by ctp_only_ext_share.sub, on 14958: scripts/ctp_only_ext_share.py, L_ext's share of
# gan_bpm's generator gradient at each fold's best.pt and at initialization. Arguments go to
# the script, e.g. --fold 0 --no-init, --target tmax. Results go to
# /radraid2/colivares/mrdpm_ctp_only/ext_share/<date_time>_<target>_<recipe>/.
#
# Reads this box's /raid (the CV data, gan_bpm's checkout) in place, as
# ctp_only_core_check.sh does; gan_bpm is imported read-only.
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
python scripts/ctp_only_ext_share.py --device cuda "$@"
status=$?
echo "ctp_only_ext_share.py exited $status   (0 done, 2 refused or a check failed)"
exit "$status"
