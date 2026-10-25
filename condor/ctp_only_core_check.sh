#!/bin/bash
# Invoked by ctp_only_core_check.sub, on 14958: scripts/ctp_only_core_check.py, the "right
# place or wrong level" check on each fold's held-out CV cases (ctp_only_decisions.md #28).
# Arguments go to the script, e.g. --fold 0 --fold 1, --target tmax. Results go to
# /radraid2/colivares/mrdpm_ctp_only/cv_core_check/<date_time>_<target>/.
#
# Reads this box's /raid (the CV data, gan_bpm's checkout) in place, as
# ctp_only_isles18_ganbpm_maps.sh does; gan_bpm is imported read-only.
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
python scripts/ctp_only_core_check.py --device cuda "$@"
status=$?
echo "ctp_only_core_check.py exited $status   (0 done, 2 refused or a control failed)"
exit "$status"
