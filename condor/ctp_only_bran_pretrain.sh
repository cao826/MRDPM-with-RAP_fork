#!/bin/bash
# Invoked by ctp_only_bran_pretrain.sub as: ctp_only_bran_pretrain.sh FOLD MODE JOBID
#   FOLD   0-4
#   MODE   debug: 1 epoch on 8 training slices, 4 held-out maps, into bran/debug_<name>/.
#          full: the config's recipe (50 epochs), into bran/<name>/.
#   JOBID  Condor's cluster.process: names this job's folder on /home.
# and in the environment (the .sub's macros): MRDPM_TARGET (rcbf by default) and MRDPM_VARIANT,
# the arm whose config is used: config/ctp_only_f13_<target>_<variant>_fold0.json.
#
# scripts/pretrain_bran.py (bran_ext_plan.md §2) on the data staged on this box's own disk,
# <root>/stage (ctp_only_stage_home.sub with this box as gpu_host; <root> is
# /raid/colivares/mrdpm_ctp_only on 14959, /home/colivares/mrdpm_ctp_only elsewhere), with the
# fold's split.json from the staged gan_bpm run, as the training jobs read them on the H100.
# It works in <root>/bran_runs/<job>/ on this box, so a job
# Condor runs again resumes from its last epoch, and at the end copies everything but the resume
# file to /radraid2/colivares/mrdpm_ctp_only/bran/<name>/fold_<k>/ under a temporary name, then
# renames it. It refuses to start if that folder exists: a finished pretraining stays as it is.
set -u

W="${_CONDOR_SCRATCH_DIR:-$PWD}"
: > "$W/job_summary.txt"            # the one file Condor sends back (named in the .sub)
export HOME="$W"
export MPLCONFIGDIR="$W/.matplotlib"
export PYTHONDONTWRITEBYTECODE=1    # this repo tracks __pycache__; don't add to it
export PYTHONUNBUFFERED=1

FOLD="${1:-0}"
MODE="${2:-debug}"
JOBID="${3:-job$$}"
TARGET="${MRDPM_TARGET:-rcbf}"
VARIANT="${MRDPM_VARIANT:-none}"
# This box's own disk for our files (decisions #37): the Quadro box 14959 has no /home/colivares
# but a local /raid/colivares; the H100 box and the L40S box 35840 use /home/colivares.
box=$(grep -m1 '^Machine = ' "${_CONDOR_MACHINE_AD:-/dev/null}" 2>/dev/null | cut -d'"' -f2)
[ -n "$box" ] || box=$(hostname)        # inside docker: colivares-<job>-<machine>
case "$box" in
    *REDLRADADM14959*) H=/raid/colivares/mrdpm_ctp_only ;;
    *) H=/home/colivares/mrdpm_ctp_only ;;
esac
STAGE="$H/stage"
LOCAL="$H/bran_runs/$JOBID"
CODE="$W/MRDPM-with-RAP_fork"

say() { echo "$*" | tee -a "$W/job_summary.txt"; }
fail() { say "FATAL: $*"; exit 1; }

case "$MODE" in
    debug) DEBUG=--debug ;;
    full) DEBUG= ;;
    *) fail "mode must be debug or full, not '$MODE'" ;;
esac
case "$TARGET" in
    tmax) GAN_RUN="$STAGE/gan_run"; STAGED="$STAGE/STAGED_OK" ;;
    rcbf) GAN_RUN="$STAGE/gan_run_rcbf"; STAGED="$STAGE/STAGED_OK_rcbf" ;;
    *) fail "target must be tmax or rcbf, not '$TARGET'" ;;
esac
case "$FOLD" in [0-4]) ;; *) fail "fold must be 0-4, not '$FOLD'" ;; esac
[ "$VARIANT" != none ] || fail "give the arm: -append 'variant = bran_l1' (or bran_ext, bran_extlow)"
CONFIG="$CODE/config/ctp_only_f13_${TARGET}_${VARIANT}_fold0.json"

say "host: $(hostname)   date: $(date -Is)   target $TARGET   variant $VARIANT   fold $FOLD   mode $MODE   job $JOBID"
say "box $box: data in $STAGE, work in $LOCAL"
nvidia-smi -L || echo "(nvidia-smi -L failed)"
[ -d "$CODE" ] || fail "no checkout at $CODE (transfer_input_files)"
say "MRDPM code: $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null || echo '<git unavailable>')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /' | tee -a "$W/job_summary.txt"
# Never on the CPU (decisions #38): 14959's first debug job got a GPU slot but no GPU in its
# container, and pretrain_bran.py fell back to the CPU without a word.
n_gpus=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [ "${n_gpus:-0}" -lt 1 ]; then
    say "no GPU in this container: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}  NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}  /dev/nvidia*: $(ls /dev/nvidia* 2>/dev/null | tr '\n' ' ')"
    say "slot: $(grep -E '^(AssignedGPUs|GPUs|TotalGPUs) ' "${_CONDOR_MACHINE_AD:-/dev/null}" 2>/dev/null | tr '\n' ' ')"
    fail "torch sees no GPU on $box: refusing to pretrain on the CPU (condor/probe_gpu.sub tests a box)"
fi
[ -f "$CONFIG" ] || fail "no $CONFIG"
[ -f "$STAGED" ] || fail "no $STAGED on $(hostname): stage it there first (ctp_only_stage_home.sub, target $TARGET, gpu_host this box)"
[ -f "$GAN_RUN/fold_$FOLD/split.json" ] || fail "fold_$FOLD of $TARGET isn't staged"
mkdir -p "$LOCAL" || fail "cannot create $LOCAL"
[ -n "$(ls "$LOCAL" 2>/dev/null)" ] && say "restarted by Condor: $LOCAL holds $(ls "$LOCAL" | tr '\n' ' ')"

echo "==================================================================="
echo "1. the job's config"
DST=$(python - "$CONFIG" "$W/job_config.json" "$STAGE" "$FOLD" "$MODE" "$GAN_RUN/fold_$FOLD" <<'PYEOF'
import json, sys
from pathlib import Path
src, dst, stage, fold, mode, run = sys.argv[1:]
cfg = json.loads("".join(line.split("//")[0] + "\n" for line in open(src)))    # as core/praser.py reads it
for phase, key in (("train", "which_dataset"), ("train", "which_val_dataset"), ("test", "which_dataset")):
    cfg["datasets"][phase][key]["args"].update(mctp_root=f"{stage}/mctp_dataset_f13_isles24_ucla_v2",
                                               gan_bpm_root=f"{stage}/gan_bpm", split_json=f"{run}/split.json")
ip = cfg["model"]["which_networks"][0]["args"]["initial_predictor"]
if ip.get("type") != "bran":
    sys.exit(f"{src}: initial_predictor.type is {ip.get('type')!r}: nothing to pretrain")
ck = Path(ip["checkpoint"])                      # .../bran/<name>/fold_0/bran_G.pth
name = ck.parent.parent.name
out = ck.parent.parent.parent / (f"debug_{name}" if mode == "debug" else name) / f"fold_{fold}"
ip.update(checkpoint=str(out / ck.name), expect_fold=int(fold))
json.dump(cfg, open(dst, "w"), indent=1)
print(out)
PYEOF
) || fail "writing the job config"
say "BRAN: $(basename "$(dirname "$DST")") fold $FOLD -> $DST"
[ -e "$DST" ] && fail "$DST exists: this BRAN is pretrained already (move it aside to redo it)"

echo "==================================================================="
echo "2. pretrain (scripts/pretrain_bran.py's output: pretrain.out, copied next to the results)"
cd "$CODE" || fail "cd $CODE"
python scripts/pretrain_bran.py -c "$W/job_config.json" --out "$LOCAL" $DEBUG --device cuda --workers 8 > "$W/pretrain.out" 2>&1 &
PID=$!
while kill -0 "$PID" 2>/dev/null; do
    for _ in $(seq 60); do
        kill -0 "$PID" 2>/dev/null || break
        sleep 10
    done
    echo "$(date '+%m-%d %H:%M')  $(grep -a 'epoch [0-9]*/' "$W/pretrain.out" | tail -n 1)"
done
wait "$PID"
status=$?
cp "$W/pretrain.out" "$W/job_config.json" "$LOCAL/"

echo "==================================================================="
say "pretrain_bran.py exited $status"
tail -n 15 "$W/pretrain.out" | tee -a "$W/job_summary.txt"
[ "$status" = 0 ] || fail "pretraining failed; its folder stays in $LOCAL"
[ -f "$LOCAL/bran_G.pth" ] || fail "no $LOCAL/bran_G.pth"
[ -e "$DST" ] && fail "$DST appeared while this job ran; this job's BRAN stays in $LOCAL"

tmp="$(dirname "$DST")/.copying_$(basename "$DST")_$JOBID"
mkdir -p "$tmp" || fail "cannot create $tmp"
(cd "$LOCAL" && tar --exclude='./last_state.pt' --exclude='./last_state.pt.tmp' -cf - .) | (cd "$tmp" && tar -xf -) \
    || fail "copying $LOCAL to $tmp"
n_here=$(cd "$LOCAL" && find . -type f ! -name 'last_state.pt*' | wc -l)
n_there=$(cd "$tmp" && find . -type f | wc -l)
[ "$n_here" = "$n_there" ] || fail "copied $n_there of $n_here files to $tmp; left there"
mv "$tmp" "$DST" || fail "renaming $tmp to $DST"
say "BRAN in $DST: $(ls "$DST" | tr '\n' ' ')"
say "held-out maps: $(ls "$DST/heldout" | grep -c '^Init_')"
say "--- pretrain_log.csv (header and last rows)"
head -1 "$DST/pretrain_log.csv" | tee -a "$W/job_summary.txt"
tail -n +2 "$DST/pretrain_log.csv" | tail -n 3 | tee -a "$W/job_summary.txt"
exit 0
