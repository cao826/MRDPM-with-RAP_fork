#!/bin/bash
# Invoked by ctp_only_train_h100.sub as: ctp_only_train_h100.sh FOLD MODE JOBID [RESUME]
#   MODE   debug: run.py -d (1 epoch on 4 slices, a monitor pass on 2). full: the whole recipe.
#   JOBID  Condor's cluster.process: names this job's run folder on /home.
#   RESUME a checkpoint prefix on /radraid2 (.../checkpoint/<epoch>) to continue from.
#
# The data, gan_bpm's code and the fold's generator are already on the H100 box's own disk,
# /home/colivares/mrdpm_ctp_only/stage (condor/ctp_only_stage_home.sub; ctp_only_decisions.md
# #17). The job reads them there and writes its run folder there too. Every 10 minutes, and at
# the end, it moves finished checkpoints and copies everything else to
# /radraid2/colivares/mrdpm_ctp_only/runs/<experiment>/, the only place 14958 can see.
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
RESUME="${4:-}"
H=/home/colivares/mrdpm_ctp_only
STAGE="$H/stage"
EXP_BASE="$H/experiments/$JOBID"
OUT=/radraid2/colivares/mrdpm_ctp_only/runs
CODE="$W/MRDPM-with-RAP_fork"

say() { echo "$*" | tee -a "$W/job_summary.txt"; }
fail() { say "FATAL: $*"; exit 1; }

case "$MODE" in
    debug) DEBUG=-d ;;
    full) DEBUG= ;;
    *) fail "mode must be debug or full, not '$MODE'" ;;
esac

say "host: $(hostname)   date: $(date -Is)   fold $FOLD   mode $MODE   job $JOBID   resume ${RESUME:-<none>}"
nvidia-smi -L || echo "(nvidia-smi -L failed)"
# run.py sets CUDA_VISIBLE_DEVICES to the config's gpu_ids ("0"), which is only this job's GPU
# when the container sees exactly one.
n_gpus=$(env -u CUDA_VISIBLE_DEVICES python -c "import torch; print(torch.cuda.device_count())")
[ "$n_gpus" = 1 ] || fail "the container sees $n_gpus GPUs; run.py would take GPU 0, which may not be this job's"
[ -d "$CODE" ] || fail "no checkout at $CODE (transfer_input_files)"
say "MRDPM code: $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null || echo '<git unavailable>')"
[ -f "$STAGE/STAGED_OK" ] || fail "no $STAGE/STAGED_OK: run condor/ctp_only_stage_home.sub first"
say "stage: $(cat "$STAGE/STAGED_OK")"
[ -f "$STAGE/gan_run/fold_$FOLD/checkpoints/best.pt" ] || fail "fold $FOLD isn't staged"
df -h /home | tail -1
mkdir -p "$EXP_BASE" || fail "cannot create $EXP_BASE"

RESUME_LOCAL=
if [ -n "$RESUME" ]; then
    mkdir -p "$H/resume/$JOBID"
    for s in _Network.pth _Network_ema.pth .state; do
        cp "$RESUME$s" "$H/resume/$JOBID/" || fail "copying $RESUME$s"
    done
    RESUME_LOCAL="$H/resume/$JOBID/$(basename "$RESUME")"
fi

echo "==================================================================="
echo "1. the job's config"
python - "$CODE/config/ctp_only_f13_tmax_fold0.json" "$W/job_config.json" "$STAGE" "$EXP_BASE" "$FOLD" "$RESUME_LOCAL" <<'PYEOF' || fail "writing the job config"
import json, sys
src, dst, stage, base, fold, resume = sys.argv[1:]
cfg = json.loads("".join(line.split("//")[0] + "\n" for line in open(src)))    # as core/praser.py reads it
data, gan, run = f"{stage}/mctp_dataset_f13_isles24_ucla_v2", f"{stage}/gan_bpm", f"{stage}/gan_run/fold_{fold}"
for phase, key in (("train", "which_dataset"), ("train", "which_val_dataset"), ("test", "which_dataset")):
    cfg["datasets"][phase][key]["args"].update(mctp_root=data, gan_bpm_root=gan, split_json=f"{run}/split.json")
cfg["model"]["which_networks"][0]["args"]["initial_predictor"].update(
    gan_bpm_root=gan, checkpoint=f"{run}/checkpoints/best.pt", expect_fold=int(fold))
cfg["name"] = f"ctp_only_f13_tmax_fold{fold}"
cfg["path"].update(base_dir=base, resume_state=resume or None)
json.dump(cfg, open(dst, "w"), indent=1)
print(f"name {cfg['name']}, base_dir {base}, resume_state {cfg['path']['resume_state']}")
PYEOF

# Finished checkpoints move to $OUT; the newest set stays until a newer one exists, since it may
# still be being written ("final" moves everything). Everything else is copied.
sync_out() {
    local exp dst ck e newest
    exp=$(ls -d "$EXP_BASE"/*/ 2>/dev/null | head -1)
    [ -n "$exp" ] || return 0
    exp="${exp%/}"
    dst="$OUT/$(basename "$exp")"
    ck="$exp/checkpoint"
    mkdir -p "$dst/checkpoint"
    newest=$(ls "$ck" 2>/dev/null | sed -n 's/^\([0-9]*\)\.state$/\1/p' | sort -n | tail -1)
    for e in $(ls "$ck" 2>/dev/null | sed -n 's/^\([0-9]*\)\.state$/\1/p' | sort -n); do
        if [ "${1:-}" = final ] || [ "$e" -lt "$newest" ]; then
            mv "$ck/${e}_Network.pth" "$ck/${e}_Network_ema.pth" "$ck/${e}.state" "$dst/checkpoint/"
        fi
    done
    for item in "$exp"/*; do
        [ "$(basename "$item")" = checkpoint ] || cp -ru "$item" "$dst/"
    done
    cp -u "$W/train.out" "$W/job_config.json" "$dst/" 2>/dev/null
    RUN_OUT="$dst"
}

echo "==================================================================="
echo "2. train (run.py's own output: train.out, copied next to the results)"
cd "$CODE" || fail "cd $CODE"
python run.py -p train $DEBUG -c "$W/job_config.json" > "$W/train.out" 2>&1 &
PID=$!
RUN_OUT=
while kill -0 "$PID" 2>/dev/null; do
    for _ in $(seq 60); do
        kill -0 "$PID" 2>/dev/null || break
        sleep 10
    done
    sync_out
    last=$(tail -n 1 "$EXP_BASE"/*/health/train.csv 2>/dev/null)
    echo "$(date '+%m-%d %H:%M')  ${last:-<no epoch finished yet>}"
done
wait "$PID"
status=$?
sync_out final

echo "==================================================================="
say "train exited $status; results in ${RUN_OUT:-<none>} (run folder on the H100 box: $EXP_BASE)"
tr '\r' '\n' < "$W/train.out" | grep -v "it/s]\|s/it]\|^ *$" | tail -n 25
if [ -n "$RUN_OUT" ]; then
    for f in health/train.csv health/monitor.csv; do
        [ -f "$RUN_OUT/$f" ] && { say "--- $f (header and last rows)"; head -1 "$RUN_OUT/$f" | tee -a "$W/job_summary.txt"; tail -n +2 "$RUN_OUT/$f" | tail -n 3 | tee -a "$W/job_summary.txt"; }
    done
    say "--- checkpoints: $(ls "$RUN_OUT/checkpoint" | tr '\n' ' ')"
fi
exit "$status"
