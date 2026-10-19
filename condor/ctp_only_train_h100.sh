#!/bin/bash
# Invoked by ctp_only_train_h100.sub as: ctp_only_train_h100.sh FOLD MODE [RESUME]
#   MODE   debug: run.py -d (1 epoch on 2 slices, a monitor pass on 2). full: the whole recipe.
#   RESUME a checkpoint prefix on /radraid2 (.../checkpoint/<epoch>) to continue from.
#
# 1. Copy into the workspace: this fold's training cases from the staged data (not the whole
#    31 GB root), gan_bpm's code and the fold's generator.
# 2. Write a config pointing at those copies (and at the fold).
# 3. Train. Every 10 minutes, and at the end, move finished checkpoints and copy everything else
#    to /radraid2/colivares/mrdpm_ctp_only/runs/<experiment>/.
set -u

W="${_CONDOR_SCRATCH_DIR:-$PWD}"
: > "$W/job_summary.txt"            # the one file Condor sends back (named in the .sub)
export HOME="$W"
export MPLCONFIGDIR="$W/.matplotlib"
export PYTHONDONTWRITEBYTECODE=1    # this repo tracks __pycache__; don't add to it
export PYTHONUNBUFFERED=1

FOLD="${1:-0}"
MODE="${2:-debug}"
RESUME="${3:-}"
STAGE=/radraid2/colivares/mrdpm_ctp_only/stage
GAN_RUN=/radraid2/colivares/matrix/gan_bpm_tmax_f13uclav2_t12_adv0_k5
OUT=/radraid2/colivares/mrdpm_ctp_only/runs
CODE="$W/MRDPM-with-RAP_fork"

say() { echo "$*" | tee -a "$W/job_summary.txt"; }
fail() { say "FATAL: $*"; exit 1; }

case "$MODE" in
    debug) DEBUG=-d ;;
    full) DEBUG= ;;
    *) fail "mode must be debug or full, not '$MODE'" ;;
esac

say "host: $(hostname)   date: $(date -Is)   fold $FOLD   mode $MODE   resume ${RESUME:-<none>}"
nvidia-smi -L || echo "(nvidia-smi -L failed)"
# run.py sets CUDA_VISIBLE_DEVICES to the config's gpu_ids ("0"), which is only this job's GPU
# when the container sees exactly one.
n_gpus=$(env -u CUDA_VISIBLE_DEVICES python -c "import torch; print(torch.cuda.device_count())")
[ "$n_gpus" = 1 ] || fail "the container sees $n_gpus GPUs; run.py would take GPU 0, which may not be this job's"
[ -d "$CODE" ] || fail "no checkout at $CODE (transfer_input_files)"
say "MRDPM code: $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null || echo '<git unavailable>')"
df -h "$W" | tail -1

echo "==================================================================="
echo "1. copy in"
t0=$(date +%s)
mkdir -p "$W/data" "$W/gan_run/fold_$FOLD/checkpoints" "$W/experiments"
cp "$GAN_RUN/fold_$FOLD/split.json" "$W/gan_run/fold_$FOLD/" || fail "copying fold $FOLD's split.json"
cp "$GAN_RUN/fold_$FOLD/checkpoints/best.pt" "$W/gan_run/fold_$FOLD/checkpoints/" || fail "copying fold $FOLD's best.pt"
cp -r "$STAGE/gan_bpm" "$W/gan_bpm" || fail "copying gan_bpm from $STAGE"
python "$CODE/scripts/ctp_only_stage_data.py" "$STAGE/mctp_dataset_f13_isles24_ucla_v2" \
    "$W/data/mctp_dataset_f13_isles24_ucla_v2" --split "$W/gan_run/fold_$FOLD/split.json" --side train \
    | tee -a "$W/job_summary.txt"
[ "${PIPESTATUS[0]}" = 0 ] || fail "staging fold $FOLD's training cases"
RESUME_LOCAL=
if [ -n "$RESUME" ]; then
    mkdir -p "$W/resume"
    for s in _Network.pth _Network_ema.pth .state; do
        cp "$RESUME$s" "$W/resume/" || fail "copying $RESUME$s"
    done
    RESUME_LOCAL="$W/resume/$(basename "$RESUME")"
fi
[ -z "$(find "$W/data" -type l | head -1)" ] || fail "the staged dataset has symlinks; stage it with rsync -aL"
say "copied $(du -sh "$W/data" | cut -f1) of data and the fold-$FOLD generator in $(( $(date +%s) - t0 )) s"

echo "==================================================================="
echo "2. the job's config"
python - "$CODE/config/ctp_only_f13_tmax_fold0.json" "$W/job_config.json" "$W" "$FOLD" "$RESUME_LOCAL" <<'PYEOF' || fail "writing the job config"
import json, sys
src, dst, W, fold, resume = sys.argv[1:]
cfg = json.loads("".join(line.split("//")[0] + "\n" for line in open(src)))    # as core/praser.py reads it
data, gan, run = f"{W}/data/mctp_dataset_f13_isles24_ucla_v2", f"{W}/gan_bpm", f"{W}/gan_run/fold_{fold}"
for phase, key in (("train", "which_dataset"), ("train", "which_val_dataset"), ("test", "which_dataset")):
    cfg["datasets"][phase][key]["args"].update(mctp_root=data, gan_bpm_root=gan, split_json=f"{run}/split.json")
cfg["model"]["which_networks"][0]["args"]["initial_predictor"].update(
    gan_bpm_root=gan, checkpoint=f"{run}/checkpoints/best.pt", expect_fold=int(fold))
cfg["name"] = f"ctp_only_f13_tmax_fold{fold}"
cfg["path"].update(base_dir=f"{W}/experiments", resume_state=resume or None)
json.dump(cfg, open(dst, "w"), indent=1)
print(f"name {cfg['name']}, resume_state {cfg['path']['resume_state']}")
PYEOF

# Finished checkpoints move to $OUT; the newest set stays until a newer one exists, since it may
# still be being written ("final" moves everything). Everything else is small and is copied.
sync_out() {
    local exp dst ck e newest
    exp=$(ls -d "$W"/experiments/*/ 2>/dev/null | head -1)
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
echo "3. train (run.py's own output: train.out, copied next to the results)"
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
    last=$(tail -n 1 "$W"/experiments/*/health/train.csv 2>/dev/null)
    echo "$(date '+%m-%d %H:%M')  ${last:-<no epoch finished yet>}"
done
wait "$PID"
status=$?
sync_out final

echo "==================================================================="
say "train exited $status; results in ${RUN_OUT:-<none>}"
tr '\r' '\n' < "$W/train.out" | grep -v "it/s]\|s/it]\|^ *$" | tail -n 25
if [ -n "$RUN_OUT" ]; then
    for f in health/train.csv health/monitor.csv; do
        [ -f "$RUN_OUT/$f" ] && { say "--- $f (first and last rows)"; head -2 "$RUN_OUT/$f" | tee -a "$W/job_summary.txt"; tail -n 3 "$RUN_OUT/$f" | tee -a "$W/job_summary.txt"; }
    done
    say "--- checkpoints: $(ls "$RUN_OUT/checkpoint" | tr '\n' ' ')"
fi
exit "$status"
