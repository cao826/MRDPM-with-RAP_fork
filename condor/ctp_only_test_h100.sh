#!/bin/bash
# Invoked by ctp_only_test_h100.sub as: ctp_only_test_h100.sh RUN TESTSET MODE JOBID [EPOCH]
#   RUN      a training run folder on /radraid2, e.g.
#            /radraid2/colivares/mrdpm_ctp_only/runs/train_ctp_only_f13_tmax_fold0_261019_202928;
#            its name gives the target and the fold (or all)
#   TESTSET  cv: the fold's held-out cases (folds 0-4 only). isles18: all 94 ISLES 2018 scans.
#   MODE     debug: run.py -d (the first 4 slices). full: every slice.
#   JOBID    Condor's cluster.process: names this job's folder on /home.
#   EPOCH    the checkpoint, 200 by default: the last (ctp_only_eval_plan.md §4).
#
# MRDPM's maps for ctp_only_eval_plan.md: run.py -p test, sampling once per slice with the EMA
# weights (the config's test_ema). The checkpoint is copied from /radraid2 to the H100 box's
# /home, the data are the staged copies there (ctp_only_stage_home.sub, and for isles18
# ctp_only_stage_isles18.sub), and the results go back to
# /radraid2/colivares/mrdpm_ctp_only/runs/<experiment>/, without the Process_ trajectories
# (about 5 MB a slice), which stay on /home.
set -u

W="${_CONDOR_SCRATCH_DIR:-$PWD}"
: > "$W/job_summary.txt"            # the one file Condor sends back (named in the .sub)
export HOME="$W"
export MPLCONFIGDIR="$W/.matplotlib"
export PYTHONDONTWRITEBYTECODE=1    # this repo tracks __pycache__; don't add to it
export PYTHONUNBUFFERED=1

RUN="${1:-}"
TESTSET="${2:-cv}"
MODE="${3:-debug}"
JOBID="${4:-job$$}"
EPOCH="${5:-200}"
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
[ -n "$RUN" ] && [ "$RUN" != none ] || fail "give the training run: -append 'run = /radraid2/colivares/mrdpm_ctp_only/runs/<folder>'"
RUN="${RUN%/}"
name=$(basename "$RUN")
[[ "$name" =~ ^(train|debug)_ctp_only_f13_(tmax|rcbf)_(fold[0-4]|all)_[0-9]{6}_[0-9]{6}$ ]] \
    || fail "$name is not a ctp-only training run folder"
TARGET="${BASH_REMATCH[2]}"
FOLD="${BASH_REMATCH[3]#fold}"

case "$TARGET" in
    tmax) GAN_RUN="$STAGE/gan_run"; STAGED="$STAGE/STAGED_OK" ;;
    rcbf) GAN_RUN="$STAGE/gan_run_rcbf"; STAGED="$STAGE/STAGED_OK_rcbf" ;;
esac
[ -f "$STAGED" ] || fail "no $STAGED: the $TARGET data aren't staged"
if [ "$FOLD" = all ]; then
    GAN_RUN="$STAGE/gan_run_all_$TARGET"; RUN_SUB=all
    CONFIG="$CODE/config/ctp_only_f13_${TARGET}_all.json"
    [ -f "$STAGE/STAGED_OK_all" ] || fail "no $STAGE/STAGED_OK_all: run ctp_only_stage_alldata.sub first"
else
    RUN_SUB="fold_$FOLD"
    CONFIG="$CODE/config/ctp_only_f13_${TARGET}_fold0.json"
fi
case "$TESTSET" in
    cv) [ "$FOLD" != all ] || fail "an all-data model trained on every CV case: test it on isles18 only" ;;
    isles18) [ -f "$STAGE/STAGED_OK_isles18" ] || fail "no $STAGE/STAGED_OK_isles18: run ctp_only_stage_isles18.sub first" ;;
    *) fail "testset must be cv or isles18, not '$TESTSET'" ;;
esac

say "host: $(hostname)   date: $(date -Is)   run $name   target $TARGET   fold $FOLD   test set $TESTSET   mode $MODE   epoch $EPOCH   job $JOBID"
nvidia-smi -L || echo "(nvidia-smi -L failed)"
n_gpus=$(env -u CUDA_VISIBLE_DEVICES python -c "import torch; print(torch.cuda.device_count())")
[ "$n_gpus" = 1 ] || fail "the container sees $n_gpus GPUs; run.py would take GPU 0, which may not be this job's"
[ -d "$CODE" ] || fail "no checkout at $CODE (transfer_input_files)"
say "MRDPM code: $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null || echo '<git unavailable>')"
for s in _Network.pth _Network_ema.pth .state; do
    [ -f "$RUN/checkpoint/$EPOCH$s" ] || fail "no $RUN/checkpoint/$EPOCH$s"
done
mkdir -p "$EXP_BASE" "$H/resume/$JOBID" || fail "cannot create the job's folders on /home"
for s in _Network.pth _Network_ema.pth .state; do
    cp "$RUN/checkpoint/$EPOCH$s" "$H/resume/$JOBID/" || fail "copying $EPOCH$s"
done
say "checkpoint $EPOCH copied from $RUN"
# Which training run made these maps: the results folder's name says only target and fold.
ema_sha=$(sha256sum < "$H/resume/$JOBID/${EPOCH}_Network_ema.pth" | cut -c1-64)
printf '{"run": "%s", "epoch": %s, "ema_sha256": "%s", "testset": "%s", "mode": "%s", "job": "%s", "host": "%s", "date": "%s", "mrdpm_commit": "%s"}\n' \
    "$RUN" "$EPOCH" "$ema_sha" "$TESTSET" "$MODE" "$JOBID" "$(hostname)" "$(date -Is)" \
    "$(git -C "$CODE" rev-parse --short HEAD 2>/dev/null)" > "$W/source_run.json"

echo "==================================================================="
echo "1. the job's config"
python - "$CONFIG" "$W/job_config.json" "$STAGE" "$EXP_BASE" "$FOLD" "$H/resume/$JOBID/$EPOCH" \
    "$TARGET" "$GAN_RUN/$RUN_SUB" "$TESTSET" <<'PYEOF' || fail "writing the job config"
import json, sys
from pathlib import Path
src, dst, stage, base, fold, resume, target, run, testset = sys.argv[1:]
cfg = json.loads("".join(line.split("//")[0] + "\n" for line in open(src)))    # as core/praser.py reads it
gan = f"{stage}/gan_bpm"
if testset == "cv":
    test = dict(mctp_root=f"{stage}/mctp_dataset_f13_isles24_ucla_v2", split_json=f"{run}/split.json")
else:
    test = dict(mctp_root=f"{stage}/isles2018/mctp_dataset_msu_f13_isles2018_all94",
                split_json=f"{stage}/isles2018/split.json")
args = cfg["datasets"]["test"]["which_dataset"]["args"]
args.update(gan_bpm_root=gan, part="test", **test)
if not cfg["model"]["which_model"]["args"].get("test_ema"):
    sys.exit("the config doesn't set test_ema: the evaluation tests the EMA weights")
ip = cfg["model"]["which_networks"][0]["args"]["initial_predictor"]
checkpoint = f"{run}/checkpoints/{Path(ip['checkpoint']).name}"     # the predictor is rebuilt, then overwritten by the checkpoint
if not Path(checkpoint).is_file():
    sys.exit(f"{checkpoint} isn't staged")
ip.update(gan_bpm_root=gan, checkpoint=checkpoint)
if fold != "all":
    ip["expect_fold"] = int(fold)
cfg["name"] = f"ctp_only_f13_{target}_" + ("all" if fold == "all" else f"fold{fold}") + f"_{testset}"
cfg["path"].update(base_dir=base, resume_state=resume)
json.dump(cfg, open(dst, "w"), indent=1)
print(f"name {cfg['name']}, test {args['mctp_root']} / {args['split_json']}, resume_state {resume}")
PYEOF

# Everything but the Process_ trajectories goes to $OUT; the checkpoint copy and run.py's own
# copy of the code stay behind.
sync_out() {
    local exp dst
    exp=$(ls -d "$EXP_BASE"/*/ 2>/dev/null | head -1)
    [ -n "$exp" ] || return 0
    exp="${exp%/}"
    dst="$OUT/$(basename "$exp")"
    mkdir -p "$dst"
    (cd "$exp" && tar --exclude='Process_*' --exclude='./code' -cf - .) | (cd "$dst" && tar -xf -)
    cp -u "$W/test.out" "$W/job_config.json" "$W/source_run.json" "$dst/" 2>/dev/null
    [ -f "$CODE/ddim_test.txt" ] && cp "$CODE/ddim_test.txt" "$dst/"
    RUN_OUT="$dst"
}

echo "==================================================================="
echo "2. test (run.py's own output: test.out, copied next to the results)"
cd "$CODE" || fail "cd $CODE"
python run.py -p test $DEBUG -c "$W/job_config.json" > "$W/test.out" 2>&1 &
PID=$!
RUN_OUT=
while kill -0 "$PID" 2>/dev/null; do
    for _ in $(seq 60); do
        kill -0 "$PID" 2>/dev/null || break
        sleep 10
    done
    n=$(ls "$EXP_BASE"/*/results/test/*/Out_*.npy 2>/dev/null | wc -l)
    echo "$(date '+%m-%d %H:%M')  $n slices sampled"
done
wait "$PID"
status=$?
sync_out

echo "==================================================================="
n=$(ls "$EXP_BASE"/*/results/test/*/Out_*.npy 2>/dev/null | wc -l)
say "test exited $status; $n slices sampled; results in ${RUN_OUT:-<none>} (Process_ files stay in $EXP_BASE)"
tr '\r' '\n' < "$W/test.out" | grep -v "it/s]\|s/it]\|^ *$" | tail -n 15 | tee -a "$W/job_summary.txt"
exit "$status"
