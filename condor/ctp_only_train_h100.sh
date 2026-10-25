#!/bin/bash
# Invoked by ctp_only_train_h100.sub as: ctp_only_train_h100.sh FOLD MODE JOBID [RESUME]
#   FOLD   0-4, or all: the all-data model for ISLES 2018 (ctp_only_eval_plan.md §6), from
#          config/ctp_only_f13_<target>_all.json and gan_bpm's all-data generator
#   MODE   debug: run.py -d (1 epoch on 4 slices, a monitor pass on 2). full: the whole recipe.
#   JOBID  Condor's cluster.process: names this job's run folder on /home.
#   RESUME a checkpoint prefix on /radraid2 (.../checkpoint/<epoch>) to continue from.
# and MRDPM_TARGET in the environment (the .sub's `target`): tmax, the default, or rcbf. It picks
# the config (config/ctp_only_f13_<target>_fold0.json) and the staged generators. A job submitted
# without it is a Tmax job, as before. MRDPM_VARIANT (the .sub's `variant`, none by default) picks
# an arm's config instead, config/ctp_only_f13_<target>_<variant>_fold0.json (bran_ext_plan.md §5);
# a BRAN arm's predictor is copied from /radraid2/colivares/mrdpm_ctp_only/bran/<name>/fold_<k>/.
#
# The data, gan_bpm's code and the fold's generator are already on the H100 box's own disk,
# /home/colivares/mrdpm_ctp_only/stage (condor/ctp_only_stage_home.sub, once per target;
# ctp_only_decisions.md #17). The job reads them there and writes its run folder there too.
# Every 10 minutes, and at the end, it moves finished checkpoints and copies everything else to
# /radraid2/colivares/mrdpm_ctp_only/runs/<experiment>/, the only place 14958 can see.
#
# When Condor evicts the job and runs it again, the job keeps its JOBID, so it finds its earlier
# attempts' run folders in $EXP_BASE. It resumes from their newest complete checkpoint instead of
# epoch 0, and every run folder there is copied to /radraid2, not only the first
# (ctp_only_decisions.md #26).
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
TARGET="${MRDPM_TARGET:-tmax}"
VARIANT="${MRDPM_VARIANT:-none}"
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
# Each target's generators and done-marker, as condor/ctp_only_stage_home.sh writes them.
case "$TARGET" in
    tmax) GAN_RUN="$STAGE/gan_run"; STAGED="$STAGE/STAGED_OK" ;;
    rcbf) GAN_RUN="$STAGE/gan_run_rcbf"; STAGED="$STAGE/STAGED_OK_rcbf" ;;
    *) fail "target must be tmax or rcbf, not '$TARGET'" ;;
esac
# fold k reads gan_run*/fold_k with the fold-0 config; all reads gan_run_all_<target>/all
# (condor/ctp_only_stage_alldata.sh) with the all-data config.
if [ "$FOLD" = all ]; then
    [ -f "$STAGED" ] || fail "no $STAGED: the $TARGET data aren't staged"
    GAN_RUN="$STAGE/gan_run_all_$TARGET"; STAGED="$STAGE/STAGED_OK_all"; RUN_SUB=all
    CONFIG="$CODE/config/ctp_only_f13_${TARGET}_all.json"
else
    case "$FOLD" in [0-4]) ;; *) fail "fold must be 0-4 or all, not '$FOLD'" ;; esac
    RUN_SUB="fold_$FOLD"
    CONFIG="$CODE/config/ctp_only_f13_${TARGET}_fold0.json"
fi
if [ "$VARIANT" != none ]; then
    [ "$FOLD" != all ] || fail "the arms (variant) are fold models; there is no all-data arm"
    CONFIG="$CODE/config/ctp_only_f13_${TARGET}_${VARIANT}_fold0.json"
fi

say "host: $(hostname)   date: $(date -Is)   target $TARGET   variant $VARIANT   fold $FOLD   mode $MODE   job $JOBID   resume ${RESUME:-<none>}"
nvidia-smi -L || echo "(nvidia-smi -L failed)"
# run.py sets CUDA_VISIBLE_DEVICES to the config's gpu_ids ("0"), which is only this job's GPU
# when the container sees exactly one.
n_gpus=$(env -u CUDA_VISIBLE_DEVICES python -c "import torch; print(torch.cuda.device_count())")
[ "$n_gpus" = 1 ] || fail "the container sees $n_gpus GPUs; run.py would take GPU 0, which may not be this job's"
[ -d "$CODE" ] || fail "no checkout at $CODE (transfer_input_files)"
say "MRDPM code: $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null || echo '<git unavailable>')"
[ -f "$STAGED" ] || fail "no $STAGED: run condor/ctp_only_stage_home.sub (target $TARGET) first"
say "stage: $(cat "$STAGED")"
[ -f "$GAN_RUN/$RUN_SUB/split.json" ] || fail "$RUN_SUB of $TARGET isn't staged"
df -h /home | tail -1
mkdir -p "$EXP_BASE" || fail "cannot create $EXP_BASE"

# this attempt's run folder, once run.py has made it: names end in the start time, so it is the last
current() { ls -d "$EXP_BASE"/*/ 2>/dev/null | tail -1; }

# A run folder already here means Condor ran this job before. Resume from the newest checkpoint
# of those attempts whose three files all load, on /home or already moved to $OUT, if it is newer
# than RESUME. Without this a restarted job trained from epoch 0, unseen (decisions #26).
if [ -n "$(current)" ]; then
    say "restarted by Condor: earlier run folders in $EXP_BASE: $(ls "$EXP_BASE" | tr '\n' ' ')"
    own=$(python - "$EXP_BASE" "$OUT" <<'PYEOF'
import sys
from pathlib import Path
import torch
base, out = Path(sys.argv[1]), Path(sys.argv[2])
found = []
for run in (r for r in base.iterdir() if r.is_dir()):
    for ck in (run / "checkpoint", out / run.name / "checkpoint"):
        for state in ck.glob("*.state"):
            e = state.name[:-len(".state")]
            if e.isdigit() and all((ck / f"{e}{s}").is_file() for s in ("_Network.pth", "_Network_ema.pth")):
                found.append((int(e), str(ck / e)))
for e, prefix in sorted(found, reverse=True):
    try:                    # the eviction may have cut a save short
        for s in (".state", "_Network.pth", "_Network_ema.pth"):
            torch.load(prefix + s, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"checkpoint {prefix} doesn't load ({exc}); trying an older one", file=sys.stderr)
        continue
    print(prefix)
    break
PYEOF
)
    resume_epoch=-1
    [ -n "$RESUME" ] && resume_epoch=$(basename "$RESUME")
    if [ -n "$own" ] && [ "$(basename "$own")" -gt "$resume_epoch" ]; then
        say "resuming from this job's own checkpoint $own"
        RESUME="$own"
    else
        say "no checkpoint of the earlier attempts is newer than resume ${RESUME:-<none>}: as submitted"
    fi
fi

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
mkdir -p "$H/bran_in/$JOBID" || fail "cannot create $H/bran_in/$JOBID"
python - "$CONFIG" "$W/job_config.json" "$STAGE" "$EXP_BASE" "$FOLD" "$RESUME_LOCAL" "$TARGET" "$GAN_RUN/$RUN_SUB" \
    "$VARIANT" "$MODE" "$H/bran_in/$JOBID" <<'PYEOF' || fail "writing the job config"
import json, shutil, sys
from pathlib import Path
src, dst, stage, base, fold, resume, target, run, variant, mode, bran_in = sys.argv[1:]
cfg = json.loads("".join(line.split("//")[0] + "\n" for line in open(src)))    # as core/praser.py reads it
data, gan = f"{stage}/mctp_dataset_f13_isles24_ucla_v2", f"{stage}/gan_bpm"
# The all-data config has no monitor, and its test set is ISLES 2018, which training never reads.
blocks = ([("train", "which_dataset")] if fold == "all" else
          [("train", "which_dataset"), ("train", "which_val_dataset"), ("test", "which_dataset")])
for phase, key in blocks:
    cfg["datasets"][phase][key]["args"].update(mctp_root=data, gan_bpm_root=gan, split_json=f"{run}/split.json")
ip = cfg["model"]["which_networks"][0]["args"]["initial_predictor"]
if ip.get("type") == "bran":
    # bran_ext_plan.md §5: the fold's pretrained BRAN on /radraid2 (a debug run takes the debug one
    # if there is one), copied to /home: jobs don't read /radraid2 while they run
    ck = Path(ip["checkpoint"])                  # .../bran/<name>/fold_0/bran_G.pth
    name, root = ck.parent.parent.name, ck.parent.parent.parent
    tries = ([root / f"debug_{name}"] if mode == "debug" else []) + [root / name]
    found = [t / f"fold_{fold}" / ck.name for t in tries if (t / f"fold_{fold}" / ck.name).is_file()]
    if not found:
        sys.exit(f"no pretrained BRAN at {tries[-1] / f'fold_{fold}' / ck.name}: run "
                 f"ctp_only_bran_pretrain.sub for it first")
    shutil.copy(found[0], f"{bran_in}/{ck.name}")
    print(f"BRAN {found[0]} -> {bran_in}/{ck.name}")
    ip.update(checkpoint=f"{bran_in}/{ck.name}", expect_fold=int(fold), allow_debug=(mode == "debug"))
else:
    checkpoint = f"{run}/checkpoints/{Path(ip['checkpoint']).name}"     # best.pt, or the all-data epoch_<M>.pt
    if not Path(checkpoint).is_file():
        sys.exit(f"{checkpoint} isn't staged")
    ip.update(gan_bpm_root=gan, checkpoint=checkpoint)
    if fold != "all":
        ip["expect_fold"] = int(fold)
cfg["name"] = (f"ctp_only_f13_{target}_" + (f"{variant}_" if variant != "none" else "")
               + ("all" if fold == "all" else f"fold{fold}"))
cfg["path"].update(base_dir=base, resume_state=resume or None)
json.dump(cfg, open(dst, "w"), indent=1)
print(f"name {cfg['name']}, base_dir {base}, resume_state {cfg['path']['resume_state']}")
PYEOF

# Every run folder in $EXP_BASE, earlier attempts' too. Finished checkpoints move to $OUT; the
# newest set stays until a newer one exists, since it may still be being written ("final" moves
# everything). Everything else is copied. RUN_OUT ends as this attempt's copy.
sync_out() {
    local exp dst ck e newest
    for exp in "$EXP_BASE"/*/; do
        [ -d "$exp" ] || continue
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
        RUN_OUT="$dst"
    done
    if [ -n "$RUN_OUT" ]; then
        cp -u "$W/train.out" "$W/job_config.json" "$RUN_OUT/" 2>/dev/null
    fi
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
    last=$(tail -n 1 "$(current)health/train.csv" 2>/dev/null)
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
