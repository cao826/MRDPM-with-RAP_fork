#!/bin/bash
# Invoked by ctp_only_smoke.sub. Runs from the box's checkout of this repo (git pull first):
# run.py backs up config/ models/ core/ data/ from its cwd into each experiment folder.
set -u

# The execute node's $HOME is not writable inside the container (gan_bpm condor/run_cv.sh).
export HOME="${_CONDOR_SCRATCH_DIR:-/tmp}"
export MPLCONFIGDIR="$HOME/.matplotlib"
export PYTHONDONTWRITEBYTECODE=1    # this repo tracks __pycache__; don't add to it
export PYTHONUNBUFFERED=1           # the probe's lines reach the streamed .out as they print

REPO="/raid/colivares/github_repos/MRDPM-with-RAP_fork"
CONFIG="config/ctp_only_f13_tmax_fold0.json"
BASE="/radraid2/colivares/mrdpm_ctp_only/experiments"      # the config's path.base_dir
OUT="/radraid2/colivares/mrdpm_ctp_only/smoke"
mkdir -p "$OUT"

echo "host: $(hostname)   date: $(date -Is)   CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || echo "(nvidia-smi -L failed)"
# run.py sets CUDA_VISIBLE_DEVICES to the config's gpu_ids ("0"), which is only this job's GPU
# when the container sees exactly one.
n_gpus=$(env -u CUDA_VISIBLE_DEVICES python -c "import torch; print(torch.cuda.device_count())")
if [ "$n_gpus" != 1 ]; then
    echo "FATAL: the container sees $n_gpus GPUs; run.py would take GPU 0, which may not be this job's"
    exit 1
fi
cd "$REPO" || { echo "FATAL: cannot cd to $REPO"; exit 1; }
echo "MRDPM checkout: $(git rev-parse --short HEAD 2>/dev/null || echo '<not a git checkout>')"

# Experiment folders end in _YYMMDD_HHMMSS, so the last in sorted order is the newest.
newest_exp() { ls -d "$BASE"/debug_ctp_only_f13_tmax_fold0_* 2>/dev/null | sort | tail -1; }

# run.py's output, minus tqdm's progress redraws but with each sampling bar's last line.
show() {
    tr '\r' '\n' < "$1" | grep -v "it/s]\|s/it]\|^ *$" | tail -n 40
    tr '\r' '\n' < "$1" | grep "sampling loop.*1000/1000" | tail -1
}

# The card's memory in use, every half second, while run.py runs.
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > "$OUT/gpu_memory_mib.txt" &
SMI=$!

echo "==================================================================="
echo "1. run.py -p train -d"
python run.py -p train -d -c "$CONFIG" > "$OUT/train.out" 2>&1
train_status=$?
show "$OUT/train.out"
TRAIN_EXP=$(newest_exp)
echo "train exited $train_status; experiment: $TRAIN_EXP"
ls -la "$TRAIN_EXP/checkpoint" "$TRAIN_EXP/results/val/1" 2>&1

echo "==================================================================="
echo "2. run.py -p test -d from checkpoint 1"
if [ "$train_status" != 0 ] || [ ! -e "$TRAIN_EXP/checkpoint/1_Network.pth" ]; then
    test_status="skipped (no checkpoint 1)"
    echo "$test_status"
else
    TEST_CONFIG="$OUT/test_config.json"
    python - "$CONFIG" "$TRAIN_EXP/checkpoint/1" "$TEST_CONFIG" <<'EOF'
import json, sys
src, resume, dst = sys.argv[1:]
cfg = json.loads("".join(line.split("//")[0] + "\n" for line in open(src)))    # as core/praser.py reads it
cfg["path"]["resume_state"] = resume
json.dump(cfg, open(dst, "w"), indent=1)
EOF
    had_ddim=$([ -e ddim_test.txt ] && echo 1 || echo 0)
    python run.py -p test -d -c "$TEST_CONFIG" > "$OUT/test.out" 2>&1
    test_status=$?
    show "$OUT/test.out"
    TEST_EXP=$(newest_exp)
    # test() appends to ./ddim_test.txt in the cwd; keep it with its run, not in the checkout.
    [ "$had_ddim" = 0 ] && [ -e ddim_test.txt ] && mv ddim_test.txt "$TEST_EXP/"
    echo "test exited $test_status; experiment: $TEST_EXP"
    ls -la "$TEST_EXP/results/test/0" 2>&1
fi

kill "$SMI" 2>/dev/null
echo "card memory in use during 1-2: max $(sort -n "$OUT/gpu_memory_mib.txt" | tail -1) MiB"

echo "==================================================================="
echo "3. memory and seconds per step by batch size"
python scripts/ctp_only_gpu_probe.py --out "$OUT"
probe_status=$?

echo "==================================================================="
echo "train exited $train_status, test exited $test_status, probe exited $probe_status"
[ "$train_status" = 0 ] && [ "$test_status" = 0 ] && [ "$probe_status" = 0 ]
