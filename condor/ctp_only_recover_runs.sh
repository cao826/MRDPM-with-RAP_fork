#!/bin/bash
# Invoked by ctp_only_recover_runs.sub, on the H100 box, as: ctp_only_recover_runs.sh [EPOCH]
#   EPOCH  a run is finished when it has this checkpoint: 200 by default, the last.
#
# Lists every training run folder on the box's /home (experiments/<job>/train_*) and copies the
# finished ones that aren't on /radraid2 yet (ctp_only_decisions.md #26). Each is copied under a
# hidden name, checked (the same files at the same sizes) and only then renamed, so nothing on
# 14958 sees half a copy. A folder already on /radraid2 is never written to; unfinished folders
# and checkpoints left behind on the box are listed, not copied. Nothing on the box changes.
set -u

export HOME="${_CONDOR_SCRATCH_DIR:-/tmp}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

EPOCH="${1:-200}"
EXP=/home/colivares/mrdpm_ctp_only/experiments
OUT=/radraid2/colivares/mrdpm_ctp_only/runs

echo "host: $(hostname)   date: $(date -Is)   finished = has checkpoint $EPOCH"
df -h /home "$OUT" | tail -n 2
[ -d "$EXP" ] || { echo "FATAL: no $EXP"; exit 1; }
[ -d "$OUT" ] || { echo "FATAL: no $OUT"; exit 1; }
echo "==================================================================="

python - "$EXP" "$OUT" "$EPOCH" <<'EOF'
import re
import shutil
import sys
from pathlib import Path

exp, out, epoch = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
NAME = re.compile(r"^train_ctp_only_f13_(tmax|rcbf)_(fold\d|all)_\d{6}_\d{6}$")


def epochs(ck):
    """The checkpoint epochs in `ck` with all three files."""
    have = set(p.name for p in ck.iterdir()) if ck.is_dir() else set()
    return sorted(int(n[:-6]) for n in have if n.endswith(".state") and n[:-6].isdigit()
                  and {f"{n[:-6]}_Network.pth", f"{n[:-6]}_Network_ema.pth"} <= have)


def sizes(root):
    return {str(p.relative_to(root)): p.stat().st_size for p in root.rglob("*") if p.is_file()}


copied, failed = [], []
print(f"{'job':<11} {'run folder on the box':<46} {'newest':>6}  what")
for run in sorted(exp.glob("*/train_*"), key=lambda p: p.name):
    if not run.is_dir():
        continue
    here = epochs(run / "checkpoint")
    dst = out / run.name
    line = f"{run.parent.name:<11} {run.name:<46} {here[-1] if here else '-':>6}  "
    if dst.exists():
        left = [e for e in here if e not in epochs(dst / "checkpoint")]
        print(line + "on /radraid2" + (f"; checkpoint {left} only on the box" if left else ""), flush=True)
        continue
    if epoch not in here:
        print(line + "NOT on /radraid2, unfinished: left on the box", flush=True)
        continue
    print(line + "NOT on /radraid2, finished: copying ...", flush=True)
    tmp = out / f".recovering_{run.name}"
    if tmp.exists():
        shutil.rmtree(tmp)                     # this job's own half copy from an earlier attempt
    shutil.copytree(run, tmp, symlinks=True)
    want, got = sizes(run), sizes(tmp)
    if want != got:
        diff = sorted(set(want.items()) ^ set(got.items()))[:5]
        print(f"{'':<11} COPY FAILED, left as {tmp.name}: {len(want)} files on the box, "
              f"{len(got)} copied; first differences {diff}", flush=True)
        failed.append(run.name)
        continue
    tmp.rename(dst)
    print(f"{'':<11} copied: {len(want)} files, {sum(want.values()) / 1e9:.1f} GB, "
          f"checkpoints {here[0]}-{here[-1]}", flush=True)
    copied.append(run.name)

print("\n===================================================================")
print(f"finished runs (checkpoint {epoch}) on /radraid2 now, by target and fold:")
done = {}
for d in out.iterdir():
    m = NAME.match(d.name)
    if m and (d / "checkpoint" / f"{epoch}_Network_ema.pth").is_file():
        done.setdefault(m.groups(), []).append(d.name)
for target in ("rcbf", "tmax"):
    for fold in [f"fold{k}" for k in range(5)] + ["all"]:
        names = sorted(done.get((target, fold), []))
        flag = "   <- MORE THAN ONE: the ISLES DAG script leaves it out" if len(names) > 1 else ""
        print(f"  {target:<5} {fold:<6} {', '.join(names) or '-'}{flag}")
print(f"\ncopied {len(copied)}: {', '.join(copied) or 'none'}")
if failed:
    print(f"FAILED {len(failed)}: {', '.join(failed)}")
sys.exit(1 if failed else 0)
EOF
status=$?
echo "exit $status"
exit "$status"
