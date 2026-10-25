"""Write the DAG that samples MRDPM's ISLES 2018 maps (ctp_only_eval_plan.md §6.1): one
condor/ctp_only_test_h100.sub node per trained fold model, testset = isles18, mode = full.

Runs on 14958 with any python3 (standard library only). A (target, fold) is trained when
exactly one train_ctp_only_f13_<target>_fold<k>_* folder under --runs has
checkpoint/<epoch>_Network_ema.pth (a killed run and its resume are two folders; only the
resume reaches 200). It is left out when a complete ISLES 2018 test of that same training
folder exists (its source_run.json names the folder, and all 502 slices are there). Prints
every (target, fold) and what it found. rCBF first: its DWI Dice is the headline.

    python3 scripts/ctp_only_isles18_dag.py [--maxjobs 1] [--target rcbf] [--fold 0 --fold 3]
    cd condor && condor_submit_dag ctp_only_isles18_tests_<date_time>.dag

The account may hold 3 GPUs at once, and going over gets jobs booted: --maxjobs is how
many of this DAG's tests run at once, so count what else is running first (condor_q -run).
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = Path("/radraid2/colivares/mrdpm_ctp_only/runs")
SLICES = 502                     # the ISLES 2018 build: 94 scans, 502 slices
TARGETS = ("rcbf", "tmax")
FOLDS = (0, 1, 2, 3, 4)


def latest_checkpoint(run):
    names = [re.match(r"^(\d+)_Network_ema\.pth$", p.name) for p in (run / "checkpoint").glob("*")]
    return max((int(m.group(1)) for m in names if m), default=None)


def n_slices(test):
    return sum(1 for p in (test / "results" / "test").glob("*/Out_*.npy"))


def tests_of(runs, target, fold):
    """{training folder: [complete ISLES 2018 test folders made from it]}."""
    out = {}
    for test in runs.glob(f"test_ctp_only_f13_{target}_fold{fold}_isles18_*"):
        src = test / "source_run.json"
        if not src.is_file():
            continue
        made_from = json.loads(src.read_text()).get("run", "")
        if n_slices(test) == SLICES:
            out.setdefault(Path(made_from).name, []).append(test.name)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", type=Path, default=RUNS)
    p.add_argument("--target", action="append", choices=TARGETS)
    p.add_argument("--fold", action="append", type=int, choices=FOLDS)
    p.add_argument("--epoch", type=int, default=200)
    p.add_argument("--maxjobs", type=int, default=1, help="this DAG's tests at once (default 1)")
    p.add_argument("--out", type=Path, help="default: condor/ctp_only_isles18_tests_<date_time>.dag")
    p.add_argument("--dry-run", action="store_true", help="print the table, write no DAG")
    args = p.parse_args(argv)

    nodes = []
    print(f"{'target':<6} {'fold':<5} state")
    for target in args.target or TARGETS:
        for k in args.fold or FOLDS:
            runs = sorted(r for r in args.runs.glob(f"train_ctp_only_f13_{target}_fold{k}_*") if r.is_dir())
            done = [r for r in runs if (r / "checkpoint" / f"{args.epoch}_Network_ema.pth").is_file()]
            if len(done) > 1:
                state = f"AMBIGUOUS: {len(done)} runs reached {args.epoch} ({', '.join(r.name for r in done)}); left out"
            elif not done:
                last = [f"{r.name} at {latest_checkpoint(r)}" for r in runs]
                state = "not trained yet" + (f" ({'; '.join(last)})" if last else " (no run folder)")
            else:
                tested = tests_of(args.runs, target, k).get(done[0].name)
                if tested:
                    state = f"tested already: {', '.join(sorted(tested))}"
                else:
                    state = f"ready to test (no complete ISLES test on /radraid2 yet): {done[0].name}"
                    nodes.append((f"{target.upper()}_F{k}", done[0]))
            print(f"{target:<6} {k:<5} {state}")
    if not nodes or args.dry_run:
        print("nothing to test: no DAG written" if not nodes else f"\n{len(nodes)} to test (dry run: no DAG written)")
        return 0

    out = args.out or REPO / "condor" / f"ctp_only_isles18_tests_{datetime.now():%y%m%d_%H%M%S}.dag"
    lines = [
        "# Written by scripts/ctp_only_isles18_dag.py: MRDPM's ISLES 2018 maps, one test per",
        "# trained fold model (ctp_only_eval_plan.md §6.1). About 4 h each on the H100.",
        "# Results: /radraid2/colivares/mrdpm_ctp_only/runs/test_ctp_only_f13_<target>_fold<k>_isles18_<time>/",
        "# A retried node samples from scratch (a test has nothing to resume), into a new folder.",
        "",
    ]
    # DAGMan doesn't submit ready nodes in file order (the rCBF training DAG ran 0, 4, 1, 3),
    # so PRIORITY keeps this order: rCBF first, then by fold.
    for i, (name, run) in enumerate(nodes):
        lines += [f"JOB      {name}  ctp_only_test_h100.sub",
                  f'VARS     {name}  run="{run}" testset="isles18" mode="full" epoch="{args.epoch}"',
                  f"CATEGORY {name}  h100",
                  f"PRIORITY {name}  {len(nodes) - i}",
                  f"RETRY    {name}  1",
                  ""]
    lines.append(f"MAXJOBS  h100  {args.maxjobs}")
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}: {len(nodes)} tests, {args.maxjobs} at a time "
          f"(about {4 * -(-len(nodes) // args.maxjobs)} h)")
    print(f"  cd {out.parent} && condor_submit_dag {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
