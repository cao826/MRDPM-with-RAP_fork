"""Copy only the part of an mCTP root that one job reads (ctp_only_plan.md §13 step 4).

That is the label index, the mask cache, and the input and label files of the cases on one
side of a fold's split: `--side train` for a training job (training and monitor cases),
`--side val` for a test job (the held-out fold), `--side all` for both (a persistent copy every
fold reads). The index is copied whole, so the dataset class still finds every case of the
split. A file already at DST with the source's size is skipped, so a rerun only finishes the
copy.

    python scripts/ctp_only_stage_data.py SRC DST --split fold_0/split.json --side train [--dry-run]
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("src", type=Path, help="the staged mCTP root")
    p.add_argument("dst", type=Path, help="where the copy goes, e.g. the job's workspace")
    p.add_argument("--split", type=Path, required=True, help="the fold's split.json")
    p.add_argument("--side", choices=["train", "val", "all"], required=True)
    p.add_argument("--label-type", default="tmax")
    p.add_argument("--dry-run", action="store_true", help="print what would be copied, copy nothing")
    args = p.parse_args()

    index = f"{args.label_type}_index.parquet"
    rows = pd.read_parquet(args.src / index)
    split = json.loads(args.split.read_text())
    sides = ("train", "val") if args.side == "all" else (args.side,)
    cases = set().union(*(split[f"{s}_case_ids"] for s in sides))
    missing = cases - set(rows["case_id"])
    if missing:
        sys.exit(f"{len(missing)} cases of the split are not in {index}, e.g. {sorted(missing)[:3]}")
    keep = rows[rows["case_id"].isin(cases)]
    files = sorted(set(keep["input_relpath"]) | set(keep["label_relpath"]))
    cache = args.src / "mask_cache"
    cache_files = [f for f in cache.rglob("*") if f.is_file()] if cache.is_dir() else []
    total = (sum((args.src / f).stat().st_size for f in files) + sum(f.stat().st_size for f in cache_files)
             + (args.src / index).stat().st_size)
    print(f"{args.side}: {len(cases)} cases, {len(keep)} slices, {len(files)} files, "
          f"{len(cache_files)} mask-cache entries: {total / 1e9:.1f} GB")
    if args.dry_run:
        return

    args.dst.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.dst).free
    if total > 0.95 * free:
        sys.exit(f"{total / 1e9:.1f} GB to copy but only {free / 1e9:.1f} GB free at {args.dst}")
    t0 = time.time()
    shutil.copyfile(args.src / index, args.dst / index)       # always: it names what the files are
    copied, skipped = (args.src / index).stat().st_size, 0
    for f in [args.src / f for f in files] + cache_files:
        out = args.dst / f.relative_to(args.src)
        if out.is_file() and out.stat().st_size == f.stat().st_size:
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, out)
        copied += f.stat().st_size
    seconds = time.time() - t0
    print(f"copied {copied / 1e9:.1f} GB in {seconds:.0f} s ({copied / 1e6 / max(seconds, 1e-3):.0f} MB/s); "
          f"{skipped} files already there")


if __name__ == "__main__":
    main()
