"""ctp_only_plan.md §7.1 / §13 step 1: does MRDPM's input masking hurt gan_bpm's generator?

MRDPM multiplies its [0, 1] condition by the brain mask, so the generator (which reads
2*cond - 1) would see -1 everywhere outside gan_bpm's brain envelope. gan_bpm trained it on
the unmasked input. This runs the fold-0 checkpoint over fold 0's held-out slices both ways
and compares the in-brain L1 over `support`, in gan_bpm's [-1, 1] units and in seconds.

The `native` arm goes through gan_bpm's own `build_dataloaders`, so its pooled L1 must
reproduce the checkpoint's stored `val_L_pix_brain`. That is the check that this script
measures what gan_bpm measured; the `masked` arm differs from it only in the input.

Run from the MRDPM repo root, inside the job image (CPU is fine):

    python scripts/masked_input_check.py --out /radraid2/colivares/mrdpm_ctp_only/masked_input_check
"""
import argparse
import csv
import json
import sys
import time
from dataclasses import fields
from pathlib import Path

import torch
from torch.utils.data import DataLoader

GAN_BPM = Path("/raid/colivares/github_repos/gan_bpm")
RUN = Path("/radraid2/colivares/matrix/gan_bpm_tmax_f13uclav2_t12_adv0_k5")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gan-bpm", type=Path, default=GAN_BPM)
    p.add_argument("--run", type=Path, default=RUN, help="gan_bpm k5 run dir")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="where per_slice.csv and summary.json go")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--limit-cases", type=int, default=0, help="first N val cases only (timing dry run)")
    args = p.parse_args()

    # gan_bpm's root goes first: its `models` package must win, and MRDPM's root is not on
    # sys.path when this file is run as `python scripts/...`.
    sys.path.insert(0, str(args.gan_bpm))
    from training import TrainingConfig, build_dataloaders                # noqa: E402
    from models.generator import Generator                                # noqa: E402
    from mctp.label_scale import label_bounds                             # noqa: E402
    from mctp.transforms import split_mask                                # noqa: E402

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    fold_dir = args.run / f"fold_{args.fold}"
    ckpt_path = fold_dir / "checkpoints/best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt["config"])

    want = {"target": "tmax", "tmax_ceiling": 12.0, "n_phases": 3, "label_transform": "linear",
            "fold": args.fold}
    bad = {k: cfg.get(k) for k, v in want.items() if cfg.get(k) != v}
    if bad:
        sys.exit(f"refusing {ckpt_path}: config {bad} differs from {want}")

    # The stored split_file is relative to the gan_bpm checkout.
    if cfg.get("split_file") and not Path(cfg["split_file"]).is_absolute():
        cfg["split_file"] = str(args.gan_bpm / cfg["split_file"])
    cfg["num_workers"] = args.num_workers
    known = {f.name for f in fields(TrainingConfig)}
    config = TrainingConfig(**{k: v for k, v in cfg.items() if k in known}).validate()

    _, val_loader = build_dataloaders(config)
    ds = val_loader.dataset
    split = json.loads((fold_dir / "split.json").read_text())
    loader_cases = set(ds.rows["case_id"].astype(str))
    if loader_cases != set(split["val_case_ids"]):
        sys.exit(f"val cases from build_dataloaders ({len(loader_cases)}) != "
                 f"{fold_dir}/split.json val_case_ids ({len(split['val_case_ids'])})")
    if args.limit_cases:
        keep = sorted(loader_cases)[:args.limit_cases]
        ds.rows = ds.rows[ds.rows["case_id"].astype(str).isin(keep)].reset_index(drop=True)

    G = Generator(n_phases=config.n_phases)
    G.load_state_dict(ckpt["generator"])
    G.to(device).eval()

    lo, hi = label_bounds(config.target, config.tmax_ceiling)
    seconds_per_unit = (hi - lo) / 2.0                     # linear: |Δ seconds| = 6 |Δ y| at 12 s

    print(f"checkpoint  {ckpt_path}  epoch {ckpt['epoch']}")
    print(f"val fold    {len(loader_cases)} cases (matches split.json), {len(ds)} slices "
          f"{'(limited to ' + str(args.limit_cases) + ' cases)' if args.limit_cases else ''}")
    print(f"device      {device}, {args.threads} threads, batch {args.batch_size}")

    # Default collate, not gan_bpm's drop-empty one: an empty-support slice adds 0 to both
    # the numerator and the denominator, so the pooled ratio is the same either way.
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    per_slice = []
    t0, i0 = time.time(), 0
    with torch.no_grad():
        for bi, (a, b, m) in enumerate(loader):
            a, b, m = a.to(device), b.to(device), m.to(device)
            support, _, envelope = split_mask(m)               # (N, 1, H, W) each
            # What MRDPM's dataset would hand the generator: 2 * (cond01 * envelope) - 1.
            inside = (envelope > 0.5).unsqueeze(2)             # (N, 1, 1, H, W) over the phases
            a_masked = torch.where(inside, a, torch.full_like(a, -1.0))
            sums = {}
            for arm, inp in (("native", a), ("masked", a_masked)):
                err = (G(inp) - b).abs() * support
                sums[arm] = err.sum(dim=(1, 2, 3)).double().cpu()
            sup_n = support.sum(dim=(1, 2, 3)).double().cpu()
            env_n = envelope.sum(dim=(1, 2, 3)).double().cpu()
            for j in range(a.shape[0]):
                row = ds.rows.iloc[i0 + j]
                per_slice.append({
                    "case_id": str(row["case_id"]), "dataset": str(row.get("dataset", "")),
                    "slice_idx": int(row.get("slice_idx", -1)),
                    "support_n": float(sup_n[j]), "envelope_n": float(env_n[j]),
                    "native_err_sum": float(sums["native"][j]),
                    "masked_err_sum": float(sums["masked"][j]),
                })
            i0 += a.shape[0]
            if bi % 10 == 0:
                rate = i0 / (time.time() - t0)
                print(f"  {i0:>5}/{len(ds)} slices  {rate:.2f} slices/s  "
                      f"eta {(len(ds) - i0) / rate / 60:.1f} min", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "per_slice.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_slice[0]))
        w.writeheader()
        w.writerows(per_slice)

    def pooled(rows):
        den = sum(r["support_n"] for r in rows)
        return {arm: sum(r[f"{arm}_err_sum"] for r in rows) / den for arm in ("native", "masked")}

    def line(name, rows):
        l1 = pooled(rows)
        d = l1["masked"] - l1["native"]
        print(f"  {name:<16} {len({r['case_id'] for r in rows}):>4} cases {len(rows):>5} slices | "
              f"native {l1['native']:.5f} ({l1['native'] * seconds_per_unit:.3f} s) | "
              f"masked {l1['masked']:.5f} ({l1['masked'] * seconds_per_unit:.3f} s) | "
              f"Δ {d * seconds_per_unit:+.3f} s ({100 * d / l1['native']:+.1f} %)")
        return l1

    print("\nin-brain L1 over support, pooled (gan_bpm [-1, 1] units, then seconds)")
    overall = line("all", per_slice)
    by_dataset = {}
    for name in sorted({r["dataset"] for r in per_slice}):
        by_dataset[name] = line(name, [r for r in per_slice if r["dataset"] == name])

    cases = {}
    for r in per_slice:
        cases.setdefault(r["case_id"], []).append(r)
    case_delta = [pooled(rs)["masked"] - pooled(rs)["native"]
                  for rs in cases.values() if sum(r["support_n"] for r in rs) > 0]
    case_delta.sort()
    worse = sum(d > 0 for d in case_delta)
    print(f"\nper case (masked − native, seconds): {worse}/{len(case_delta)} cases worse; "
          f"min {case_delta[0] * seconds_per_unit:+.3f}  "
          f"median {case_delta[len(case_delta) // 2] * seconds_per_unit:+.3f}  "
          f"max {case_delta[-1] * seconds_per_unit:+.3f}")

    stored = ckpt.get("metrics", {}).get("val_L_pix_brain", ckpt.get("select_value"))
    control = None
    if not args.limit_cases:
        control = abs(overall["native"] - stored)
        verdict = "OK" if control < 1e-4 * stored else "MISMATCH: native arm is not gan_bpm's metric"
        print(f"\ncontrol: native {overall['native']:.6f} vs checkpoint val_L_pix_brain "
              f"{stored:.6f}, |diff| {control:.2e}  {verdict}")

    summary = {"checkpoint": str(ckpt_path), "epoch": ckpt["epoch"], "seconds_per_unit": seconds_per_unit,
               "n_slices": len(per_slice), "n_cases": len(cases), "limit_cases": args.limit_cases,
               "overall": overall, "by_dataset": by_dataset,
               "cases_worse": worse, "cases_total": len(case_delta),
               "stored_val_L_pix_brain": stored, "control_abs_diff": control}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {args.out}/per_slice.csv and summary.json")


if __name__ == "__main__":
    main()
