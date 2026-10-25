"""Pretrain BRAN, the paper's initial predictor, alone (paper §2.1, Eq. 1; bran_ext_plan.md §2).

    python scripts/pretrain_bran.py -c CONFIG --out DIR [--debug] [--device cuda] [--workers 8]

CONFIG is the MRDPM config the predictor is for (config/ctp_only_f13_<target>_bran_*_fold0.json, or
the job config condor/ctp_only_bran_pretrain.sh writes from it). Everything comes from it, so
the predictor and the MRDPM run that fine-tunes it read the same data:

- training slices: `datasets.train.which_dataset` (the fold's training cases minus the monitor
  patients); the monitor: `which_val_dataset`; the held-out fold: `datasets.test.which_dataset`;
- the network and its loss: `initial_predictor` (`in_channels`, `mask_cond`, `ext`, `expect_fold`)
  and its recipe, `initial_predictor.pretrain` (models/bran_predictor.py:PRETRAIN_DEFAULTS).

Loss: L1 over `loss_mask` (gan_bpm's support) plus, with `ext`, its weight × the extreme loss
(models/loss.py), both means over support. The input is the condition × mask when `mask_cond`,
and the map is cut to the mask, as Network.initial does. The LR falls linearly from `lr` to 0
over the epochs. The model is the last epoch: nothing is chosen on any validation set.

Writes to --out:
  bran_G.pth         {"generator": state_dict, "meta": {...}}, what Network loads (strict)
  pretrain_log.csv   per epoch: the train and monitor losses, the monitor's MAE in label units,
                     the extreme loss split into its low and high halves, the core's bias and Dice
                     on the monitor slices, LR, seconds, peak GPU memory
  last_state.pt      the resume point: a job Condor runs again continues from it
  heldout/           Init_/GT_/Mask_ .npy for every held-out slice, named as MRDPM's test names
                     them, for scripts/ctp_only_core_check.py --bran
  heldout.json       what heldout/ holds
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))       # MRDPM's `models` and `data` must win over gan_bpm's

from models.bran_predictor import build_bran, expected_meta, normalize_ext, pretrain_recipe   # noqa: E402
from models.loss import ext_weight                                                          # noqa: E402
from models.networks_unet256 import init_weights                                            # noqa: E402

DEBUG_SLICES = {"train": 8, "monitor": 2, "test": 4}
# The core on [0, 1] labels: gan_bpm's level-3 regions, rCBF < 0.3 (0.3 / 2) and Tmax > 6 s (6 / 12)
CORE = {"rcbf": ("below", 0.15), "tmax": ("above", 0.5)}


def load_config(path):
    """As core/praser.py reads it: everything from `//` to the end of a line goes."""
    return json.loads("".join(line.split("//")[0] + "\n" for line in Path(path).open()))


def masked_mean(values, mask):
    """gan_bpm's masked_mean: one numerator and one denominator over the whole batch."""
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def predictor_loss(y, gt, valid, ext):
    """(total, terms). ext_low + ext_high = ext_both exactly: (2gt − 1)² split at its middle."""
    err = y - gt
    sq = err * err
    l1 = masked_mean(err.abs(), valid)
    low = masked_mean(ext_weight(gt, "low") * sq, valid)
    high = masked_mean(ext_weight(gt, "high") * sq, valid)
    terms = {"l1": l1, "ext_low": low, "ext_high": high, "ext_both": low + high}
    total = l1 if ext is None else l1 + ext["weight"] * masked_mean(ext_weight(gt, ext["side"]) * sq, valid)
    return total, terms


def git_commit():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def inputs(batch, device, mask_cond):
    cond, gt, mask, valid = (batch[k].to(device, non_blocking=True).float()
                             for k in ("cond_image", "gt_image", "mask", "loss_mask"))
    return (cond * mask if mask_cond else cond), gt, mask, valid


@torch.no_grad()
def evaluate(G, loader, device, mask_cond, scale, target):
    """The monitor: pooled over its slices. MAE in label units; the core's bias (prediction − label
    on the label's core voxels, label units) and its Dice at the cutoff, pooled."""
    G.eval()
    side, cut = CORE[target]
    s = defaultdict(float)
    for batch in loader:
        x, gt, mask, valid = inputs(batch, device, mask_cond)
        y = G(x)[0] * mask
        err = y - gt
        s["valid"] += float(valid.sum())
        s["abs"] += float((err.abs() * valid).sum())
        s["low"] += float((ext_weight(gt, "low") * err * err * valid).sum())
        s["high"] += float((ext_weight(gt, "high") * err * err * valid).sum())
        core = ((gt < cut) if side == "below" else (gt > cut)) & (valid > 0)
        called = ((y < cut) if side == "below" else (y > cut)) & (valid > 0)
        s["core"] += float(core.sum())
        s["called"] += float(called.sum())
        s["both"] += float((core & called).sum())
        s["core_err"] += float(err[core].sum())
    n = s["valid"] or float("nan")
    ext_both = (s["low"] + s["high"]) / n
    return {"mon_l1": s["abs"] / n, "mon_mae_label": scale * s["abs"] / n, "mon_ext_both": ext_both,
            "mon_ext_low_share": s["low"] / (s["low"] + s["high"]) if s["low"] + s["high"] else float("nan"),
            "mon_core_voxels": int(s["core"]),
            "mon_core_bias_label": scale * s["core_err"] / s["core"] if s["core"] else float("nan"),
            "mon_core_dice": 2 * s["both"] / (s["core"] + s["called"]) if s["core"] + s["called"] else float("nan"),
            "mon_called_over_core": s["called"] / s["core"] if s["core"] else float("nan")}


def append_csv(path, row):
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def save_atomic(obj, path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def write_heldout(G, dataset, device, mask_cond, out, workers):
    """Init_ (the predictor's map × mask), GT_ and Mask_ for every held-out slice, as Palette.test
    names and shapes them ((H, W) float32), so the core check reads them as it reads MRDPM's."""
    G.eval()
    out.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=workers, pin_memory=True)
    names = []
    for batch in loader:
        x, gt, mask, _valid = inputs(batch, device, mask_cond)
        y = (G(x)[0] * mask).cpu()
        for i, name in enumerate(batch["path"]):
            np.save(out / f"Init_{name}", y[i].numpy().squeeze())
            np.save(out / f"GT_{name}", batch["gt_image"][i].float().numpy().squeeze())
            np.save(out / f"Mask_{name}", batch["mask_image"][i].float().numpy().squeeze())
            names.append(name)
    return names


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("-c", "--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--debug", action="store_true", help=f"1 epoch on {DEBUG_SLICES} slices; the checkpoint "
                                                         "is marked debug, and Network refuses it without allow_debug")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--stop-after", type=int, default=0, help="stop after this many epochs of this call "
                                                              "(scripts/bran_ext_checks.py: the resume)")
    args = p.parse_args(argv)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    torch.backends.cudnn.benchmark = True

    from data.dataset import MCTPDataset_mask                                       # noqa: PLC0415
    cfg = load_config(args.config)
    ip = cfg["model"]["which_networks"][0]["args"].get("initial_predictor") or {}
    if ip.get("type") != "bran":
        sys.exit(f"{args.config}: initial_predictor.type is {ip.get('type')!r}, not 'bran'")
    if ip.get("expect_fold") is None:
        sys.exit(f"{args.config}: initial_predictor.expect_fold is not set")
    blocks = {"train": cfg["datasets"]["train"]["which_dataset"]["args"],
              "monitor": cfg["datasets"]["train"]["which_val_dataset"]["args"],
              "test": cfg["datasets"]["test"]["which_dataset"]["args"]}
    fold = int(ip["expect_fold"])
    for part, a in blocks.items():
        if not str(a["split_json"]).endswith(f"fold_{fold}/split.json") or a["part"] != part:
            sys.exit(f"{part}: split {a['split_json']} / part {a['part']} is not fold {fold}'s {part}")
    if len({json.dumps({k: a[k] for k in ("mctp_root", "split_json", "label_type")}) for a in blocks.values()}) != 1:
        sys.exit("the train, monitor and test blocks read different roots, splits or labels")
    target = blocks["train"]["label_type"]
    recipe, ext = pretrain_recipe(ip.get("pretrain")), normalize_ext(ip.get("ext"))
    in_channels, mask_cond = int(ip.get("in_channels", 3)), bool(ip.get("mask_cond", False))
    scale = float(cfg["model"]["which_model"]["args"].get("health_log", {}).get("label_scale", 1.0))
    epochs = 1 if args.debug else recipe["epochs"]

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "bran_G.pth").exists():
        sys.exit(f"{out / 'bran_G.pth'} exists: this pretraining is done (delete it to redo)")

    data = {part: MCTPDataset_mask(**a) for part, a in blocks.items()}
    if set(data["train"].case_ids) & set(data["test"].case_ids):
        sys.exit("training and held-out cases overlap")
    used = {part: (Subset(d, range(min(len(d), DEBUG_SLICES[part]))) if args.debug else d) for part, d in data.items()}
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  BRAN pretraining, {target} fold {fold}, {in_channels} channels, "
          f"mask_cond {mask_cond}, ext {ext}, recipe {recipe}, debug {args.debug}, device {device}")
    print("slices: " + ", ".join(f"{k} {len(v)}" for k, v in used.items()), flush=True)
    loader = DataLoader(used["train"], batch_size=recipe["batch_size"], shuffle=True, drop_last=True,
                        num_workers=args.workers, pin_memory=True)
    monitor = DataLoader(used["monitor"], batch_size=8, shuffle=False, num_workers=min(args.workers, 4),
                         pin_memory=True)
    if len(loader) == 0:
        sys.exit(f"{len(used['train'])} training slices make no batch of {recipe['batch_size']}")

    G = build_bran(in_channels)
    init_weights(G, recipe["init"], recipe["init_gain"])
    G.to(device)
    opt = torch.optim.Adam(G.parameters(), lr=recipe["lr"], betas=tuple(recipe["betas"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: max(0.0, 1.0 - e / epochs))
    identity = expected_meta(target, in_channels, mask_cond, ext, recipe, fold)
    identity["debug"] = bool(args.debug)
    start = 0
    if (out / "last_state.pt").exists():
        state = torch.load(out / "last_state.pt", map_location=device, weights_only=False)
        if state["identity"] != identity:
            sys.exit(f"{out / 'last_state.pt'} is another pretraining's ({state['identity']}), not {identity}")
        G.load_state_dict(state["generator"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start = state["epochs_done"]
        print(f"resuming after epoch {start} from {out / 'last_state.pt'}", flush=True)

    log = out / "pretrain_log.csv"
    run_epochs = 0
    for epoch in range(start, epochs):
        G.train()
        t0, sums, steps = time.time(), defaultdict(float), 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for batch in loader:
            x, gt, mask, valid = inputs(batch, device, mask_cond)
            y = G(x)[0] * mask
            total, terms = predictor_loss(y, gt, valid, ext)
            if not torch.isfinite(total):
                sys.exit(f"epoch {epoch + 1}, step {steps + 1}: non-finite loss {float(total)}")
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
            sums["loss"] += float(total)
            for k, v in terms.items():
                sums[k] += float(v)
            steps += 1
        lr = opt.param_groups[0]["lr"]
        sched.step()
        mon = evaluate(G, monitor, device, mask_cond, scale, target)
        row = {"epoch": epoch + 1, "lr": lr, "steps": steps, **{f"train_{k}": v / steps for k, v in sums.items()},
               "train_ext_low_share": sums["ext_low"] / sums["ext_both"] if sums["ext_both"] else float("nan"),
               **mon, "seconds": time.time() - t0,
               "gpu_peak_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0}
        append_csv(log, row)
        save_atomic({"identity": identity, "epochs_done": epoch + 1, "generator": G.state_dict(),
                     "optimizer": opt.state_dict(), "scheduler": sched.state_dict()}, out / "last_state.pt")
        print(f"{datetime.now():%m-%d %H:%M}  epoch {epoch + 1}/{epochs}  loss {row['train_loss']:.5f}  "
              f"L1 {row['train_l1']:.5f}  monitor MAE {mon['mon_mae_label']:.4f}  core Dice {mon['mon_core_dice']:.3f}  "
              f"lr {lr:.2e}  {row['seconds']:.0f} s  GPU peak {row['gpu_peak_gib']:.1f} GiB", flush=True)
        run_epochs += 1
        if args.stop_after and run_epochs >= args.stop_after and epoch + 1 < epochs:
            print(f"stopping after {run_epochs} epochs (--stop-after)")
            return 3

    names = write_heldout(G, used["test"], device, mask_cond, out / "heldout", args.workers)
    (out / "heldout.json").write_text(json.dumps({
        "fold": fold, "target": target, "slices": len(names), "case_ids": sorted({n.split("_slice")[0] for n in names}),
        "debug": bool(args.debug)}, indent=1) + "\n")
    meta = {**identity, "epochs_done": epochs, "split_json": str(blocks["train"]["split_json"]),
            "mctp_root": str(blocks["train"]["mctp_root"]), "train_case_ids": data["train"].case_ids,
            "monitor_case_ids": data["monitor"].case_ids, "heldout_case_ids": data["test"].case_ids,
            "n_train_slices": len(used["train"]), "mrdpm_commit": git_commit(), "config": str(args.config),
            "date": datetime.now().isoformat(timespec="seconds"), "seed": torch.initial_seed()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    save_atomic({"generator": {k: v.cpu() for k, v in G.state_dict().items()}, "meta": meta}, out / "bran_G.pth")
    print(f"wrote {out / 'bran_G.pth'} and {len(names)} held-out maps in {out / 'heldout'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
