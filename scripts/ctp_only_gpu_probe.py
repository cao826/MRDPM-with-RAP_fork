"""ctp_only_plan.md §13 step 3 / §14: GPU memory and seconds per step at 512, by batch size.

Runs on the GPU, inside condor/ctp_only_smoke.sh after the run.py smoke:

    python scripts/ctp_only_gpu_probe.py --out /radraid2/colivares/mrdpm_ctp_only/smoke

- Training step: what Palette.train_step does per batch (Network.forward with loss_mask,
  backward, Adam, the EMA update), at batch 1, 2, 4 and 8.
- Sampling step: one p_sample, which restoration repeats 1,000 times per batch, at 1, 4, 8, 16.

Real train-part slices and the real config's Network and optimizer. A series stops at its first
out-of-memory. Each row also gives what that batch size costs per epoch (the train part), per
monitor pass and per test pass.
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from ctp_only_box_checks import CONFIG, Log, load_config     # also puts MRDPM's root first on sys.path

TRAIN_BATCHES = (1, 2, 4, 8)
SAMPLE_BATCHES = (1, 4, 8, 16)


def timed(fn, dev, n):
    """Median seconds of `n` calls after one warm-up call."""
    fn()
    seconds = []
    for _ in range(n):
        torch.cuda.synchronize(dev)
        t0 = time.time()
        fn()
        torch.cuda.synchronize(dev)
        seconds.append(time.time() - t0)
    return float(np.median(seconds))


def series(name, batches, make_step, dev, n):
    """Time `make_step(b)()` for each batch size `b`; `make_step` does the untimed set-up."""
    rows = []
    for b in batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        try:
            s = timed(make_step(b), dev, n)
        except torch.cuda.OutOfMemoryError:
            print(f"     {name} batch {b:>2}: out of memory", flush=True)
            rows.append({"batch": b, "oom": True})
            break
        peak = torch.cuda.max_memory_allocated(dev) / 2**30
        print(f"     {name} batch {b:>2}: {s:.3f} s/step, peak {peak:.2f} GiB allocated", flush=True)
        rows.append({"batch": b, "seconds_per_step": s, "peak_gib": peak})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--steps", type=int, default=5, help="timed steps per batch size, after one warm-up")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        print("CUDA is not available in this process: run it through condor/ctp_only_smoke.sub")
        sys.exit(2)
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    print(f"{props.name}, {props.total_memory / 2**30:.1f} GiB")

    cfg = load_config(args.config)
    net_opt = cfg["model"]["which_networks"][0]
    gan_bpm = net_opt["args"]["initial_predictor"]["gan_bpm_root"]
    if gan_bpm not in sys.path:
        sys.path.append(gan_bpm)
    from data.dataset import MCTPDataset_mask
    parts = {p: MCTPDataset_mask(**cfg["datasets"][s][k]["args"]) for p, (s, k) in
             {"train": ("train", "which_dataset"), "monitor": ("train", "which_val_dataset"),
              "test": ("test", "which_dataset")}.items()}
    sizes = {p: len(d) for p, d in parts.items()}
    train = parts["train"]
    picks = np.linspace(0, len(train) - 1, max(TRAIN_BATCHES + SAMPLE_BATCHES)).round().astype(int)
    batch = default_collate([train[int(i)] for i in picks])
    batch = {k: v.to(dev) for k, v in batch.items() if torch.is_tensor(v)}
    gt, cond, mask, loss_mask = (batch[k] for k in ("gt_image", "cond_image", "mask", "loss_mask"))

    import models.loss
    from models import define_network
    from models.model import EMA
    net = define_network(Log(), {"phase": "train"}, net_opt).to(dev)
    net.set_loss(models.loss.mse_loss)
    palette = cfg["model"]["which_model"]["args"]
    ema, ema_avg = copy.deepcopy(net), EMA(beta=palette["ema_scheduler"]["ema_decay"])
    optimizer = torch.optim.Adam([q for q in net.parameters() if q.requires_grad], **palette["optimizers"][0])

    print("\ntraining step (forward with loss_mask, backward, Adam)")
    net.set_new_noise_schedule(device=dev, phase="train")
    net.train()

    def train_step(b):
        def step():
            optimizer.zero_grad()
            loss = net(gt[:b], cond[:b], mask=mask[:b], loss_mask=loss_mask[:b])
            loss.backward()
            optimizer.step()
            loss.item()
            ema_avg.update_model_average(ema, net)
        return step

    train_rows = series("train", TRAIN_BATCHES, train_step, dev, args.steps)
    optimizer.zero_grad(set_to_none=True)
    for r in train_rows:
        if not r.get("oom"):
            r["hours_per_epoch"] = math.ceil(sizes["train"] / r["batch"]) * r["seconds_per_step"] / 3600

    print("\nsampling step (one p_sample of restoration's 1,000)")
    net.set_new_noise_schedule(device=dev, phase="test")
    net.eval()

    def sample_step(b):
        with torch.no_grad():                   # once per batch in restoration, so not timed
            _, init_cond = net.initial_net(cond[:b])
            y_cond = net.unet_cond(cond[:b], mask[:b])
        y_t = torch.randn_like(gt[:b]) * mask[:b] + gt[:b] * (1. - mask[:b])
        t = torch.full((b,), net.num_timesteps - 1, device=dev, dtype=torch.long)
        return lambda: net.p_sample(y_t, t, y_cond=y_cond, init_cond=init_cond)

    sample_rows = series("sample", SAMPLE_BATCHES, sample_step, dev, args.steps)
    for r in sample_rows:
        if not r.get("oom"):
            per_pass = lambda n: math.ceil(n / r["batch"]) * net.num_timesteps * r["seconds_per_step"] / 3600
            r["hours_per_monitor_pass"], r["hours_per_test_pass"] = per_pass(sizes["monitor"]), per_pass(sizes["test"])

    print(f"\nwhat that costs ({sizes['train']} train, {sizes['monitor']} monitor, {sizes['test']} test slices;"
          f" {net.num_timesteps} sampling steps per batch)")
    for r in train_rows:
        if not r.get("oom"):
            print(f"     train batch {r['batch']:>2}: {r['hours_per_epoch']:.2f} h per epoch")
    for r in sample_rows:
        if not r.get("oom"):
            print(f"     sample batch {r['batch']:>2}: {r['hours_per_monitor_pass']:.2f} h per monitor pass, "
                  f"{r['hours_per_test_pass']:.1f} h per test pass")
    out = args.out / "gpu_probe.json"
    out.write_text(json.dumps({"device": props.name, "total_gib": props.total_memory / 2**30, "slices": sizes,
                               "train": train_rows, "sample": sample_rows}, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
