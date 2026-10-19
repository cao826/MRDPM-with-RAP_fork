"""Training-health log for the ctp-only runs (ctp_only_plan.md §16).

Palette calls it when its config has `health_log`; without that key nothing here runs. It
writes, under `{experiments_root}/health/`:

- `train.csv`, one row per epoch: the loss and its two terms, the learning rate and where the
  schedule is, the gradient norm (mean and max over the epoch's steps), non-finite steps,
  seconds and slices per second, peak GPU memory.
- `monitor.csv`, one row per monitor pass: the pooled absolute error over `loss_mask`, in label
  units (`label_scale`, e.g. 12 for seconds of Tmax), for the final output and for the initial
  predictor alone on the same slices; how much of the envelope `loss_mask` covers.
- `monitor_e{epoch:03d}.png`: label, predictor, output and both errors for a few slices.

It only reads: no weight, gradient or loss is changed (scripts/ctp_only_checks.py checks this).
"""
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch


class HealthLog:
    def __init__(self, root, label_scale=1.0, unit="", panels=4):
        self.dir = Path(root) / "health"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.label_scale, self.unit, self.panels = float(label_scale), unit, int(panels)

    def _append(self, name, row):
        path = self.dir / name
        new = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)

    def start_epoch(self):
        self.t0 = time.time()
        self.sums = {"loss": 0.0, "loss_diff": 0.0, "loss_init": 0.0}
        self.steps = self.term_steps = self.nonfinite = self.slices = 0
        self.grad_norms = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @torch.no_grad()
    def step(self, net, loss, batch_size):
        """After `loss.backward()` and before the optimizer step."""
        net = getattr(net, "module", net)
        value = float(loss)
        self.steps += 1
        self.slices += batch_size
        if not math.isfinite(value):
            self.nonfinite += 1
            return
        self.sums["loss"] += value
        terms = getattr(net, "loss_terms", None)
        if terms is not None:
            self.sums["loss_diff"] += float(terms["loss_diff"])
            self.sums["loss_init"] += float(terms["loss_init"])
            self.term_steps += 1
        norms = [p.grad.norm() for p in net.parameters() if p.grad is not None]
        if norms:
            self.grad_norms.append(float(torch.stack(norms).norm()))

    def end_epoch(self, epoch, iters, lr, lr_epoch):
        finite = self.steps - self.nonfinite
        seconds = time.time() - self.t0
        mean = lambda total, n: total / n if n else float("nan")
        self._append("train.csv", {
            "epoch": epoch, "iters": iters, "lr": lr, "lr_epoch": lr_epoch,
            "loss": mean(self.sums["loss"], finite),
            "loss_diff": mean(self.sums["loss_diff"], self.term_steps),
            "loss_init": mean(self.sums["loss_init"], self.term_steps),
            "grad_norm_mean": float(np.mean(self.grad_norms)) if self.grad_norms else float("nan"),
            "grad_norm_max": float(np.max(self.grad_norms)) if self.grad_norms else float("nan"),
            "steps": self.steps, "nonfinite_steps": self.nonfinite,
            "seconds": seconds, "slices_per_s": self.slices / seconds if seconds else float("nan"),
            "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0,
        })

    def start_monitor(self):
        self.t_monitor = time.time()
        self.err_out = self.err_init = self.support = self.envelope = 0.0
        self.kept = []

    @torch.no_grad()
    def monitor_batch(self, net, cond, gt, mask, loss_mask, output):
        """After restoration, on the batch it sampled."""
        net = getattr(net, "module", net)
        y_initial = net.initial_net(cond)[0] * mask
        valid = mask if loss_mask is None else loss_mask
        self.err_out += float(((output - gt).abs() * valid).sum())
        self.err_init += float(((y_initial - gt).abs() * valid).sum())
        self.support += float(valid.sum())
        self.envelope += float(mask.sum())
        for i in range(len(gt)):
            self.kept.append([a[i, 0].float().cpu().numpy() for a in (gt, y_initial, output, valid)])

    def end_monitor(self, epoch):
        s = self.label_scale
        self._append("monitor.csv", {
            "epoch": epoch, "slices": len(self.kept),
            f"mae_output_{self.unit or 'label'}": s * self.err_out / self.support if self.support else float("nan"),
            f"mae_predictor_{self.unit or 'label'}": s * self.err_init / self.support if self.support else float("nan"),
            "support_of_envelope": self.support / self.envelope if self.envelope else float("nan"),
            "seconds": time.time() - self.t_monitor,
        })
        self._panel(epoch)

    def _panel(self, epoch):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        if not self.kept:
            return
        picks = np.unique(np.linspace(0, len(self.kept) - 1, min(self.panels, len(self.kept))).round().astype(int))
        s, unit = self.label_scale, f" ({self.unit})" if self.unit else ""
        maps_cmap, err_cmap = plt.get_cmap("viridis").copy(), plt.get_cmap("magma").copy()
        maps_cmap.set_bad("0.6")
        err_cmap.set_bad("0.6")
        fig, axes = plt.subplots(len(picks), 5, figsize=(17, 3.5 * len(picks)), squeeze=False)
        for row, i in zip(axes, picks):
            gt, y_initial, output, valid = self.kept[i]
            off = valid == 0
            show = lambda a: np.ma.masked_where(off, a * s)
            maps = [(gt, "label"), (y_initial, "predictor"), (output, "MRDPM")]
            for ax, (a, name) in zip(row[:3], maps):
                im = ax.imshow(show(a), vmin=0, vmax=s, cmap=maps_cmap)
                ax.set_title(name + unit)
            for ax, (a, name) in zip(row[3:], [(output, "|MRDPM − label|"), (y_initial, "|predictor − label|")]):
                err = np.abs(a - gt) * s
                im_err = ax.imshow(np.ma.masked_where(off, err), vmin=0, vmax=s / 2, cmap=err_cmap)
                ax.set_title(f"{name}, mean {err[~off].mean():.2f}{unit}" if (~off).any() else name)
            fig.colorbar(im, ax=row[:3].tolist(), shrink=0.8)
            fig.colorbar(im_err, ax=row[3:].tolist(), shrink=0.8)
            for ax in row:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(f"monitor, epoch {epoch} (grey: outside loss_mask)")
        fig.savefig(self.dir / f"monitor_e{epoch:03d}.png", dpi=60)
        plt.close(fig)
