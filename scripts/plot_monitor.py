"""How MRDPM's monitor maps change over training: value histograms against the label's, the
distribution's numbers per pass, and a few slices across epochs (ctp_only_plan.md §16).

Reads what every monitor pass saved (every val_epoch = 10 epochs; 16 slices of 3 training
patients): <run>/results/val/<epoch>/{Out_,GT_,Mask_}<case>_slice<k>.npy. They are training
patients, so this shows how training goes, not how good the model is. For a resumed run, give
its folders in order (the first run, then each resume): a later folder's pass replaces an
earlier one at the same epoch.

    python scripts/plot_monitor.py <run> [<resumed run> ...] [--epochs 10 50 100 200] [--out DIR]

numpy and matplotlib only: it runs on a laptop with the folders copied, or on the box. Writes
to --out (default: the last run's health/):
  monitor_histograms.png    the label's values against MRDPM's at a few epochs, pooled over
                            the slices; the second panel zooms on the tail the lesion lives in
  monitor_distribution.png  per pass: MAE, bias, spread ratio, W1, and the share of voxels
                            past the lesion threshold (rCBF < 0.3, Tmax > 6 s), MRDPM's
                            against the label's
  monitor_slices.png        a few slices: the label, then MRDPM at the chosen epochs, the
                            lesion region outlined in red
Voxels: inside the brain mask (Mask_), leaving out those where the label is exactly 0, which
are mostly no-data holes (RAPID's blanked voxels; UCLA's −1 is clipped to 0).
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

# label range (Out_ and GT_ are in [0, 1] of it), name, unit, lesion threshold and side
TARGETS = {"rcbf": (2.0, "rCBF", "", 0.3, "below"), "tmax": (12.0, "Tmax", "s", 6.0, "above")}
RUN_NAME = re.compile(r"_ctp_only_f13_(tmax|rcbf)_")


def target_of(runs):
    found = set()
    for r in runs:
        m = RUN_NAME.search(r.resolve().name)
        if m:
            found.add(m.group(1))
    if len(found) != 1:
        sys.exit(f"can't tell the target from the folder names ({[r.resolve().name for r in runs]}): "
                 "pass --target rcbf or --target tmax")
    return found.pop()


def passes(runs):
    """{epoch: folder}, ascending; a later run's pass replaces an earlier one."""
    out = {}
    for run in runs:
        for d in (run / "results" / "val").glob("*"):
            if d.is_dir() and d.name.isdigit() and any(d.glob("Out_*.npy")):
                out[int(d.name)] = d
    return dict(sorted(out.items()))


def load_pass(folder):
    """{slice name: (MRDPM, label, mask)}, the maps in [0, 1] of the label range."""
    out = {}
    for p in sorted(folder.glob("Out_*.npy")):
        name = p.name[len("Out_"):]
        gt, mask = folder / f"GT_{name}", folder / f"Mask_{name}"
        if gt.is_file() and mask.is_file():
            out[name] = (np.load(p).astype(np.float64), np.load(gt).astype(np.float64), np.load(mask) > 0.5)
    return out


def voxels(maps, scale):
    """(MRDPM, label) values in label units over every slice's scored voxels."""
    v, g = [], []
    for out, gt, mask in maps.values():
        keep = mask & (gt > 0)
        v.append(out[keep] * scale)
        g.append(gt[keep] * scale)
    return np.concatenate(v), np.concatenate(g)


def past(values, threshold, side):
    return float((values < threshold).mean() if side == "below" else (values > threshold).mean())


def stats(v, g, threshold, side):
    return {"MAE": float(np.abs(v - g).mean()), "bias": float((v - g).mean()),
            "spread ratio": float(v.std() / g.std()) if g.std() else float("nan"),
            "W1": float(np.abs(np.sort(v) - np.sort(g)).mean()),      # the same voxels: equal counts
            "past MRDPM": past(v, threshold, side), "past label": past(g, threshold, side)}


def pick_epochs(available, n=5):
    idx = np.unique(np.linspace(0, len(available) - 1, min(n, len(available))).round().astype(int))
    return [available[i] for i in idx]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run", type=Path, nargs="+", help="a training run folder; for a resumed run, "
                   "the first run, then each resume")
    p.add_argument("--epochs", type=int, nargs="+", help="the epochs to draw (default: 5, first to last)")
    p.add_argument("--slices", type=int, default=4, help="slices in monitor_slices.png")
    p.add_argument("--out", type=Path, help="default: the last run's health/")
    p.add_argument("--target", choices=sorted(TARGETS), help="default: from the folder names")
    args = p.parse_args()

    target = args.target or target_of(args.run)
    scale, label, unit, threshold, side = TARGETS[target]
    u = f" ({unit})" if unit else ""
    found = passes(args.run)
    if not found:
        sys.exit(f"no monitor pass under {[str(r / 'results' / 'val') for r in args.run]}")
    epochs = args.epochs or pick_epochs(list(found))
    missing = [e for e in epochs if e not in found]
    if missing:
        sys.exit(f"no monitor pass at epochs {missing}; there are {list(found)}")
    out_dir = args.out or args.run[-1] / "health"
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, values, drawn, names = [], {}, {}, None
    for epoch, folder in found.items():
        maps = load_pass(folder)
        if names is None:
            names = sorted(maps)
        elif sorted(maps) != names:
            print(f"note: epoch {epoch}'s monitor slices differ from the first pass's")
        v, g = voxels(maps, scale)
        rows.append({"epoch": epoch, "slices": len(maps), **stats(v, g, threshold, side)})
        if epoch in epochs:
            values[epoch] = v
            drawn[epoch] = maps
            truth = g
    sign = "<" if side == "below" else ">"
    print(f"{label}, {len(found)} monitor passes, {rows[-1]['slices']} slices; "
          f"voxels in the mask with a non-zero label: {truth.size}")
    print(f"{'epoch':>5} {'MAE':>7} {'bias':>7} {'spread':>7} {'W1':>7}  {label} {sign} {threshold:g}: MRDPM / label")
    for r in rows:
        print(f"{r['epoch']:>5} {r['MAE']:7.3f} {r['bias']:7.3f} {r['spread ratio']:7.3f} {r['W1']:7.3f}  "
              f"{r['past MRDPM']:.3f} / {r['past label']:.3f}")
    title = " + ".join(r.name for r in args.run)
    colors = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(epochs)))

    # 1. histograms
    lo, hi = -0.05 * scale, 1.1 * scale
    bins = np.linspace(lo, hi, 89)
    centers = (bins[:-1] + bins[1:]) / 2
    tail = (0, 3 * threshold) if side == "below" else (threshold * 2 / 3, scale)
    hist = lambda a: np.histogram(np.clip(a, lo, hi), bins=bins, density=True)[0]   # noqa: E731
    h_label, h_out = hist(truth), {ep: hist(values[ep]) for ep in epochs}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, xlim in zip(axes, [(lo, hi), tail]):
        ax.fill_between(centers, h_label, step="mid", color="0.75", label="label")
        for c, ep in zip(colors, epochs):
            ax.step(centers, h_out[ep], where="mid", color=c, label=f"MRDPM, epoch {ep}")
        ax.axvline(threshold, color="red", ls=":", lw=1)
        ax.set(xlim=xlim, xlabel=f"{label}{u}", ylabel="density")
    axes[0].set_title(f"{label} over the monitor slices (red: the lesion threshold {threshold:g}{u})")
    axes[1].set_title(f"the lesion tail, {label} {sign} {threshold:g}")
    in_tail = (centers >= tail[0]) & (centers <= tail[1])
    top = max(h[in_tail].max() for h in [h_label] + list(h_out.values()))
    axes[1].set_ylim(0, 1.15 * top if top > 0 else 1)
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "monitor_histograms.png", dpi=90)
    plt.close(fig)

    # 2. the distribution per pass
    e = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.8))
    for ax, key, ref in zip(axes, ["MAE", "bias", "spread ratio", "W1"], [None, 0.0, 1.0, 0.0]):
        ax.plot(e, [r[key] for r in rows], "o-")
        if ref is not None:
            ax.axhline(ref, color="0.6", ls="--", lw=1)
        ax.set(title=f"{key}{u if key != 'spread ratio' else ''}", xlabel="epoch")
    axes[4].plot(e, [r["past MRDPM"] for r in rows], "o-", label="MRDPM")
    axes[4].plot(e, [r["past label"] for r in rows], "--", color="0.4", label="label")
    axes[4].set(title=f"share of voxels with {label} {sign} {threshold:g}", xlabel="epoch")
    axes[4].legend()
    fig.suptitle(f"{title}   (spread ratio = std(MRDPM) / std(label): below 1 = squeezed toward the middle)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "monitor_distribution.png", dpi=90)
    plt.close(fig)

    # 3. slices across epochs
    picks = [names[i] for i in np.unique(np.linspace(0, len(names) - 1, min(args.slices, len(names)))
                                         .round().astype(int))]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.6")
    fig, axes = plt.subplots(len(picks), 1 + len(epochs), figsize=(3.2 * (1 + len(epochs)), 3.2 * len(picks)),
                             squeeze=False)
    for row, name in zip(axes, picks):
        first = drawn[epochs[0]][name]
        panels = [(first[1], "label")] + [(drawn[ep][name][0], f"MRDPM, epoch {ep}") for ep in epochs]
        mask = first[2]
        for ax, (a, what) in zip(row, panels):
            im = ax.imshow(np.ma.masked_where(~mask, a * scale), vmin=0, vmax=scale, cmap=cmap)
            region = mask & ((a * scale < threshold) if side == "below" else (a * scale > threshold))
            if region.any():
                ax.contour(region, levels=[0.5], colors="red", linewidths=0.6)
            ax.set_title(what, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(name.replace(".npy", ""), fontsize=8)
        fig.colorbar(im, ax=row.tolist(), shrink=0.8, label=f"{label}{u}")
    fig.suptitle(f"{title}   (red: {label} {sign} {threshold:g}; grey: outside the brain mask)", fontsize=9)
    fig.savefig(out_dir / "monitor_slices.png", dpi=80)
    plt.close(fig)
    print(f"wrote monitor_histograms.png, monitor_distribution.png, monitor_slices.png to {out_dir}")


if __name__ == "__main__":
    main()
