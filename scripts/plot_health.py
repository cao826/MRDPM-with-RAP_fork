"""Training curves from a run's health files (models/health.py), at any point during the run.

CPU, anywhere with pandas and matplotlib, e.g. on 14958 in the job image:

    python scripts/plot_health.py /radraid2/colivares/mrdpm_ctp_only/runs/<experiment>
    python scripts/plot_health.py <first run> <its resume>     # a resumed run, in order

Writes <last experiment>/health/curves.png and prints the last few rows of each CSV. With
several folders (a run and its resumes), their rows are joined by epoch, a later folder's row
replacing an earlier one's.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def read(runs, name):
    frames = [pd.read_csv(r / "health" / name) for r in runs if (r / "health" / name).is_file()]
    if not frames:
        return None
    joined = pd.concat(frames, ignore_index=True)
    return joined.drop_duplicates("epoch", keep="last").sort_values("epoch").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run", type=Path, nargs="+", help="an experiment directory, the one holding "
                   "health/; for a resumed run, the first run, then each resume")
    args = p.parse_args()
    health = args.run[-1] / "health"
    train = read(args.run, "train.csv")
    if train is None:
        sys.exit(f"no health/train.csv in {[str(r) for r in args.run]}: no epoch has finished "
                 "yet, or the run has no health_log")
    monitor = read(args.run, "monitor.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(train.tail(5).to_string(index=False))
        if monitor is not None:
            print()
            print(monitor.tail(5).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    e = train["epoch"]
    ax[0, 0].plot(e, train["loss_diff"])
    ax[0, 0].set(title="diffusion loss (noise MSE over loss_mask)", yscale="log")
    ax[0, 1].plot(e, train["loss_init"])
    ax[0, 1].set(title="predictor's L1 vs the label (training term)")
    ax[0, 2].plot(e, train["grad_norm_mean"], label="mean")
    ax[0, 2].plot(e, train["grad_norm_max"], label="max", alpha=0.7)
    ax[0, 2].set(title="gradient norm per step", yscale="log")
    ax[0, 2].legend()
    if monitor is not None and len(monitor):
        out = [c for c in monitor.columns if c.startswith("mae_output")][0]
        init = [c for c in monitor.columns if c.startswith("mae_predictor")][0]
        unit = out.rsplit("_", 1)[-1]
        ax[1, 0].plot(monitor["epoch"], monitor[out], "o-", label="MRDPM")
        ax[1, 0].plot(monitor["epoch"], monitor[init], "s--", label="predictor alone")
        ax[1, 0].set(title=f"monitor: error over loss_mask ({unit})")
        ax[1, 0].legend()
    else:
        ax[1, 0].set(title="monitor: no pass yet")
    ax[1, 1].plot(e, train["lr"])
    ax[1, 1].set(title="learning rate")
    ax[1, 2].plot(e, train["seconds"] / 60, label="minutes per epoch")
    ax[1, 2].plot(e, train["gpu_peak_gib"], label="peak GPU GiB")
    ax[1, 2].set(title="time and memory")
    ax[1, 2].legend()
    title = " + ".join(r.name for r in args.run)
    if train["nonfinite_steps"].sum():
        fig.suptitle(f"{title}: {int(train['nonfinite_steps'].sum())} non-finite steps", color="red")
    else:
        fig.suptitle(title)
    for a in ax.flat:
        a.set_xlabel("epoch")
    fig.tight_layout()
    out_png = health / "curves.png"
    fig.savefig(out_png, dpi=80)
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
