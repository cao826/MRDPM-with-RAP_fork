"""The "right place or wrong level" check (ctp_only_decisions.md #28), on each fold's held-out
CV cases: when a model misses the rCBF < 0.3 core, is it because its values come out too high
(right place, wrong level), or because it doesn't know where the core is (wrong place)?

Per scan, inside the brain (gan_bpm's `support`), against the label's own core (gan_bpm's
level-3 region: rCBF < 0.3, or Tmax > 6 s with --target tmax):

  at the cutoff   Dice of the map's own region at the cutoff, as level 3 scores it
  cutoff sweep    the same at other cutoffs (rCBF 0.2 ... 1.0; Tmax 2 ... 10 s)
  matched         the map's k lowest voxels (highest for Tmax), k = the size of the label's
                  core, and their Dice with it: no cutoff, only where the map puts its
                  lowest values. With k voxels on both sides, Dice = the share that land in
                  the core.
  floor           k / N (N = brain voxels): what k voxels drawn at random score
  place score     (matched − floor) / (1 − floor): 0 = no information, 1 = the core exactly

How to read it: matched well above "at the cutoff" means right place, wrong level (a cutoff or
rescaling chosen on the CV data could recover it without retraining); matched near the floor
means wrong place, and no cutoff helps.

The maps, on the same slices: MRDPM (Out_: epoch 200, EMA weights, one sample), MRDPM's own
initial predictor (Init_: the fine-tuned one, when the test saved it), and gan_bpm λ_adv = 0 at
the fold's best.pt (that predictor before fine-tuning). The controls run first and stop the
script if they fail: the label as a map scores 1 on every scan, the label one slice off scores
well below 1, and k random voxels land on the analytic floor.

Reads MRDPM's CV test folders (condor/ctp_only_test_h100.sub with testset = cv) under --runs,
the CV data on 14958's /raid through gan_bpm's CaseLoader, and gan_bpm's fold models through
its CheckpointPredictor, read-only. Writes per_scan.csv, checks.csv and summary.txt to --out.
A GPU makes gan_bpm's predictions fast (condor/ctp_only_core_check.sub).

--gan-bpm-only scores gan_bpm's fold models alone, on every held-out case of every fold, with
no MRDPM test: minutes on a GPU, against about 6 h of sampling per fold for MRDPM. Since MRDPM
starts from gan_bpm λ_adv = 0's map, it also says whether MRDPM's starting point knows where
the core is. --recipe lsgan --recipe wgan_gp adds those fold models.

--run-dir scores other gan_bpm runs too, at --run-checkpoint (last.pt): gan_bpm's λ_ext
pilots (its ext_gan_plan.md step 1). With two or more, the first is the control, every other
is compared with it per scan, and the pre-declared pick rule of that plan's §1.4 is applied.
Since 2026-10-26 gan_bpm scores its own runs (its scripts/core_check.py, D-100; decisions #39):
this path stays only for that script's scan-for-scan comparison, and must not judge step 2's
fine-tunes (it knows §1.4's rule alone).

    python scripts/ctp_only_core_check.py [--target rcbf] [--fold 0 --fold 1] [--device cuda]
    python scripts/ctp_only_core_check.py --gan-bpm-only --recipe lsgan --recipe wgan_gp
    python scripts/ctp_only_core_check.py --gan-bpm-only --fold 0 \\
        --run-dir /radraid2/colivares/matrix/gan_bpm_rcbf_f13uclav2_adv0_ext50_e40_f0 \\
        --run-dir /radraid2/colivares/matrix/gan_bpm_rcbf_f13uclav2_adv0_ext200_e40_f0 \\
        --run-dir /radraid2/colivares/matrix/gan_bpm_rcbf_f13uclav2_adv0_ext500_e40_f0

The BRAN arms (bran_ext_plan.md §5). --variant scores an arm's CV test (Out_ as mrdpm_<arm>, Init_
as mrdpm_<arm>_init) beside today's MRDPM (mrdpm, when its test exists). --bran scores a pretrained
BRAN alone (bran_<name>) from the held-out maps scripts/pretrain_bran.py saved, before any
diffusion. Every map is scored on the slices all of them have. With two or more of either, the
first is the control of the same pre-declared pick rule:

    python scripts/ctp_only_core_check.py --gan-bpm-only --fold 0 \\
        --bran /radraid2/colivares/mrdpm_ctp_only/bran/rcbf_l1 --bran /radraid2/colivares/mrdpm_ctp_only/bran/rcbf_ext
    python scripts/ctp_only_core_check.py --fold 0 --variant bran_l1 --variant bran_ext
"""
import argparse
import json
import math
import re
import sys
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctp_only_isles18_maps as M                                       # noqa: E402

CV_ROOT = Path("/raid/colivares/mctp_dataset_f13_isles24_ucla_v2")
OUT = Path("/radraid2/colivares/mrdpm_ctp_only/cv_core_check")
# an arm's runs carry its name before the fold (bran_ext_plan.md §5); today's MRDPM has none
TEST_NAME = re.compile(r"^test_ctp_only_f13_(tmax|rcbf)_(?:([a-z0-9_]+?)_)?fold([0-4])_cv_\d{6}_\d{6}$")
TRAIN_NAME = re.compile(r"^train_ctp_only_f13_(tmax|rcbf)_(?:([a-z0-9_]+?)_)?(fold[0-4]|all)_\d{6}_\d{6}$")
INIT_FILE = re.compile(r"^Init_(.+)_slice(\d+)\.npy$")
SWEEP = {"rcbf": (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
         "tmax": (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)}
SIZE_BINS_ML = (0.0, 5.0, 15.0, 50.0, math.inf)
SHOWN = ("mrdpm", "mrdpm_init", "gan_bpm", "gan_bpm_lsgan", "gan_bpm_wgan_gp")     # report rows, in order
GAN_NAME = {"adv0": "gan_bpm", "lsgan": "gan_bpm_lsgan", "wgan_gp": "gan_bpm_wgan_gp"}
LABEL_OFF_MAX = 0.9          # the label one slice off must score below this, on average
SEED = 0


class Report:
    """Prints and keeps every line, for summary.txt."""

    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text, flush=True)
        self.lines.append(str(text))

    def table(self, title, df):
        self(f"\n### {title}")
        self(df.to_string() if len(df) else "(nothing)")


def fmt(v, digits=3):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{digits}f}"


# ===========================================================================
# MRDPM's CV test folders
# ===========================================================================

def read_cv_test(folder, epoch):
    """What a CV test folder is and where its slices are; refuses one the job didn't finish
    or that isn't the training run's own fold."""
    folder = Path(folder)
    m = TEST_NAME.match(folder.name)
    if not m:
        raise M.Blocked(f"{folder.name} is not a CV test folder (test_ctp_only_f13_<target>[_<arm>]_fold<k>_cv_<time>)")
    target, variant, fold = m.group(1), m.group(2), int(m.group(3))
    arm = f"{variant}_" if variant else ""
    for name in ("job_config.json", "source_run.json", "ddim_test.txt"):
        if not (folder / name).is_file():
            raise M.Blocked(f"{folder.name}: no {name} (ddim_test.txt: run.py writes it after the last slice)")
    cfg = json.loads((folder / "job_config.json").read_text())
    src = json.loads((folder / "source_run.json").read_text())
    args = cfg["datasets"]["test"]["which_dataset"]["args"]
    problems = []
    if cfg.get("name") != f"ctp_only_f13_{target}_{arm}fold{fold}_cv":
        problems.append(f"job config name {cfg.get('name')}")
    if args.get("label_type") != target or args.get("part") != "test" \
            or not str(args.get("split_json", "")).endswith(f"fold_{fold}/split.json"):
        problems.append(f"test data {args.get('label_type')} / {args.get('part')} / {args.get('split_json')}")
    if not cfg["model"]["which_model"]["args"].get("test_ema"):
        problems.append("not sampled with the EMA weights")
    if (src.get("testset"), src.get("mode"), int(src.get("epoch", -1))) != ("cv", "full", epoch):
        problems.append(f"source_run says {src.get('testset')} / {src.get('mode')} / epoch {src.get('epoch')}")
    run = Path(src.get("run", ""))
    t = TRAIN_NAME.match(run.name)
    if not t or t.groups() != (target, variant, f"fold{fold}"):
        problems.append(f"made from {run.name}, not a {target} {arm}fold {fold} training run")
    ema = run / "checkpoint" / f"{epoch}_Network_ema.pth"
    if not ema.is_file():
        problems.append(f"{ema} is gone")
    elif M.sha256(ema) != src.get("ema_sha256"):
        problems.append(f"{ema} is not the checkpoint the test sampled with (sha256)")
    results = sorted(p for p in (folder / "results" / "test").glob("*") if p.is_dir())
    if len(results) != 1:
        problems.append(f"results/test/ holds {len(results)} folders, expected one")
    if problems:
        raise M.Blocked(f"{folder.name}: " + "; ".join(problems))
    return {"folder": folder, "target": target, "variant": variant, "fold": fold, "results": results[0],
            "source": src}


def map_name(variant):
    """An MRDPM test's map: mrdpm for today's model, mrdpm_<arm> for an arm's."""
    return "mrdpm" if variant is None else f"mrdpm_{variant}"


def find_cv_tests(runs, target, folds, epoch, variant=None):
    """{fold: info} for every fold with one complete CV test folder of `variant` (None: today's
    MRDPM); refuses two."""
    found = {}
    arm = f"{variant}_" if variant else ""
    for folder in sorted(runs.glob(f"test_ctp_only_f13_{target}_{arm}fold*_cv_*")):
        m = TEST_NAME.match(folder.name)
        if m and m.group(2) == variant and folder.is_dir() and (folder / "ddim_test.txt").is_file() \
                and int(m.group(3)) in folds:
            found.setdefault(int(m.group(3)), []).append(folder)
    for k, folders in found.items():
        if len(folders) > 1:
            raise M.Blocked(f"fold {k}: {len(folders)} complete CV test folders "
                            f"({', '.join(f.name for f in folders)}); name the one to use")
    return {k: read_cv_test(f[0], epoch) for k, f in sorted(found.items())}


# ===========================================================================
# The check, one scan at a time
# ===========================================================================

def dice(a, b):
    total = int(a.sum()) + int(b.sum())
    return 1.0 if total == 0 else 2.0 * int(np.logical_and(a, b).sum()) / total


def extreme_k(values, k, side, rng):
    """Indices of the k lowest values (side "below") or highest ("above"), ties broken at random."""
    key = values if side == "below" else -values
    return np.lexsort((rng.random(values.size), key))[:k]


def scan_rows(case, maps, scale, target, region, fold):
    """One row per map for one scan, plus its base facts. `maps`: {name: physical (S, H, W)}.
    Returns (base, rows); rows is empty when the label has no core on this scan."""
    _tag, side, cut = REGION[target]
    brain = case.support
    label = np.asarray(scale.to_physical(case.labels.astype(np.float64)))
    core = region(label, brain, side, cut)
    k, n = int(core.sum()), int(brain.sum())
    voxel = case.meta.get("voxel_mL")
    floor = k / n if n else math.nan
    base = {"fold": fold, "case_id": case.case_id, "patient_id": str(case.meta.get("patient_id", case.case_id)),
            "n_slices": case.n_slices, "k": k, "n_brain": n, "floor": floor,
            "core_mL": k * voxel if voxel else math.nan}
    if k == 0 or n == 0:
        return base, []
    truth = core[brain]
    seed = [SEED, zlib.crc32(case.case_id.encode())]
    rows = []
    in_brain = label[brain]
    label_sd = float(in_brain.std())
    for name, phys in maps.items():
        rng = np.random.default_rng(seed + [zlib.crc32(name.encode())])
        values = np.asarray(phys, dtype=np.float64)[brain]
        idx = extreme_k(values, k, side, rng)
        matched = float(truth[idx].sum()) / k
        row = {**base, "map": name, "matched": matched,
               "place": (matched - floor) / (1.0 - floor) if floor < 1 else math.nan}
        for t in SWEEP[target]:
            row[f"dice_{t:g}"] = dice(region(phys, brain, side, t), core)
        row["at_cut"] = row[f"dice_{cut:g}"]
        row["vol_ratio"] = float(region(phys, brain, side, cut).sum()) / k
        # The level-1 and level-2 view on the same voxels: brain MAE (physical units) and how
        # much the map varies against the label (1 = as much; gan_bpm's squeeze is about 0.8).
        row["mae_brain"] = float(np.abs(values - in_brain).mean())
        row["spread_ratio"] = float(values.std()) / label_sd if label_sd > 0 else math.nan
        rows.append(row)
    rng = np.random.default_rng(seed + [zlib.crc32(b"random")])          # the floor, simulated
    idx = extreme_k(rng.random(n), k, side, rng)
    rows.append({**base, "map": "random", "matched": float(truth[idx].sum()) / k})
    return base, rows


def saved_slice(case, k, s, folder, stem, prefix):
    """The `prefix` map saved for slice `s` (row k of `case`), after its GT_ and Mask_ beside it
    are checked against the scan: a map on the wrong slice or the wrong brain is refused."""
    o, g, mk = (np.load(folder / f"{pre}{stem}").astype(np.float32) for pre in (prefix, "GT_", "Mask_"))
    env = case.envelope[k]
    if mk.shape != env.shape or not np.array_equal(mk.astype(bool), env) or not np.array_equal(mk, mk.astype(bool)):
        raise M.Blocked(f"{case.case_id} slice {s}: Mask_ is not the scan's envelope ({folder})")
    err = float(np.abs(2 * g[env] - 1 - case.labels[k][env]).max()) if env.any() else 0.0
    if err > M.GT_TOL:
        raise M.Blocked(f"{case.case_id} slice {s}: 2·GT_ − 1 differs from the label by {err:.2e} ({folder})")
    if not np.isfinite(o).all():
        raise M.Blocked(f"{case.case_id} slice {s}: {prefix} is not finite ({folder})")
    return o


def init_slices_in(folder):
    """{case_id: {slice_idx: Init_ path}}, as M.slices_in reads Out_."""
    out = {}
    for p in folder.glob("Init_*.npy"):
        m = INIT_FILE.match(p.name)
        if m:
            out.setdefault(m.group(1), {})[int(m.group(2))] = p
    return out


def read_bran(folder, fold, target, held, train, cv_root, allow_debug):
    """A pretrained BRAN's fold (scripts/pretrain_bran.py's fold_<k>/): {case: {slice: Init_ path}},
    or None when it has no such fold. Refuses one of another target, fold or data, one that
    trained on a held-out case, and a debug run unless `allow_debug`."""
    d = Path(folder) / f"fold_{fold}"
    if not (d / "bran_G.pth").is_file():
        return None
    meta = json.loads((d / "meta.json").read_text())
    problems = []
    if (meta.get("target"), meta.get("fold")) != (target, fold):
        problems.append(f"target / fold {meta.get('target')} / {meta.get('fold')}")
    if not str(meta.get("split_json", "")).endswith(f"fold_{fold}/split.json"):
        problems.append(f"split {meta.get('split_json')}")
    if Path(str(meta.get("mctp_root", ""))).name != Path(cv_root).name:
        problems.append(f"trained on {meta.get('mctp_root')}, not {Path(cv_root).name}")
    if set(map(str, meta.get("train_case_ids", []))) & set(held):
        problems.append("trained on held-out cases")
    if not set(map(str, meta.get("train_case_ids", []))) <= set(train):
        problems.append("trained on cases outside the fold's training cases")
    if not set(map(str, meta.get("heldout_case_ids", []))) <= set(held):
        problems.append("maps cases outside the fold's held-out set")
    if meta.get("debug") and not allow_debug:
        problems.append("a debug pretraining (--allow-debug-bran)")
    if problems:
        raise M.Blocked(f"{d}: " + "; ".join(problems))
    return init_slices_in(d / "heldout")


def mrdpm_slices(info, case, have):
    """The case's slices that MRDPM sampled, and its Out_ and Init_ there (Init_ None when
    the test didn't save it). Refuses a Mask_ that isn't the envelope or a GT_ that isn't
    the label: the maps would sit on the wrong slice or the wrong brain."""
    keep = np.array([s in have for s in case.slice_idx], dtype=bool)
    out, init = [], []
    for k, s in enumerate(case.slice_idx):
        if not keep[k]:
            continue
        p = have[s]
        o = saved_slice(case, k, s, p.parent, p.name[4:], "Out_")
        out.append(o)
        i = p.parent / f"Init_{p.name[4:]}"
        init.append(np.load(i).astype(np.float32) if i.is_file() else None)
    if any(x is None for x in init) and not all(x is None for x in init):
        raise M.Blocked(f"{case.case_id}: Init_ saved for some slices only")
    init = None if not init or init[0] is None else np.stack(init)
    return keep, (np.stack(out) if out else None), init


# ===========================================================================
# Tables
# ===========================================================================

def summary_table(df, maps, target):
    rows = {}
    for name in maps:
        d = df[df["map"] == name]
        if d.empty:
            continue
        rows[name] = {"scans": len(d), "cutoff mean": fmt(d["at_cut"].mean()),
                      "cutoff median": fmt(d["at_cut"].median()), "matched mean": fmt(d["matched"].mean()),
                      "matched median": fmt(d["matched"].median()), "floor": fmt(d["floor"].mean()),
                      "place score": fmt(d["place"].mean()),
                      "volume / label's (median)": fmt(d["vol_ratio"].median(), 2),
                      "MAE brain": fmt(d["mae_brain"].mean()), "spread": fmt(d["spread_ratio"].mean())}
    return pd.DataFrame.from_dict(rows, orient="index")


def pick(df, control, others, bootstrap_interval, mae_max=1.15):
    """gan_bpm's ext_gan_plan.md §1.4, pre-declared: among `others`, the map with the highest
    mean Dice at the cutoff whose paired per-scan difference from `control` has a 95 %
    bootstrap CI above 0 and whose mean brain MAE is at most `mae_max` × the control's.
    Returns (table, the pick or None)."""
    c = df[df["map"] == control].set_index(["fold", "case_id"])
    rows, qualified = {}, []
    for name in others:
        o = df[df["map"] == name].set_index(["fold", "case_id"])
        both = o.index.intersection(c.index)
        if both.empty:
            continue
        diff = (o.loc[both, "at_cut"] - c.loc[both, "at_cut"]).to_numpy(dtype=np.float64)
        lo, hi = bootstrap_interval(diff, np.mean, n=2000, ci=0.95, rng=np.random.default_rng(SEED))
        mae = float(o.loc[both, "mae_brain"].mean()) / float(c.loc[both, "mae_brain"].mean())
        ok = lo > 0 and mae <= mae_max
        rows[name] = {"scans": diff.size, "Dice at cutoff": fmt(float(o.loc[both, "at_cut"].mean())),
                      f"− {control}": f"{fmt(diff.mean())} [{fmt(lo)}, {fmt(hi)}]",
                      "MAE / control's": fmt(mae, 3), "qualifies": "yes" if ok else "no"}
        if ok:
            qualified.append((float(o.loc[both, "at_cut"].mean()), name))
    return pd.DataFrame.from_dict(rows, orient="index"), (max(qualified)[1] if qualified else None)


def per_fold_table(df, maps, column):
    t = df[df["map"].isin(maps)].pivot_table(index="map", columns="fold", values=column, aggfunc="mean")
    t = t.reindex([m for m in maps if m in t.index])
    t.columns = [f"fold {c}" for c in t.columns]
    return t.round(3)


def sweep_table(df, maps, target):
    cols = [f"dice_{t:g}" for t in SWEEP[target]]
    t = df[df["map"].isin(maps)].groupby("map")[cols].mean()
    t = t.reindex([m for m in maps if m in t.index])
    t.columns = [f"{c[5:]}" for c in cols]
    return t.round(3)


def size_table(df, maps):
    d = df[df["map"].isin(maps)].dropna(subset=["core_mL"]).copy()
    labels = [f"{lo:g}–{hi:g} mL" if hi < math.inf else f"≥ {lo:g} mL" for lo, hi in zip(SIZE_BINS_ML, SIZE_BINS_ML[1:])]
    d["label core"] = pd.cut(d["core_mL"], SIZE_BINS_ML, right=False, labels=labels)
    out = {}
    for name in maps:
        part = d[d["map"] == name]
        if part.empty:
            continue
        g = part.groupby("label core", observed=False)
        out[(name, "at cutoff")] = g["at_cut"].mean()
        out[(name, "matched")] = g["matched"].mean()
    floor = d[d["map"] == maps[0]].groupby("label core", observed=False)
    out[("", "floor")] = floor["floor"].mean()
    out[("", "scans")] = floor["case_id"].count()
    return pd.DataFrame(out).T.round(3)


def paired(df, a, b, column, bootstrap_interval):
    """Mean of (a − b) per scan, its 95 % bootstrap CI (2000 resamples, seed 0), and the share
    of scans where a is higher."""
    x = df[df["map"] == a].set_index(["fold", "case_id"])[column]
    y = df[df["map"] == b].set_index(["fold", "case_id"])[column]
    both = x.index.intersection(y.index)
    if both.empty:
        return None
    diff = (x.loc[both] - y.loc[both]).to_numpy(dtype=np.float64)
    lo, hi = bootstrap_interval(diff, np.mean, n=2000, ci=0.95, rng=np.random.default_rng(SEED))
    return f"{fmt(diff.mean())} [{fmt(lo)}, {fmt(hi)}], {a} higher on {np.mean(diff > 0):.0%} of {diff.size} scans"


# ===========================================================================
# Main
# ===========================================================================

REGION = {}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("test_folder", nargs="*", help="MRDPM CV test folders (default: every complete one under --runs)")
    p.add_argument("--target", choices=M.TARGETS, default="rcbf")
    p.add_argument("--gan-bpm-only", action="store_true",
                   help="gan_bpm's fold models alone, on every held-out case: no MRDPM test needed")
    p.add_argument("--recipe", action="append", choices=M.RECIPES,
                   help="gan_bpm recipes besides adv0 (MRDPM's predictor, always scored): lsgan, wgan_gp")
    p.add_argument("--run-dir", action="append", type=Path, default=[],
                   help="another gan_bpm run_cv folder to score (e.g. the λ_ext pilots), named by its "
                        "folder; with two or more, the first is the control of the pick (§1.4)")
    p.add_argument("--run-checkpoint", default="last.pt", help="the --run-dir runs' checkpoint (last.pt)")
    p.add_argument("--variant", action="append", default=[],
                   help="an arm's CV test to score too (bran_ext_plan.md §5), e.g. bran_l1; with two or "
                        "more, the first is the control of the pick")
    p.add_argument("--bran", action="append", type=Path, default=[],
                   help="a pretrained BRAN's folder (/radraid2/colivares/mrdpm_ctp_only/bran/<name>), scored "
                        "alone from its held-out maps; with two or more, the first is the control of the pick")
    p.add_argument("--allow-debug-bran", action="store_true", help="score a debug pretraining too")
    p.add_argument("--fold", action="append", type=int, choices=M.FOLDS, help="default: every fold with a test")
    p.add_argument("--gan-bpm", type=Path, default=M.GAN_BPM, help="gan_bpm's checkout, read only")
    p.add_argument("--cv-root", type=Path, default=CV_ROOT)
    p.add_argument("--matrix", type=Path, default=M.MATRIX, help="gan_bpm's runs")
    p.add_argument("--runs", type=Path, default=M.RUNS)
    p.add_argument("--epoch", type=int, default=200)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--case-limit", type=int, help="the first N held-out cases of each fold (a smoke run)")
    p.add_argument("--out", type=Path, help=f"default: {OUT}/<date_time>_<target>/")
    args = p.parse_args(argv)
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    out = args.out or OUT / f"{stamp}_{args.target}{'_smoke' if args.case_limit else ''}"
    say = Report()
    try:
        H, S = M.load_gan_bpm(args.gan_bpm)
    except M.Blocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    from evaluation.aggregate import bootstrap_interval                  # noqa: PLC0415
    from evaluation.data import CaseLoader, TrainingFold                 # noqa: PLC0415
    from evaluation.metrics import LESION_REGIONS, _region               # noqa: PLC0415
    from evaluation.predictors import CheckpointPredictor, predict_case  # noqa: PLC0415
    REGION.update(LESION_REGIONS)
    target = args.target
    _tag, side, cut = REGION[target]

    try:
        folds = args.fold or list(M.FOLDS)
        # {fold: {map name: test info}}; an empty dict scores gan_bpm (and --bran) alone on that fold
        if args.gan_bpm_only:
            tests = {k: {} for k in folds}
        elif args.test_folder:
            tests = {}
            for i in (read_cv_test(f, args.epoch) for f in args.test_folder):
                if map_name(i["variant"]) in tests.setdefault(i["fold"], {}):
                    raise M.Blocked(f"fold {i['fold']}: two test folders of {map_name(i['variant'])}")
                tests[i["fold"]][map_name(i["variant"])] = i
        else:
            tests = {}
            for variant in [None] + args.variant:
                found = find_cv_tests(args.runs, target, folds, args.epoch, variant)
                if variant is not None and not found:
                    raise M.Blocked(f"no complete {target} CV test of {variant} under {args.runs} for folds {folds}")
                for k, i in found.items():
                    tests.setdefault(k, {})[map_name(variant)] = i
        if any(i["target"] != target for t in tests.values() for i in t.values()):
            raise M.Blocked(f"a test folder isn't {target}")
        if not tests:
            raise M.Blocked(f"no complete {target} CV test under {args.runs} for folds {folds} "
                            "(scripts/ctp_only_isles18_dag.py --testset cv), or --gan-bpm-only")
        brans = {}
        for b in args.bran:
            name = f"bran_{b.resolve().name}"
            if name in brans:
                raise M.Blocked(f"{b} is given twice")
            brans[name] = b.resolve()
        recipes = list(dict.fromkeys(["adv0"] + (args.recipe or [])))      # adv0 first: MRDPM's predictor
        run_dirs, configs = {}, {}
        for r in recipes:
            run_dirs[r] = args.matrix / M.gan_run(target, r)
            configs[r] = H.read_config(run_dirs[r])
            found = H.recipe_of(json.loads((run_dirs[r] / "config.json").read_text()))
            if (configs[r].target, found) != (target, r):
                raise M.Blocked(f"{run_dirs[r].name}: config says {configs[r].target}/{found}, not {target}/{r}")
        convention = lambda c: (c.target, c.label_transform, c.tmax_ceiling, c.mask_cache,  # noqa: E731
                                c.min_label_fg_frac, tuple(c.qc_drop_slice_idx))
        odd = [run_dirs[r].name for r in recipes if convention(configs[r]) != convention(configs["adv0"])]
        extra = {}                                       # --run-dir: name -> (folder, config)
        for rd in args.run_dir:
            rd = rd.resolve()
            if rd.name in extra or rd.name in GAN_NAME.values():
                raise M.Blocked(f"{rd.name} is given twice")
            cfg = H.read_config(rd)
            if cfg.target != target:
                raise M.Blocked(f"{rd.name} is a {cfg.target} run, not {target}")
            if convention(cfg) != convention(configs["adv0"]):
                odd.append(rd.name)
            extra[rd.name] = (rd, cfg)
        if odd:
            raise M.Blocked(f"{odd} load the data another way than {run_dirs['adv0'].name}")
        loader = CaseLoader.from_config(configs["adv0"], data_root=args.cv_root)
        scale = loader.label_scale
    except (M.Blocked, S.Blocked) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    say(f"core check {stamp}, {target} ({side} {cut:g} {scale.unit}): MRDPM at "
        f"{M.git_commit(M.REPO) or '<no git>'}, gan_bpm {args.gan_bpm} at {M.git_commit(args.gan_bpm) or '<no git>'}")
    say(f"CV data {args.cv_root}; gan_bpm's fold models (best.pt): "
        f"{', '.join(f'{GAN_NAME[r]} = {run_dirs[r].name}' for r in recipes)}; folds {sorted(tests)}"
        + ("; gan_bpm only, no MRDPM map" if args.gan_bpm_only else "") + f"; out {out}")
    if extra:
        say(f"other runs ({args.run_checkpoint}): {', '.join(extra)}")
    for k, t in sorted(tests.items()):
        if t:
            say(f"fold {k} MRDPM tests: " + ", ".join(f"{n} = {i['folder'].name}" for n, i in t.items()))
    if brans:
        say("pretrained BRANs alone (their held-out maps): " + ", ".join(f"{n} = {d}" for n, d in brans.items()))

    phys = lambda norm: np.asarray(scale.to_physical(np.clip(norm, -1, 1).astype(np.float64)))  # noqa: E731
    bases, rows, notes = [], [], []
    try:
        for k, fold_tests in sorted(tests.items()):
            splits = {r: json.loads((run_dirs[r] / f"fold_{k}" / "split.json").read_text()) for r in recipes}
            held = sorted(map(str, splits["adv0"]["val_case_ids"]))
            if any(sorted(map(str, s["val_case_ids"])) != held for s in splits.values()):
                raise M.Blocked(f"fold {k}: the recipes hold out different cases")
            train = set(map(str, splits["adv0"]["train_case_ids"]))
            predictors = {}
            for r in recipes:
                pr = CheckpointPredictor(run_dirs[r], name=f"{run_dirs[r].name}/fold{k}", checkpoint="best.pt",
                                         batch_size=args.batch_size, device=args.device)
                pr.fit(TrainingFold(k, sorted(map(str, splits[r]["train_case_ids"])), loader))
                predictors[GAN_NAME[r]] = pr
            for name, (rd, _cfg) in extra.items():
                if not (rd / f"fold_{k}" / "checkpoints" / args.run_checkpoint).is_file():
                    notes.append(f"fold {k}: {name} has no {args.run_checkpoint}, so it isn't scored on this fold")
                    continue
                split_r = json.loads((rd / f"fold_{k}" / "split.json").read_text())
                if sorted(map(str, split_r["val_case_ids"])) != held:
                    raise M.Blocked(f"fold {k}: {name} holds out other cases than {run_dirs['adv0'].name}")
                pr = CheckpointPredictor(rd, name=f"{name}/fold{k}", checkpoint=args.run_checkpoint,
                                         batch_size=args.batch_size, device=args.device)
                pr.fit(TrainingFold(k, sorted(map(str, split_r["train_case_ids"])), loader))
                predictors[name] = pr
            # every saved map of this fold: MRDPM tests ({case: {slice: Out_ path}}) and pretrained
            # BRANs ({case: {slice: Init_ path}}), all scored on the slices every one of them has
            haves = {}
            for name, info in fold_tests.items():
                have = M.slices_in(info["results"])
                if train & set(have):
                    raise M.Blocked(f"fold {k}: {info['folder'].name} holds training cases {sorted(train & set(have))[:3]}")
                if set(have) - set(held):
                    raise M.Blocked(f"fold {k}: {info['folder'].name} holds cases outside the fold's held-out set")
                haves[name] = have
            for name, d in brans.items():
                have = read_bran(d, k, target, held, train, args.cv_root, args.allow_debug_bran)
                if have is None:
                    notes.append(f"fold {k}: {name} has no pretrained fold_{k}, so it isn't scored on this fold")
                    continue
                haves[name] = have
            ids = [c for c in held if all(c in h for h in haves.values())]
            if haves and len(ids) < len(held):
                lacking = [n for n, h in haves.items() if any(c not in h for c in held)]
                notes.append(f"fold {k}: {len(held) - len(ids)} held-out cases lack a map of "
                             f"{', '.join(lacking)} and are left out (e.g. {[c for c in held if c not in ids][:3]})")
            ids = ids[:args.case_limit]
            say(f"fold {k}: {len(ids)} held-out cases; " + (", ".join(haves) if haves else "gan_bpm only") + " ...")
            n_init = {}
            for cid in ids:
                case = loader.load(cid)
                maps = {}
                if haves:
                    common = set.intersection(*(set(h[cid]) for h in haves.values()))
                    keep = np.array([s in common for s in case.slice_idx], dtype=bool)
                    if not keep.all():
                        notes.append(f"fold {k} {cid}: {int((~keep).sum())} of {case.n_slices} slices without every map")
                    if not keep.any():
                        continue
                    for name, have in haves.items():
                        sub = {s: q for s, q in have[cid].items() if s in common}
                        if name in brans:
                            init_maps = np.stack([saved_slice(case, j, s, sub[s].parent, sub[s].name[len("Init_"):], "Init_")
                                                  for j, s in enumerate(case.slice_idx) if keep[j]])
                            maps[name] = phys(2 * init_maps - 1)
                            continue
                        _keep, out_maps, init_maps = mrdpm_slices(fold_tests[name], case, sub)
                        maps[name] = phys(2 * out_maps - 1)
                        if init_maps is not None:
                            maps[f"{name}_init"] = phys(2 * init_maps - 1)
                            n_init[name] = n_init.get(name, 0) + 1
                    case = case.select_slices(keep)
                for name, pr in predictors.items():
                    maps[name] = phys(predict_case(pr, case.without_labels()))
                label = case.labels.astype(np.float64)
                maps["label"] = np.asarray(scale.to_physical(label))
                if case.n_slices > 1:
                    maps["label_rolled"] = np.roll(maps["label"], 1, axis=0)
                base, scan = scan_rows(case, maps, scale, target, _region, k)
                bases.append(base)
                rows += scan
            for name, n in n_init.items():
                if n != len(ids):
                    notes.append(f"fold {k}: {name}: Init_ for {n} of {len(ids)} cases")
    except (M.Blocked, S.Blocked) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_scan.csv", index=False)
    no_core = sum(b["k"] == 0 for b in bases)
    for n in notes:
        say(f"note: {n}")
    say(f"{len(bases)} scans scored; {no_core} have no {target} core in the label and are left out "
        f"(they have nothing to find)")

    # the controls, before any model number
    checks = []
    lab = df[df["map"] == "label"]
    checks.append(("the label as a map: matched and cutoff Dice 1 on every scan",
                   f"min {fmt(lab[['matched', 'at_cut']].min().min(), 6)}",
                   bool(len(lab)) and bool((lab[["matched", "at_cut"]] == 1.0).all().all())))
    off = df[df["map"] == "label_rolled"]
    checks.append((f"negative control, the label one slice off: mean matched Dice < {LABEL_OFF_MAX}",
                   f"{fmt(off['matched'].mean())} on {len(off)} scans",
                   bool(len(off)) and off["matched"].mean() < LABEL_OFF_MAX))
    rnd = df[df["map"] == "random"]
    se = rnd["matched"].std(ddof=1) / math.sqrt(len(rnd)) if len(rnd) > 1 else math.nan
    gap = abs(rnd["matched"].mean() - rnd["floor"].mean())
    checks.append(("k random voxels land on the analytic floor k / N (mean over scans, ≤ max(0.005, 4 SE))",
                   f"simulated {fmt(rnd['matched'].mean(), 4)}, floor {fmt(rnd['floor'].mean(), 4)}",
                   bool(len(rnd)) and gap <= max(0.005, 4 * se)))
    say("\n## Checks")
    for name, got, ok in checks:
        say(f"{'PASS' if ok else 'FAIL'}: {name}: {got}")
    pd.DataFrame(checks, columns=["check", "got", "pass"]).to_csv(out / "checks.csv", index=False)
    if not all(ok for *_x, ok in checks):
        say("STOPPED: a control failed; no model number is reported")
        (out / "summary.txt").write_text("\n".join(say.lines) + "\n")
        return 2

    present = set(df["map"])
    arm_maps = [n for v in args.variant for n in (map_name(v), f"{map_name(v)}_init") if n in present]
    maps = ([m for m in SHOWN if m in present] + arm_maps + [m for m in brans if m in present]
            + [m for m in extra if m in present])
    say(f"\n## {target}: how well each map finds the label's core ({side} {cut:g} {scale.unit})")
    say("unit: one scan (a held-out CV case), mean over scans; folds pooled. matched = the map's k "
        "most extreme voxels, k = the label's core size; floor = k random voxels; place score = "
        "(matched − floor) / (1 − floor)")
    say.table("Each map, all folds", summary_table(df, maps, target))
    say.table("Matched Dice by fold", per_fold_table(df, maps, "matched"))
    say.table(f"Dice at the cutoff ({cut:g}) by fold", per_fold_table(df, maps, "at_cut"))
    say.table(f"Cutoff sweep: mean Dice with the label's core when the map is cut at t ({scale.unit})",
              sweep_table(df, maps, target))
    say.table("By the size of the label's core", size_table(df, maps))
    say("\n### Paired, per scan")
    pairs = [("mrdpm", "gan_bpm"), ("mrdpm", "mrdpm_init"), ("mrdpm_init", "gan_bpm")]
    for v in args.variant:
        a = map_name(v)
        pairs += [(a, "mrdpm"), (a, "gan_bpm"), (a, f"{a}_init"), (f"{a}_init", "gan_bpm")]
    pairs += [(b, "gan_bpm") for b in brans]
    for a, b in pairs:
        if a in maps and b in maps:
            for column, what in (("matched", "matched Dice"), ("at_cut", "Dice at the cutoff")):
                say(f"{a} − {b}, {what}: {paired(df, a, b, column, bootstrap_interval)}")
    scored = [m for m in extra if m in maps]
    if len(scored) >= 2:
        control, others = scored[0], scored[1:]
        for b in others:
            for column, what in (("matched", "matched Dice"), ("at_cut", "Dice at the cutoff"),
                                 ("mae_brain", "brain MAE")):
                say(f"{b} − {control}, {what}: {paired(df, b, control, column, bootstrap_interval)}")
        table, chosen = pick(df, control, others, bootstrap_interval)
        say.table(f"The pick (gan_bpm ext_gan_plan.md §1.4, pre-declared): the highest Dice at the "
                  f"cutoff whose paired difference from {control} has a 95 % CI above 0, with brain "
                  "MAE at most 1.15 × the control's", table)
        say(f"PICK: {chosen}" if chosen else
            f"PICK: none: no run beats {control} by the rule, so step 1 is negative and step 2 "
            "starts from the control")
    # bran_ext_plan.md §5.3: the same pre-declared rule among the MRDPM arms and among the BRANs alone
    for label, names in (("MRDPM arms", [map_name(v) for v in args.variant]), ("pretrained BRANs", list(brans))):
        names = [n for n in names if n in maps]
        if len(names) < 2:
            continue
        control, others = names[0], names[1:]
        for b in others:
            for column, what in (("matched", "matched Dice"), ("at_cut", "Dice at the cutoff"),
                                 ("mae_brain", "brain MAE")):
                say(f"{b} − {control}, {what}: {paired(df, b, control, column, bootstrap_interval)}")
        table, chosen = pick(df, control, others, bootstrap_interval)
        say.table(f"The pick among the {label} (bran_ext_plan.md §5.3, pre-declared): the highest Dice at the "
                  f"cutoff whose paired difference from {control} has a 95 % CI above 0, with brain MAE at "
                  "most 1.15 × the control's", table)
        say(f"PICK among the {label}: {chosen}" if chosen else
            f"PICK among the {label}: none: nothing beats {control} by the rule")
    say("\nReading: matched well above 'at the cutoff' = right place, wrong level. Matched near "
        "the floor (place score near 0) = wrong place.")
    (out / "summary.txt").write_text("\n".join(say.lines) + "\n")
    say(f"\nwrote {out}/summary.txt, per_scan.csv and checks.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
