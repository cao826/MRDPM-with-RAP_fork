"""Score the fold models on ISLES 2018 (ctp_only_eval_plan.md §6.1): MRDPM fold k against
gan_bpm fold k on all 94 scans, with gan_bpm's stage-2 code, read-only.

Reads the model folders scripts/ctp_only_isles18_maps.py wrote and stage 1's all-data maps,
scores every one with gan_bpm zeng_isles18_eval/helpers.py's own `score_target` (levels
1-3, after the hole rule) and `score_dwi` (the headline: Dice of the rCBF < 0.3 core with
the DWI lesion), and aggregates with gan_bpm's evaluation/aggregate.py. In order:

 1. Manifests (stage 2's 2a) for every model. A model with a problem is left out, loudly.
 2. The reproduction check, before any MRDPM number: stage 2's per-scan mean DWI Dice of
    the reference (0.238 / 0.255 / 0.163 on parenchyma / valid / envelope) and of the
    all-data models on parenchyma (λ_adv = 0 epoch_<M> 0.036, last 0.099; WGAN-GP last
    0.109). Stops on a mismatch.
 3. Scoring, every model on every scan.
 4. Checks on the result: MRDPM's saved label as a map (mrdpm_gt) scores MAE 0 and the
    reference's DWI Dice on every scan, while its one-slice-off control (mrdpm_gt_rolled)
    doesn't; and the fold-averaged comparison over one fold equals gan_bpm's own
    `comparison_rows` for that fold, exactly.
 5. Tables: each metric per fold, the mean over the folds and their range; the paired
    difference MRDPM − gan_bpm per patient within each fold, averaged over the folds per
    patient, bootstrapped over patients (decisions #24). The fold maps are never averaged.

A fold counts only when MRDPM's map and gan_bpm's are both there; the tables name the folds
used. Writes CSVs and summary.txt (the printed tables) to --out.

    python scripts/ctp_only_isles18_score.py [--target rcbf] [--case-limit 5]
"""
import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctp_only_isles18_maps as M                                       # noqa: E402

PP, SM = "patient_pooled", "slice_mean"
SHOWN = ("mrdpm", "adv0", "lsgan", "wgan_gp")             # report rows, MRDPM first
CHECKS = ("gt", "gt_rolled")
# Stage 2, 2026-10-18 (zeng_isles18_eval/PLAN.md; ctp_only_eval_plan.md §2): per-scan mean Dice
STAGE2_DWI = {("reference", "parenchyma"): 0.238, ("reference", "valid"): 0.255,
              ("reference", "envelope"): 0.163, (("adv0", "M"), "parenchyma"): 0.036,
              (("adv0", "last"), "parenchyma"): 0.099, (("wgan_gp", "last"), "parenchyma"): 0.109}
STAGE2_TOL = 0.0005                                       # they were reported to 3 decimals
GT_MAE_TOL = 1e-4                                         # 2·GT_ − 1 is the label up to float32
GT_DICE_TOL = 1e-3


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
# Models
# ===========================================================================

def fold_models(H, args, targets):
    """MapModels for MRDPM's and gan_bpm's fold maps and the GT checks (no manifest: left out)."""
    models, missing = [], []

    def add(run, snapshot, role, target, recipe, config):
        d = args.maps / run / snapshot
        m = H.MapModel(run=run, snapshot=snapshot, role=role, target=target, recipe=recipe,
                       model_dir=d, config=config, manifest=H._read_manifest(d))
        (models if m.manifest is not None else missing).append(m)

    for t in targets:
        predictor = H.read_config(args.matrix / M.gan_run(t, "adv0"))   # MRDPM's label convention
        for k in M.FOLDS:
            add(M.mrdpm_run(t), f"fold{k}", "last", t, "mrdpm", predictor)
            for r in M.RECIPES:
                run = M.gan_run(t, r)
                add(run, f"fold{k}", "M", t, r, H.read_config(args.matrix / run))
        for check in CHECKS:
            for d in sorted((args.maps / M.mrdpm_run(t, check)).glob("*/manifest.json")):
                add(M.mrdpm_run(t, check), d.parent.name, "last", t, f"mrdpm_{check}", predictor)
    return models, missing


def folds_used(models, target):
    """Folds where MRDPM and its own predictor, gan_bpm λ_adv = 0, both have maps (LSGAN
    and WGAN-GP are context: a fold missing one shows "—" there)."""
    have = {(m.recipe, m.snapshot) for m in models if m.target == target}
    return [k for k in M.FOLDS if all((r, f"fold{k}") in have for r in ("mrdpm", "adv0"))]


def by_role(models, recipe, role, target="rcbf"):
    found = [m for m in models if (m.recipe, m.role, m.target) == (recipe, role, target)
             and not m.snapshot.startswith("fold")]
    return found[0] if len(found) == 1 else None


# ===========================================================================
# The fold-averaged paired comparison
# ===========================================================================

def fold_comparisons(df, cases, metrics, a, b, folds, subsets, bootstrap, key=None):
    """`aggregate.comparison_rows` for a − b across folds: per patient (or scan) the
    difference within each fold, as comparison_rows takes it (the same per-patient values
    and the same distance-to-perfect rule), averaged over the folds where it is defined,
    then the mean with gan_bpm's bootstrap. `df` is `aggregate._scored_frame` of every
    model; the fold models are `<a>/fold<k>` and `<b>/fold<k>`. Over one fold, with
    key=(that fold's two names), it is comparison_rows' row exactly."""
    from evaluation.aggregate import _in_subset, _rng, _subsets, bootstrap_interval  # noqa: PLC0415

    ka, kb = key or (f"{a}/folds", f"{b}/folds")
    selectors = _subsets(cases, subsets)
    rows = []
    for metric in (m for m in metrics if m.scope != "cohort" and m.better != "none"):
        to_perfect = metric.better.startswith("closer to")
        for reduction in metric.reductions:
            sel = df[(df["metric"] == metric.name) & (df["reduction"] == reduction)]
            for subset, selector in selectors.items():
                part = _in_subset(sel, selector)
                per_fold = []
                for k in folds:
                    e = part[part["model"] == f"{a}/fold{k}"].set_index("case_id")["value"]
                    f = part[part["model"] == f"{b}/fold{k}"].set_index("case_id")["value"]
                    both = e.index.intersection(f.index)
                    ev, bv = e.loc[both].astype(float), f.loc[both].astype(float)
                    if to_perfect:
                        ev, bv = (ev - metric.perfect).abs(), (bv - metric.perfect).abs()
                    per_fold.append((ev - bv).rename(k))
                diff = (pd.concat(per_fold, axis=1).mean(axis=1, skipna=True).dropna()
                        if per_fold else pd.Series(dtype=float))
                values = diff.to_numpy(dtype=np.float64)
                better = values > 0 if metric.better == "higher" else values < 0
                lo, hi = bootstrap_interval(values, np.mean, n=bootstrap["n"], ci=bootstrap["ci"],
                                            rng=_rng(bootstrap["seed"],
                                                     f"compare|{ka}|{kb}|{metric.name}|{reduction}|{subset}"))
                rows.append({"model_a": ka, "model_b": kb, "metric": metric.name,
                             "reduction": reduction, "subset": subset,
                             "compared_on": "distance_to_perfect" if to_perfect else "value",
                             "folds": " ".join(map(str, folds)), "n_cases": int(values.size),
                             "mean_diff": float(values.mean()) if values.size else math.nan,
                             "ci_lo": lo, "ci_hi": hi,
                             "frac_cases_better": float(better.mean()) if values.size else math.nan})
    return rows


def check_fold_comparisons(df, per_case, cases, metrics, folds, subsets, bootstrap):
    """Over one fold, fold_comparisons on `df` = gan_bpm's comparison_rows on the raw
    per-scan rows, in every column. Negative control: another fold's row differs."""
    from evaluation.aggregate import comparison_rows                    # noqa: PLC0415

    cols = ["n_cases", "mean_diff", "ci_lo", "ci_hi", "frac_cases_better"]
    keys = ["metric", "reduction", "subset"]
    models = [SimpleNamespace(name=n) for n in sorted(set(df["model"]))]
    k = folds[0]
    a, b = f"mrdpm/fold{k}", f"adv0/fold{k}"
    theirs = pd.DataFrame(comparison_rows(per_case, cases, models, metrics, bootstrap, [(a, b)], subsets))
    mine = pd.DataFrame(fold_comparisons(df, cases, metrics, "mrdpm", "adv0", [k], subsets,
                                         bootstrap, key=(a, b)))
    joined = mine.merge(theirs, on=keys, suffixes=("_mine", "_theirs"))
    same = len(joined) == len(theirs) == len(mine) and all(
        np.allclose(joined[f"{c}_mine"].astype(float), joined[f"{c}_theirs"].astype(float),
                    rtol=0, atol=0, equal_nan=True) for c in cols)
    control = None
    if len(folds) > 1:
        j = folds[1]
        other = pd.DataFrame(comparison_rows(per_case, cases, models, metrics, bootstrap,
                                             [(f"mrdpm/fold{j}", f"adv0/fold{j}")], subsets))
        o = mine.merge(other, on=keys, suffixes=("_mine", "_other"))
        control = not np.allclose(o["mean_diff_mine"], o["mean_diff_other"], equal_nan=True)
    return same, control, k


# ===========================================================================
# Tables
# ===========================================================================

def fold_table(cohort, metric, reduction, folds, digits=3):
    """Rows = MRDPM and the gan_bpm recipes; columns = each fold's cohort mean over
    patients (or scans), their mean over the folds and the range."""
    sel = cohort[(cohort["subset"] == "all") & (cohort["statistic"] == "mean")
                 & (cohort["metric"] == metric) & (cohort["reduction"] == reduction)].set_index("model")
    rows = {}
    for r in SHOWN:
        vals = [float(sel.loc[f"{r}/fold{k}", "value"]) if f"{r}/fold{k}" in sel.index else math.nan
                for k in folds]
        row = {f"fold {k}": fmt(v, digits) for k, v in zip(folds, vals)}
        ok = [v for v in vals if not math.isnan(v)]
        row["mean over folds"] = fmt(float(np.mean(ok)), digits) if ok else "—"
        row["range"] = f"{fmt(min(ok), digits)}–{fmt(max(ok), digits)}" if ok else "—"
        n = [int(sel.loc[f"{r}/fold{k}", "n"]) for k in folds if f"{r}/fold{k}" in sel.index]
        row["n"] = f"{min(n)}" if n and min(n) == max(n) else (f"{min(n)}–{max(n)}" if n else "—")
        rows[r] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def context_line(cohort, metric, reduction, names, digits=3):
    sel = cohort[(cohort["subset"] == "all") & (cohort["statistic"] == "mean")
                 & (cohort["metric"] == metric) & (cohort["reduction"] == reduction)].set_index("model")
    parts = [f"{n} {fmt(float(sel.loc[n, 'value']), digits)}" for n in names if n in sel.index]
    return "  for reference: " + ", ".join(parts) if parts else ""


def comparison_table(comp, columns, digits=3):
    """Rows = pairs; cells = mean paired difference [95 % CI], share of units where MRDPM wins."""
    sel = comp[comp["subset"] == "all"]
    pairs = list(dict.fromkeys(zip(sel["model_a"], sel["model_b"])))
    out = pd.DataFrame(index=[f"{a} − {b}" for a, b in pairs])
    for metric, reduction in columns:
        part = sel[(sel["metric"] == metric) & (sel["reduction"] == reduction)].set_index(["model_a", "model_b"])
        out[f"{metric} ({'pooled' if reduction == PP else 'slice mean'})"] = [
            f"{fmt(part.loc[p, 'mean_diff'], digits)} [{fmt(part.loc[p, 'ci_lo'], digits)}, "
            f"{fmt(part.loc[p, 'ci_hi'], digits)}] MRDPM better {part.loc[p, 'frac_cases_better']:.0%}"
            if p in part.index else "—" for p in pairs]
    first = sel[(sel["metric"] == columns[0][0]) & (sel["reduction"] == columns[0][1])]
    out["n"] = [int(first.set_index(["model_a", "model_b"]).loc[p, "n_cases"]) if p in
                first.set_index(["model_a", "model_b"]).index else 0 for p in pairs]
    return out


def key_metrics(H, target):
    tag = H.lesion_tag(target)
    return [("MAE_brain", PP, 3), ("bias_brain", PP, 3), ("SSIM_brain", SM, 3),
            ("bias_band_low", PP, 3), ("bias_band_high", PP, 3), ("spread_ratio", PP, 3),
            ("W1", PP, 3), (f"dice_{tag}", PP, 3), (f"vol_true_{tag}", PP, 1),
            (f"vol_pred_{tag}", PP, 1), (f"abs_vol_error_{tag}", PP, 1)]


# ===========================================================================
# Main
# ===========================================================================

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--gan-bpm", type=Path, default=M.GAN_BPM, help="gan_bpm's checkout, read only")
    p.add_argument("--test-root", type=Path, default=M.TEST_ROOT)
    p.add_argument("--matrix", type=Path, default=M.MATRIX)
    p.add_argument("--maps", type=Path, default=M.MAPS)
    p.add_argument("--target", action="append", choices=M.TARGETS, help="default: both")
    p.add_argument("--case-limit", type=int, help="score only the first N scans (a smoke run; "
                   "the reproduction check is then skipped)")
    p.add_argument("--out", type=Path, help="default: <isles2018>/eval/<date_time>/")
    args = p.parse_args(argv)
    targets = tuple(args.target or M.TARGETS)
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    out = args.out or M.ISLES / "eval" / (stamp + ("_smoke" if args.case_limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    say = Report()

    H, S = M.load_gan_bpm(args.gan_bpm)
    from evaluation.aggregate import _scored_frame                       # noqa: PLC0415
    from evaluation.spec import DEFAULT_BOOTSTRAP as BOOT                # noqa: PLC0415

    say(f"ISLES 2018 fold-model scoring {stamp}: gan_bpm {args.gan_bpm} at "
        f"{M.git_commit(args.gan_bpm) or '<no git>'}, MRDPM at {M.git_commit(M.REPO) or '<no git>'}")
    say(f"test root {args.test_root}, DWI {H.DWI_DIR}, maps {args.maps}, out {out}")

    # 1. models and manifests
    runs = H.load_runs(H.RUNS, args.matrix)
    alldata = [m for m in H.find_models(runs, H.EXTERNAL_DIR) if m.target in targets]
    folds, missing = fold_models(H, args, targets)
    models = alldata + folds
    all_ids = {id(m) for m in alldata}
    is_all = lambda m: id(m) in all_ids                                  # noqa: E731
    manifests = H.check_manifests(models, args.test_root)
    manifests.to_csv(out / "manifests.csv", index=False)
    bad = manifests[~manifests["ok"]]
    for r in bad.itertuples():
        say(f"LEFT OUT {r.model} ({r.target}): {r.problems}")
    bad_keys = set(zip(bad["model"], bad["target"]))
    models = [m for m in models if (m.name, m.target) not in bad_keys and m.manifest is not None]
    used = {t: folds_used(models, t) for t in targets}
    for t in targets:
        say(f"{t}: {sum(m.target == t for m in models)} models with maps; folds with MRDPM and "
            f"gan_bpm λ_adv = 0: {used[t] or 'none'}; maps not written yet: "
            f"{sorted(m.name for m in missing if m.target == t) or 'none'}")
    indexes = {t: H.read_test_index(args.test_root, t) for t in M.TARGETS}
    ids = None if args.case_limit is None else sorted(H.expected_slices(indexes["rcbf"]))[:args.case_limit]
    old_scans, _notes = H.first_run_cases([m for m in models if is_all(m)], H.FIRST_RUN_DIR)
    subsets = H.named_subsets(indexes["rcbf"], old_scans)
    checks = []

    # 2. reproduce stage 2 before any MRDPM number
    rcbf_all = [m for m in models if is_all(m) and m.target == "rcbf"]
    dwi0 = None
    if "rcbf" in targets:
        dwi0 = H.score_dwi(rcbf_all, args.test_root, H.DWI_DIR, case_ids=ids, log=say)
        c0 = H.dwi_cohort_frame(dwi0, rcbf_all, {}, per_patient=False)
        c0 = c0[(c0["statistic"] == "mean") & (c0["subset"] == "all")].set_index(["model", "metric"])
        for (who, variant), want in STAGE2_DWI.items():
            m = who if who == H.REFERENCE else by_role(rcbf_all, *who)
            name = m if isinstance(m, str) else (m.name if m else f"{who[0]}/{who[1]} (missing)")
            got = float(c0.loc[(name, f"dwi_dice_{variant}"), "value"]) if (name, f"dwi_dice_{variant}") in c0.index else math.nan
            ok = abs(got - want) <= STAGE2_TOL
            checks.append({"check": f"stage 2: {name} DWI Dice, {variant}, per-scan mean", "want": want,
                           "got": got, "pass": bool(ok) if ids is None else None})
        failed = [c for c in checks if c["pass"] is False]
        for c in checks:
            say(f"{'PASS' if c['pass'] else ('FAIL' if c['pass'] is False else 'skipped (case limit)')}: "
                f"{c['check']}: {fmt(c['got'])} (stage 2 {c['want']:.3f})")
        if failed:
            pd.DataFrame(checks).to_csv(out / "checks.csv", index=False)
            say("STOPPED: the scoring does not reproduce stage 2, so no MRDPM number is reported")
            return 2

    # 3. score everything
    scores, dwi = {}, None
    for t in targets:
        scores[t] = H.score_target(t, models, args.test_root, case_ids=ids, log=say)
    if "rcbf" in targets:
        rest = [m for m in models if m.target == "rcbf" and not is_all(m)]
        dwi1 = H.score_dwi(rest, args.test_root, H.DWI_DIR, case_ids=ids, log=say) if rest else None
        dwi = pd.concat([dwi0] + ([dwi1[dwi1["model"] != H.REFERENCE]] if dwi1 is not None else []),
                        ignore_index=True)
        dwi.to_csv(out / "dwi_per_scan.csv", index=False)

    # 4. checks on the result
    for t, s in scores.items():
        pc = pd.DataFrame(s.per_case)
        mae = pc[(pc["metric"] == "MAE_brain") & (pc["reduction"] == PP)]
        for check, want_zero in (("gt", True), ("gt_rolled", False)):
            v = mae[mae["model"].str.startswith(f"mrdpm_{check}/")]["value"].dropna()
            if v.empty:
                checks.append({"check": f"{t}: mrdpm_{check} MAE_brain", "pass": None, "got": math.nan,
                               "want": "no map yet"})
                continue
            got = float(v.abs().max()) if want_zero else float(v.median())
            ok = got <= GT_MAE_TOL if want_zero else got > 0.01
            checks.append({"check": f"{t}: mrdpm_{check} MAE_brain " + ("max over scans" if want_zero
                           else "median (negative control)"), "want": f"≤ {GT_MAE_TOL:g}" if want_zero
                           else "> 0.01", "got": got, "pass": bool(ok)})
    if dwi is not None:
        ref = dwi[dwi["model"] == H.REFERENCE].set_index(["variant", "case_id"])["dice"]
        for check, same in (("gt", True), ("gt_rolled", False)):
            g = dwi[dwi["model"].str.startswith(f"mrdpm_{check}/")].set_index(["variant", "case_id"])["dice"]
            if g.empty:
                continue
            d = (g - ref.reindex(g.index)).abs()
            both_nan = g.isna() & ref.reindex(g.index).isna()
            d = d[~both_nan]
            got = float(d.max()) if same else float(d.mean())
            ok = (got <= GT_DICE_TOL and not d.isna().any()) if same else got > 0.01
            checks.append({"check": f"rcbf: mrdpm_{check} DWI Dice − reference's, " +
                           ("max over scans and regions" if same else "mean (negative control)"),
                           "want": f"≤ {GT_DICE_TOL:g}" if same else "> 0.01", "got": got, "pass": bool(ok)})
    frames, cohorts, comps = {}, {}, {}
    for t, s in scores.items():
        df = _scored_frame(s.per_case, s.cases, s.metrics)
        frames[t] = df
        if used[t]:
            same, control, k = check_fold_comparisons(df, s.per_case, s.cases, s.metrics, used[t],
                                                      subsets, BOOT)
            checks.append({"check": f"{t}: fold-averaged comparison over fold {k} = gan_bpm's comparison_rows",
                           "want": "equal", "got": same, "pass": bool(same)})
            if control is not None:
                checks.append({"check": f"{t}: negative control, it differs from another fold's",
                               "want": "differs", "got": control, "pass": bool(control)})
    say("\n## Checks")
    for c in checks[len(STAGE2_DWI) if "rcbf" in targets else 0:]:
        say(f"{'PASS' if c['pass'] else ('FAIL' if c['pass'] is False else 'n/a')}: {c['check']}: "
            f"{c['got'] if not isinstance(c['got'], float) else fmt(c['got'], 6)} (want {c['want']})")
    pd.DataFrame(checks).to_csv(out / "checks.csv", index=False)
    if any(c["pass"] is False for c in checks):
        say("STOPPED: a check failed; the tables below would not be trustworthy")
        return 2

    # 5. tables
    for t, s in scores.items():
        frames[t].to_csv(out / f"{t}_per_patient.csv", index=False)
        pd.DataFrame(s.per_case).to_csv(out / f"{t}_per_case.csv", index=False)
        s.holes.to_csv(out / f"{t}_holes.csv", index=False)
        cohorts[t] = H.cohort_frame(s, subsets)
        cohorts[t].to_csv(out / f"{t}_cohort.csv", index=False)
        rows = []
        for r in SHOWN[1:]:
            rows += fold_comparisons(frames[t], s.cases, s.metrics, "mrdpm", r, used[t], subsets, BOOT)
        comps[t] = pd.DataFrame(rows)
        comps[t].to_csv(out / f"{t}_fold_comparisons.csv", index=False)

    context = [H.REFERENCE] + [m.name for m in models if is_all(m) and m.target == "rcbf"]
    if dwi is not None and used["rcbf"]:
        say("\n## The headline: Dice of the rCBF < 0.3 core with the DWI lesion (gan_bpm 2f)")
        say(f"folds {used['rcbf']}; mean over scans (Zeng et al.'s unit), then over patients; "
            f"regions: {', '.join(H.DWI_VARIANTS)} (parenchyma is the headline)")
        rcbf_models = [m for m in models if m.target == "rcbf"]
        for unit, per_patient in (("per scan", False), ("per patient", True)):
            c = H.dwi_cohort_frame(dwi, rcbf_models, subsets, per_patient=per_patient)
            c.to_csv(out / f"dwi_cohort_{unit.replace(' ', '_')}.csv", index=False)
            for v in H.DWI_VARIANTS:
                say.table(f"DWI Dice, {v}, {unit}", fold_table(c, f"dwi_dice_{v}", PP, used["rcbf"]))
                say(context_line(c, f"dwi_dice_{v}", PP, context))
            rows, cases = H._dwi_rows(dwi, per_patient)
            dfd = _scored_frame(rows, cases, H.dwi_metrics())
            named = H.patient_subsets(subsets, dwi) if per_patient else subsets
            comp = []
            for r in SHOWN[1:]:
                comp += fold_comparisons(dfd, cases, H.dwi_metrics(), "mrdpm", r, used["rcbf"], named, BOOT)
            comp = pd.DataFrame(comp)
            comp.to_csv(out / f"dwi_fold_comparisons_{unit.replace(' ', '_')}.csv", index=False)
            say.table(f"DWI Dice, MRDPM − gan_bpm, {unit}, averaged over folds {used['rcbf']} "
                      "(positive: MRDPM better)",
                      comparison_table(comp, [(f"dwi_dice_{v}", PP) for v in H.DWI_VARIANTS]))
        vols = H.dwi_volumes(dwi)
        vols.to_csv(out / "dwi_volumes.csv")
        keep = [i for i in vols.index if not i[1].startswith("mrdpm_gt")]
        say.table("Median DWI and core volume (mL) and the lesion share out of reach, per region",
                  vols.loc[keep].round(3))

    for t, s in scores.items():
        if not used[t]:
            say(f"\n## {H.TARGET_LABEL[t]}: no fold has both MRDPM's and gan_bpm's maps yet")
            continue
        say(f"\n## {H.TARGET_LABEL[t]} ({s.scale.unit}): levels 1-3, folds {used[t]}, mean over patients")
        names = [m.name for m in models if is_all(m) and m.target == t]
        for metric, reduction, digits in key_metrics(H, t):
            say.table(f"{metric} ({'pooled' if reduction == PP else 'slice mean'})",
                      fold_table(cohorts[t], metric, reduction, used[t], digits))
            say(context_line(cohorts[t], metric, reduction, names, digits))
        say.table(f"{H.TARGET_LABEL[t]}: MRDPM − gan_bpm per patient, averaged over folds {used[t]} "
                  "(MAE, |bias|, |volume error|: negative = MRDPM better; Dice, SSIM: positive)",
                  comparison_table(comps[t], [(m, r) for m, r, _d in key_metrics(H, t)
                                              if not m.startswith(("vol_true", "vol_pred"))]).T)

    (out / "summary.txt").write_text("\n".join(say.lines) + "\n")
    say(f"\nwrote {out}/summary.txt and the CSVs beside it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
