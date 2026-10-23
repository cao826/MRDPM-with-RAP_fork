"""ISLES 2018 inputs and maps for scoring the fold models (ctp_only_eval_plan.md §6.1, §7).

Runs on 14958 (it reads that box's /raid), through gan_bpm's own code, imported read-only
from --gan-bpm (nothing in gan_bpm changes). Three subcommands:

  split    the split.json of every ISLES 2018 scan, as val_case_ids: what the test job's
           `testset = isles18` and the all-data configs read. Refuses unless both label
           indexes hold the same scans and the counts are the build's (94 scans, 63
           patients, 502 slices).
  gan-bpm  gan_bpm's fold models (each fold's best.pt; λ_adv = 0, LSGAN, WGAN-GP; both
           targets) on every ISLES 2018 scan, through gan_bpm's CaseLoader and
           CheckpointPredictor. Needs a GPU (ctp_only_isles18_ganbpm_maps.sub). First a
           check: the same loader and writer with stage 1's all-data λ_adv = 0 `last`
           must give stage 1's stored maps.
  mrdpm    MRDPM's ISLES 2018 test folders (test_ctp_only_f13_<target>_fold<k>_isles18_<time>)
           as maps: pred_norm = 2·Out_ − 1, clipped to [−1, 1] (decisions #25). Refuses a
           missing slice, a Mask_ that isn't the scan's envelope, a GT_ that isn't the
           scan's label, or a checkpoint that isn't the training run's.

Every model is a folder <maps>/<run>/<snapshot>/ in stage 1's format
(gan_bpm scripts/predict_external_maps.py): maps/<case_id>.npz with pred_norm, pred_phys,
envelope, ext_gate, slice_idx, case_id, patient_id, and manifest.json written last. So
gan_bpm's stage-2 code scores gan_bpm's and MRDPM's maps the same way
(scripts/ctp_only_isles18_score.py):

  gan_bpm_<target>_..._<recipe>_k5/fold<k>/   gan_bpm fold k, best.pt
  mrdpm_ctp_only_f13_<target>/fold<k>/        MRDPM fold k, epoch 200, EMA weights, 1 sample
  mrdpm_ctp_only_f13_<target>_gt/fold<k>/     check: MRDPM's saved label (GT_) as a map;
                                              it must score exactly as the reference
  mrdpm_ctp_only_f13_<target>_gt_rolled/...   its negative control: GT_ one slice off

The _gt models are written once per target, from the first test folder converted.

    python scripts/ctp_only_isles18_maps.py split
    python scripts/ctp_only_isles18_maps.py gan-bpm [--target rcbf] [--recipe adv0] [--fold 0]
    python scripts/ctp_only_isles18_maps.py mrdpm [TEST_FOLDER ...]   # default: every complete one

A folder that has its manifest is skipped unless --overwrite. Exit codes: 0 done,
2 refused (nothing written for what was refused).
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GAN_BPM = Path("/raid/colivares/github_repos/gan_bpm")
TEST_ROOT = Path("/raid/colivares/mctp_dataset_msu_f13_isles2018_all94")
MATRIX = Path("/radraid2/colivares/matrix")
RUNS = Path("/radraid2/colivares/mrdpm_ctp_only/runs")
ISLES = Path("/radraid2/colivares/mrdpm_ctp_only/isles2018")
MAPS = ISLES / "maps"

DATASET = "isles2018"
EXPECTED = {"scans": 94, "patients": 63, "slices": 502}
TARGETS = ("rcbf", "tmax")
RECIPES = ("adv0", "lsgan", "wgan_gp")
FOLDS = (0, 1, 2, 3, 4)
RUN_PATTERN = {"rcbf": "gan_bpm_rcbf_f13uclav2_{}", "tmax": "gan_bpm_tmax_f13uclav2_t12_{}"}
RECIPE_DIR = {"adv0": "adv0", "lsgan": "lsgan", "wgan_gp": "wgan"}
TEST_NAME = re.compile(r"^test_ctp_only_f13_(tmax|rcbf)_(fold[0-4]|all)_isles18_\d{6}_\d{6}$")
TRAIN_NAME = re.compile(r"^train_ctp_only_f13_(tmax|rcbf)_(fold[0-4]|all)_\d{6}_\d{6}$")
SLICE_FILE = re.compile(r"^Out_(.+)_slice(\d+)\.npy$")
GT_TOL = 1e-5            # |2·GT_ − 1 − label| on the envelope: float32 rounding only
CHECK_TOL = 1e-3         # stage 1's maps against the same weights rerun (GPU arithmetic)
CHECK_SCANS = 5


class Blocked(RuntimeError):
    """A guard refused: the output would be wrong, so nothing is written for it."""


def gan_run(target, recipe, kind="k5"):
    """gan_bpm's run folder name: kind "k5" (the 5 folds) or "all" (the all-data run)."""
    return RUN_PATTERN[target].format(f"{RECIPE_DIR[recipe]}_{kind}")


def mrdpm_run(target, check=""):
    return f"mrdpm_ctp_only_f13_{target}" + (f"_{check}" if check else "")


def load_gan_bpm(root):
    """(helpers, stage 1): gan_bpm's stage-2 helpers (zeng_isles18_eval/helpers.py, which
    puts gan_bpm's root on sys.path) and stage 1's script, imported as stage 2 imports them."""
    root = Path(root)
    if not (root / "zeng_isles18_eval" / "helpers.py").is_file():
        raise Blocked(f"{root} has no zeng_isles18_eval/helpers.py: not a gan_bpm checkout with stage 2")
    sys.path.insert(0, str(root / "zeng_isles18_eval"))
    import helpers as H                                                  # noqa: PLC0415

    return H, H._stage1()


def git_commit(path):
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=path,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_rows(S, test_root, target):
    return S.select_test_rows(S.read_index(test_root, target), DATASET)


def train_ids(split_path):
    ids = json.loads(Path(split_path).read_text()).get("train_case_ids")
    if not ids:
        raise Blocked(f"{split_path} lists no train_case_ids, so leakage cannot be ruled out")
    return [str(c) for c in ids]


def leak_guard(S, config, ids, rows, what):
    """Stage 1's guards: no ISLES 2018 scan or patient among the training cases, the same
    phase protocol as the test set."""
    index = S.read_index(config.data_root, config.target)
    problems = S.leak_problems(ids, index, rows, DATASET)
    problems += S.protocol_problems(index[index["case_id"].isin(set(ids))], rows)
    if problems:
        raise Blocked(f"{what}: " + "; ".join(problems))


def model_done(model_dir):
    return (Path(model_dir) / "manifest.json").is_file()


class MapWriter:
    """One model folder as stage 1's `write_model` writes it, one scan at a time:
    maps/<case_id>.npz first, manifest.json last. A folder without the manifest is an
    interrupted attempt; it is redone."""

    def __init__(self, S, model_dir, scale, tmax_ceiling):
        self.S, self.dir, self.scale, self.ceiling = S, Path(model_dir), scale, tmax_ceiling
        if self.dir.exists():
            shutil.rmtree(self.dir)
        (self.dir / "maps").mkdir(parents=True)
        self.rows = []

    def add(self, case, pred):
        pred = np.asarray(pred, dtype=np.float32)
        if pred.shape != case.frame_shape or not np.isfinite(pred).all() \
                or np.abs(pred).max() > 1.0 + 1e-6:                     # predict_case's tolerance
            raise Blocked(f"{case.case_id}: the map is not a finite {case.frame_shape} array in [-1, 1]")
        rel = f"maps/{case.case_id}.npz"
        patient = str(case.meta.get("patient_id", case.case_id))
        np.savez_compressed(self.dir / rel, pred_norm=pred,
                            pred_phys=self.S.to_physical_map(pred, case.envelope, self.scale),
                            envelope=case.envelope, ext_gate=case.ext_gate,
                            slice_idx=np.asarray(case.slice_idx, dtype=np.int32),
                            case_id=np.array(case.case_id), patient_id=np.array(patient))
        self.rows.append({"case_id": case.case_id, "patient_id": patient,
                          "n_slices": case.n_slices, "file": rel})

    def finish(self, run, snapshot, model, test, gan_bpm_root):
        s = self.scale
        manifest = {
            "written": datetime.now().isoformat(timespec="seconds"),
            "script": "MRDPM-with-RAP_fork scripts/ctp_only_isles18_maps.py",
            "git_commit": git_commit(REPO),
            "gan_bpm_commit": git_commit(gan_bpm_root),
            "run": run,
            "snapshot": snapshot,
            "model": model,
            "label_scale": {"target": s.target, "label_transform": s.kind, "lo": s.lo,
                            "hi": s.hi, "unit": s.unit, "tmax_ceiling": self.ceiling},
            "to_physical": "mctp.label_scale.to_physical(pred_norm, label_transform, lo, hi) "
                           "inside the envelope, NaN outside",
            "test": test,
            "cases": self.rows,
            "arrays": self.S.ARRAYS,
        }
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def abort(self):
        shutil.rmtree(self.dir, ignore_errors=True)


# ===========================================================================
# split
# ===========================================================================

def cmd_split(args, H, S):
    indexes = {t: H.read_test_index(args.test_root, t) for t in TARGETS}
    problems = H.index_problems(indexes)
    ix = indexes["rcbf"]
    counts = {"scans": int(ix["case_id"].nunique()), "patients": int(ix["patient_id"].nunique()),
              "slices": int(len(ix))}
    print(f"{args.test_root}: {counts['scans']} {DATASET} scans, {counts['patients']} patients, "
          f"{counts['slices']} slices (expected {EXPECTED})")
    if problems or counts != EXPECTED:
        raise Blocked("; ".join(problems) or f"counts {counts}, expected {EXPECTED}")
    split = {"train_case_ids": [], "val_case_ids": sorted(set(ix["case_id"])),
             "dataset": DATASET, "test_root": str(args.test_root), **{f"n_{k}": v for k, v in counts.items()},
             "note": "every ISLES 2018 scan as the test part; no model trains on it "
                     "(ctp_only_eval_plan.md §6)"}
    out = Path(args.out)
    if out.is_file():
        old = json.loads(out.read_text())
        if old.get("val_case_ids") == split["val_case_ids"] and not old.get("train_case_ids"):
            print(f"{out} is already this split; left as it is")
            return
        if not args.overwrite:
            raise Blocked(f"{out} exists with other cases; --overwrite to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=1) + "\n")
    print(f"wrote {out}")


# ===========================================================================
# gan-bpm: the fold models
# ===========================================================================

def check_writer(args, H, S, loader, cases):
    """Stage 1's all-data λ_adv = 0 `last` through this loader and MapWriter, on a few
    scans, against stage 1's stored maps: the same pred_norm (up to GPU arithmetic), the
    same envelope and ext_gate. Negative control: its `epoch_<M>` is far from `last`'s maps."""
    from evaluation.data import TrainingFold                             # noqa: PLC0415
    from evaluation.predictors import predict_case                       # noqa: PLC0415

    run = S.inspect_run(args.matrix / gan_run("rcbf", "adv0", "all"))
    if run.state != "finished":
        raise Blocked(f"check: {run.name} is {run.state}")
    stored = H.EXTERNAL_DIR / run.name
    ids = sorted(cases)[:CHECK_SCANS]
    worst = {}
    with tempfile.TemporaryDirectory() as tmp:
        for snapshot in run.snapshots:
            p = S.AllDataCheckpoint(run.run_dir, name=f"{run.name}/{snapshot}",
                                    checkpoint=run.snapshots[snapshot],
                                    batch_size=args.batch_size, device=args.device)
            p.fit(TrainingFold(0, S.read_train_ids(run), loader))
            w = MapWriter(S, Path(tmp) / snapshot, loader.label_scale, run.config.tmax_ceiling)
            for cid in ids:
                w.add(cases[cid], predict_case(p, cases[cid]))
            d, masks_equal = 0.0, True
            for cid in ids:
                with np.load(Path(tmp) / snapshot / "maps" / f"{cid}.npz") as mine, \
                        np.load(stored / "last" / "maps" / f"{cid}.npz") as theirs:
                    d = max(d, float(np.abs(mine["pred_norm"] - theirs["pred_norm"]).max()))
                    masks_equal &= bool(np.array_equal(mine["envelope"], theirs["envelope"])
                                        and np.array_equal(mine["ext_gate"], theirs["ext_gate"])
                                        and np.array_equal(mine["slice_idx"], theirs["slice_idx"]))
            worst[snapshot] = (d, masks_equal)
    m = next(s for s in run.snapshots if s != "last")
    ok = worst["last"][0] <= CHECK_TOL and worst["last"][1]
    control = worst[m][0] > 10 * CHECK_TOL
    print(f"check, {len(ids)} scans: stage 1's {run.name}/last rerun here, max |Δ pred_norm| "
          f"{worst['last'][0]:.2e} (≤ {CHECK_TOL:g}), envelope/ext_gate/slices equal "
          f"{worst['last'][1]}: {'PASS' if ok else 'FAIL'}")
    print(f"  negative control: its {m} against last's stored maps, max |Δ| {worst[m][0]:.3f} "
          f"(> {10 * CHECK_TOL:g}): {'PASS' if control else 'FAIL'}")
    if not (ok and control):
        raise Blocked("the writer check failed; no fold maps written")


def cmd_gan_bpm(args, H, S):
    from evaluation.data import TrainingFold                             # noqa: PLC0415
    from evaluation.predictors import CheckpointPredictor, predict_case  # noqa: PLC0415

    targets, recipes, folds = args.target or TARGETS, args.recipe or RECIPES, args.fold or FOLDS
    todo, have = [], 0
    for target in targets:
        rows = test_rows(S, args.test_root, target)
        for recipe in recipes:
            run_dir = args.matrix / gan_run(target, recipe)
            config = H.read_config(run_dir)
            found = H.recipe_of(json.loads((run_dir / "config.json").read_text()))
            if (config.target, found) != (target, recipe):
                raise Blocked(f"{run_dir.name}: config says {config.target}/{found}, not {target}/{recipe}")
            for k in folds:
                ids = train_ids(run_dir / f"fold_{k}" / "split.json")
                leak_guard(S, config, ids, rows, f"{run_dir.name} fold {k}")
                model_dir = args.maps / run_dir.name / f"fold{k}"
                if model_done(model_dir) and not args.overwrite:
                    have += 1
                    continue
                todo.append((target, recipe, k, run_dir, config, ids, model_dir))
    print(f"gan_bpm fold models: {len(todo)} to write, {have} already there -> {args.maps}")
    if not todo:
        return
    test = None
    for target in targets:
        jobs = [t for t in todo if t[0] == target]
        if not jobs and not (target == "rcbf" and not args.skip_check):
            continue
        config = jobs[0][4] if jobs else H.read_config(args.matrix / gan_run(target, "adv0"))
        convention = lambda c: (c.target, c.label_transform, c.tmax_ceiling, c.mask_cache)  # noqa: E731
        odd = [f"{j[3].name}" for j in jobs if convention(j[4]) != convention(config)]
        if odd:
            raise Blocked(f"{target}: {odd} use another label convention than {jobs[0][3].name}")
        rows = test_rows(S, args.test_root, target)
        loader = S.make_loader(config, args.test_root)
        expected = S._expected_slices(rows)
        cases = {}
        for cid in sorted(expected):
            case = loader.load(cid)
            S.check_slices(case, expected[cid])
            cases[cid] = case.without_labels()
        test = S._test_summary(rows, args.test_root, DATASET)
        if target == "rcbf" and not args.skip_check:
            check_writer(args, H, S, loader, cases)
        for _t, recipe, k, run_dir, cfg, ids, model_dir in jobs:
            if {c.inputs.shape[1] for c in cases.values()} != {cfg.n_phases}:
                raise Blocked(f"{run_dir.name} was trained on {cfg.n_phases} phases")
            print(f"  {run_dir.name}/fold{k}: {test['n_cases']} scans ...", flush=True)
            p = CheckpointPredictor(run_dir, name=f"{run_dir.name}/fold{k}", checkpoint="best.pt",
                                    batch_size=args.batch_size, device=args.device)
            p.fit(TrainingFold(k, ids, loader))
            w = MapWriter(S, model_dir, loader.label_scale, cfg.tmax_ceiling)
            try:
                for case in cases.values():
                    w.add(case, predict_case(p, case))
            except BaseException:
                w.abort()
                raise
            w.finish(run_dir.name, f"fold{k}", {**p.describe(), "recipe": recipe, "fold": k},
                     test, args.gan_bpm)


# ===========================================================================
# mrdpm: the test folders
# ===========================================================================

def read_test_folder(folder, epoch):
    """What a test folder is and where its slices are; refuses one the job didn't finish."""
    folder = Path(folder)
    m = TEST_NAME.match(folder.name)
    if not m:
        raise Blocked(f"{folder.name} is not a full ISLES 2018 test folder "
                      "(test_ctp_only_f13_<target>_<fold>_isles18_<time>)")
    target, fold = m.group(1), m.group(2).replace("fold", "")
    for name in ("job_config.json", "source_run.json"):
        if not (folder / name).is_file():
            raise Blocked(f"{folder.name}: no {name} (the test job writes it)")
    cfg = json.loads((folder / "job_config.json").read_text())
    src = json.loads((folder / "source_run.json").read_text())
    args = cfg["datasets"]["test"]["which_dataset"]["args"]
    problems = []
    if cfg.get("name") != f"ctp_only_f13_{target}_{m.group(2)}_isles18":
        problems.append(f"job config name {cfg.get('name')}")
    if args.get("label_type") != target or not str(args.get("split_json", "")).endswith("isles2018/split.json"):
        problems.append(f"test data {args.get('label_type')} / {args.get('split_json')}")
    if not cfg["model"]["which_model"]["args"].get("test_ema"):
        problems.append("not sampled with the EMA weights")
    if (src.get("testset"), src.get("mode"), int(src.get("epoch", -1))) != ("isles18", "full", epoch):
        problems.append(f"source_run says {src.get('testset')} / {src.get('mode')} / epoch {src.get('epoch')}")
    run = Path(src.get("run", ""))
    t = TRAIN_NAME.match(run.name)
    if not t or (t.group(1), t.group(2)) != (target, m.group(2)):
        problems.append(f"made from {run.name}, not a {target} {m.group(2)} training run")
    ema = run / "checkpoint" / f"{epoch}_Network_ema.pth"
    if not ema.is_file():
        problems.append(f"{ema} is gone")
    elif sha256(ema) != src.get("ema_sha256"):
        problems.append(f"{ema} is not the checkpoint the test sampled with (sha256)")
    results = sorted(p for p in (folder / "results" / "test").glob("*") if p.is_dir())
    if len(results) != 1:
        problems.append(f"results/test/ holds {len(results)} folders, expected one")
    if problems:
        raise Blocked(f"{folder.name}: " + "; ".join(problems))
    return {"folder": folder, "target": target, "fold": fold, "results": results[0],
            "source": src, "snapshot": "all" if fold == "all" else f"fold{fold}"}


def slices_in(results):
    """{case_id: {slice_idx: Out_ path}} from the Out_ file names."""
    out = {}
    for p in results.glob("Out_*.npy"):
        m = SLICE_FILE.match(p.name)
        if m:
            out.setdefault(m.group(1), {})[int(m.group(2))] = p
    return out


def find_test_folders(runs, expected_n):
    """Every complete ISLES 2018 test folder, one per (target, fold); refuses two."""
    found, notes = {}, []
    for folder in sorted(runs.glob("test_ctp_only_f13_*_isles18_*")):
        m = TEST_NAME.match(folder.name)
        if not m or not folder.is_dir():
            continue
        n = sum(len(v) for r in (folder / "results" / "test").glob("*")
                for v in slices_in(r).values())
        if n != expected_n:
            notes.append(f"{folder.name}: {n} of {expected_n} slices, not complete; skipped")
            continue
        found.setdefault((m.group(1), m.group(2)), []).append(folder)
    for key, folders in found.items():
        if len(folders) > 1:
            raise Blocked(f"{key}: {len(folders)} complete test folders "
                          f"({', '.join(f.name for f in folders)}); name the one to convert")
    return [f[0] for _k, f in sorted(found.items())], notes


def convert(args, H, S, info, sources):
    """One test folder -> one map model per source: "Out" (MRDPM), "GT" (the label MRDPM's
    test saved) and "GT_rolled" (GT one slice off within each scan: a negative control)."""
    target, fold = info["target"], info["fold"]
    kind = "all" if fold == "all" else "k5"
    run_dir = args.matrix / gan_run(target, "adv0", kind)
    config = H.read_config(run_dir)                       # MRDPM's predictor: its label convention
    split = run_dir / ("all" if fold == "all" else f"fold_{fold}") / "split.json"
    rows = test_rows(S, args.test_root, target)
    leak_guard(S, config, train_ids(split), rows, f"MRDPM {target} {info['snapshot']}")
    expected = S._expected_slices(rows)
    have = slices_in(info["results"])
    missing = [(c, s) for c, sl in expected.items() for s in sl if s not in have.get(c, {})]
    extra = [(c, s) for c, sl in have.items() for s in sl if s not in expected.get(c, ())]
    if missing or extra:
        raise Blocked(f"{info['folder'].name}: {len(missing)} index slices without an Out_ "
                      f"(e.g. {missing[:3]}), {len(extra)} Out_ not in the index (e.g. {extra[:3]})")
    loader = S.make_loader(config, args.test_root)
    scale = loader.label_scale
    test = S._test_summary(rows, args.test_root, DATASET)
    names = {"Out": mrdpm_run(target), "GT": mrdpm_run(target, "gt"),
             "GT_rolled": mrdpm_run(target, "gt_rolled")}
    writers = {s: MapWriter(S, args.maps / names[s] / info["snapshot"], scale, config.tmax_ceiling)
               for s in sources}
    clipped = {"low": 0, "high": 0, "envelope_voxels": 0}
    worst_gt = 0.0
    try:
        for cid in sorted(expected):
            case = loader.load(cid)
            S.check_slices(case, expected[cid])
            out, gt = [], []
            for k, s in enumerate(case.slice_idx):
                p = have[cid][s]
                o, g, mk = (np.load(p.parent / f"{pre}{p.name[4:]}").astype(np.float32)
                            for pre in ("Out_", "GT_", "Mask_"))
                env = case.envelope[k]
                if mk.shape != env.shape or not np.array_equal(mk, mk.astype(bool)) \
                        or not np.array_equal(mk.astype(bool), env):
                    raise Blocked(f"{cid} slice {s}: Mask_ is not the scan's envelope")
                err = float(np.abs(2 * g[env] - 1 - case.labels[k][env]).max()) if env.any() else 0.0
                if err > GT_TOL:
                    raise Blocked(f"{cid} slice {s}: 2·GT_ − 1 differs from the label by {err:.2e}")
                if not np.isfinite(o).all():
                    raise Blocked(f"{cid} slice {s}: Out_ is not finite")
                worst_gt = max(worst_gt, err)
                out.append(o)
                gt.append(g)
            pred = 2 * np.stack(out) - 1
            clipped["low"] += int((pred[case.envelope] < -1).sum())
            clipped["high"] += int((pred[case.envelope] > 1).sum())
            clipped["envelope_voxels"] += int(case.envelope.sum())
            label = 2 * np.stack(gt) - 1
            maps = {"Out": np.clip(pred, -1, 1), "GT": label, "GT_rolled": np.roll(label, 1, axis=0)}
            view = case.without_labels()
            for source, w in writers.items():
                w.add(view, maps[source])
    except BaseException:
        for w in writers.values():
            w.abort()
        raise
    model = {"mrdpm_test_folder": str(info["folder"]), "results": str(info["results"]),
             "source_run": info["source"], "weights": "EMA", "samples_per_slice": 1,
             "sampler": "restoration, 1000 steps", "pred_norm": "2·Out_ − 1, clipped to [−1, 1]",
             "clipped_voxels_in_envelope": clipped, "max_abs_gt_minus_label": worst_gt,
             "predictor_run": str(run_dir), "target": target, "fold": fold}
    for source, w in writers.items():
        w.finish(names[source], info["snapshot"], {**model, "source": source}, test, args.gan_bpm)
    share = (clipped["low"] + clipped["high"]) / max(clipped["envelope_voxels"], 1)
    print(f"  {info['folder'].name} -> {', '.join(names[s] for s in sources)} / {info['snapshot']}: "
          f"{test['n_cases']} scans; clipped {clipped['low']} low + {clipped['high']} high "
          f"({share:.3%} of envelope voxels); max |2·GT_ − 1 − label| {worst_gt:.1e}")


def cmd_mrdpm(args, H, S):
    rows = test_rows(S, args.test_root, "rcbf")
    if args.test_folder:
        folders, notes = [Path(f) for f in args.test_folder], []
    else:
        folders, notes = find_test_folders(args.runs, len(rows))
    for n in notes:
        print(f"note: {n}")
    infos = [read_test_folder(f, args.epoch) for f in folders]
    print(f"MRDPM test folders: {len(infos)} -> {args.maps}")
    for info in infos:
        model_dir = args.maps / mrdpm_run(info["target"]) / info["snapshot"]
        sources = ["Out"]
        if not any((args.maps / mrdpm_run(info["target"], "gt")).glob("*/manifest.json")):
            sources += ["GT", "GT_rolled"]
        if model_done(model_dir) and not args.overwrite:
            made = json.loads((model_dir / "manifest.json").read_text())["model"]["mrdpm_test_folder"]
            same = Path(made).name == info["folder"].name
            print(f"  {model_dir.parent.name}/{info['snapshot']}: already there"
                  + ("" if same else f", from {Path(made).name}, not {info['folder'].name} (--overwrite to redo)"))
            sources.remove("Out")
        if sources:
            convert(args, H, S, info, sources)


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--gan-bpm", type=Path, default=GAN_BPM, help="gan_bpm's checkout, read only")
    common.add_argument("--test-root", type=Path, default=TEST_ROOT)
    common.add_argument("--matrix", type=Path, default=MATRIX, help="gan_bpm's runs")
    common.add_argument("--maps", type=Path, default=MAPS, help="where the model folders go")
    common.add_argument("--overwrite", action="store_true")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split", parents=[common], help="write the ISLES 2018 split.json")
    s.add_argument("--out", type=Path, default=ISLES / "split.json")
    g = sub.add_parser("gan-bpm", parents=[common], help="gan_bpm's fold models on ISLES 2018 (GPU)")
    g.add_argument("--target", action="append", choices=TARGETS)
    g.add_argument("--recipe", action="append", choices=RECIPES)
    g.add_argument("--fold", action="append", type=int, choices=FOLDS)
    g.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--skip-check", action="store_true", help="skip the stage-1 reproduction check")
    m = sub.add_parser("mrdpm", parents=[common], help="MRDPM's ISLES 2018 test folders as maps")
    m.add_argument("test_folder", nargs="*", help="default: every complete one under --runs")
    m.add_argument("--runs", type=Path, default=RUNS)
    m.add_argument("--epoch", type=int, default=200)
    args = p.parse_args(argv)
    try:
        H, S = load_gan_bpm(args.gan_bpm)
    except Blocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(f"gan_bpm {args.gan_bpm} at {git_commit(args.gan_bpm) or '<no git>'}, "
          f"MRDPM at {git_commit(REPO) or '<no git>'}; test root {args.test_root}")
    try:
        {"split": cmd_split, "gan-bpm": cmd_gan_bpm, "mrdpm": cmd_mrdpm}[args.cmd](args, H, S)
    except (Blocked, S.Blocked) as exc:                    # stage 1's guards raise its own
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
