"""ctp_only_plan.md §12: CPU checks of the ctp-only pieces, on synthetic data.

Builds a small fake f13 root (three datasets, a skull, no-data holes, one patient with two
cases; Tmax and rCBF labels on the same inputs) and fake gan_bpm checkpoints in a temp dir. Then
it checks the dataset, the `which_val_dataset` hook, the gan_bpm predictor, Network's
conditioning and loss masks, the U-Net's `mirror_hs` switch, `zero_hole_residual` and the
four configs (each target's fold-0 and all-data), and runs run.py's train (debug) and test
phases end to end on the CPU: Tmax's, rCBF's (train only) and the all-data configs' (train
only, no validation set). Checks marked "negative control" show that the check above them can
fail.

    python scripts/ctp_only_checks.py [--gan-bpm PATH] [--keep DIR]

Needs MRDPM's packages (the box image has them) and a gan_bpm checkout, but no GPU and no
real data. scripts/ctp_only_box_checks.py covers what needs the real data and checkpoint.
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import types
from pathlib import Path

# A CPU check everywhere. The box's docker shows containers its GPUs even without --gpus, and
# core/util.py:set_device then puts run.py's model and batches on cuda:0, next to the CPU noise
# schedule below. Hidden before torch is imported, so torch never sees one.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))       # MRDPM's `models` and `data` must win over gan_bpm's
CONFIG = REPO / "config/ctp_only_f13_tmax_fold0.json"
CONFIG_RCBF = REPO / "config/ctp_only_f13_rcbf_fold0.json"
CONFIG_ALL = {t: REPO / f"config/ctp_only_f13_{t}_all.json" for t in ("tmax", "rcbf")}
H = 256                             # gan_bpm's generator needs H, W >= 256
MARKERS = {"isles24": -30.0, "ucla_evt": -1.0, "ucla_neg": -1.0}    # each dataset's no-data value
RCBF_MARKERS = {"isles24": None, "ucla_evt": -1.0, "ucla_neg": -1.0}  # ISLES'24 rCBF has none

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""), flush=True)


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    return False


class Log:
    """Stands in for core.logger.InfoLogger."""
    def __init__(self):
        self.lines = []

    def info(self, msg, *args, **kwargs):
        self.lines.append(str(msg))

    warning = info


def load_config(path=CONFIG):
    """The config as core/praser.py reads it: everything from `//` to the end of a line goes."""
    return json.loads("".join(line.split("//")[0] + "\n" for line in Path(path).open()))


def same_weights(module, state):
    own = module.state_dict()
    return own.keys() == state.keys() and all(torch.equal(own[k], state[k]) for k in own)


# ---------------------------------------------------------------- fixtures

def make_root(root):
    """4 cases x 2 slices per dataset, raw HU and raw Tmax as on disk, plus the slice index.

    rCBF as the builder writes it beside Tmax: its own copy of the same input, its own label and
    index. UCLA's no-data voxels are -1; ISLES'24 rCBF has no marker.
    """
    import pandas as pd
    rng = np.random.default_rng(0)
    rng_rcbf = np.random.default_rng(1)     # its own, so the Tmax fixture is what it always was
    yy, xx = np.mgrid[:H, :H]
    r = np.hypot(yy - H / 2, xx - H / 2 + 8)
    brain = r < 92
    rows, rows_rcbf = [], []
    for name, marker in MARKERS.items():
        for i in range(4):
            case = f"{name}-{i}"
            patient = f"{name}-0" if (name, i) == ("isles24", 1) else case     # 2 cases, 1 patient
            for s in range(2):
                img = np.full((3, H, H), -1000.0, np.float32)
                img[:, r < 100] = 1200.0                                     # skull
                img[:, brain] = 35 + 60 * rng.random((3, int(brain.sum())), dtype=np.float32)
                lab = np.full((H, H), marker, np.float32)
                lab[brain] = 14 * rng.random(int(brain.sum()), dtype=np.float32)   # some > 12 s
                lab[brain & (xx < 70 + 10 * s)] = marker                     # a no-data hole
                lab[brain & (yy < 60)] = 0.0                                  # real zeros stay
                rel_i, rel_l = (f"tmax/{case}/slice{s:03d}_{k}.npy" for k in ("input", "label"))
                (root / rel_i).parent.mkdir(parents=True, exist_ok=True)
                np.save(root / rel_i, img)
                np.save(root / rel_l, lab)
                rows.append(dict(case_id=case, patient_id=patient, dataset=name, slice_idx=s,
                                 input_relpath=rel_i, label_relpath=rel_l, label_fg_frac=0.5))

                fill = RCBF_MARKERS[name]
                lab = np.full((H, H), 0.0 if fill is None else fill, np.float32)
                lab[brain] = 2.4 * rng_rcbf.random(int(brain.sum()), dtype=np.float32)   # some > 2
                if fill is not None:
                    lab[brain & (xx < 70 + 10 * s)] = fill                    # a no-data hole
                lab[brain & (yy < 60)] = 0.0                                  # dead tissue stays
                rel_i, rel_l = (f"rcbf/{case}/slice{s:03d}_{k}.npy" for k in ("input", "label"))
                (root / rel_i).parent.mkdir(parents=True, exist_ok=True)
                np.save(root / rel_i, img)
                np.save(root / rel_l, lab)
                rows_rcbf.append(dict(rows[-1], input_relpath=rel_i, label_relpath=rel_l))
    pd.DataFrame(rows).to_parquet(root / "tmax_index.parquet")
    pd.DataFrame(rows_rcbf).to_parquet(root / "rcbf_index.parquet")


def write_split(path, train, val):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"train_case_ids": train, "val_case_ids": val}))
    return path


def make_checkpoints(d, gan_bpm, data_root):
    """A good Tmax checkpoint and five whose config the Tmax predictor must refuse; a good rCBF one
    (gan_bpm stores 40 as an rCBF run's unused ceiling) and an rCBF one it must refuse."""
    from models.gan_bpm_predictor import load_generator_module
    torch.manual_seed(0)
    state = load_generator_module(gan_bpm).Generator(n_phases=3).state_dict()
    good = {"target": "tmax", "tmax_ceiling": 12.0, "n_phases": 3, "label_transform": "linear",
            "fold": 0, "lambda_adv": 0.0, "data_root": str(data_root)}
    bad = {"good": {}, "ceiling40": {"tmax_ceiling": None}, "phases4": {"n_phases": 4},
           "rcbf": {"target": "rcbf"}, "sqrt": {"label_transform": "sqrt"}, "fold1": {"fold": 1},
           "rcbf_good": {"target": "rcbf", "tmax_ceiling": 40.0},
           "rcbf_sqrt": {"target": "rcbf", "tmax_ceiling": 40.0, "label_transform": "sqrt"},
           "all_tmax": {"fold": "all", "all_data": True},
           "all_rcbf": {"target": "rcbf", "tmax_ceiling": 40.0, "fold": "all", "all_data": True}}
    paths = {}
    for name, change in bad.items():
        paths[name] = d / f"{name}.pt"
        torch.save({"epoch": 3, "generator": state, "config": dict(good, **change)}, paths[name])
    return paths, state


def network_opt(cfg, gan_bpm, ckpt):
    """The config's network, pointed at the fixtures and shrunk to H."""
    opt = copy.deepcopy(cfg["model"]["which_networks"][0])
    opt["args"]["initial_predictor"].update(gan_bpm_root=str(gan_bpm), checkpoint=str(ckpt))
    opt["args"]["unet"]["image_size"] = H
    return opt


def read_csv(path):
    return list(csv.DictReader(path.open())) if path.is_file() else []


def batch_of(dataset, n=2):
    from torch.utils.data.dataloader import default_collate
    return default_collate([dataset[i] for i in range(n)])


def main_forward(self, y_0, y_cond=None, mask=None, noise=None):
    """models/network.py:Network.forward as it is on `main`, copied verbatim."""
    from models.network import default, extract
    # sampling from p(gammas)
    b, *_ = y_0.shape
    t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long() # 1-99
    gamma_t1 = extract(self.gammas, t, x_shape=(1, 1))
    sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
    sample_gammas = gamma_t1
    sample_gammas = sample_gammas.view(b, -1)

    y_initial,init_cond = self.initial_net(y_cond)
    y_initial = y_initial*mask
    y_res = (y_0 - y_initial)*mask

    noise = default(noise, lambda: torch.randn_like(y_0))
    y_noisy = self.q_sample(
        y_0=y_res, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)
    if mask is not None:
        noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas, init_cond)
        loss = self.loss_fn(mask*noise, mask*noise_hat) + self.loss_initial(y_0, y_initial)*0.1
    else:
        noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas, init_cond)
        loss = self.loss_fn(noise, noise_hat) + self.loss_initial(y_0, y_initial)*0.1
    return loss


# ---------------------------------------------------------------- checks

def check_dataset(root, split, gan_bpm, work):
    print("\ndataset: data/dataset.py MCTPDataset_mask")
    from data.dataset import MCTPDataset_mask
    from mctp.brain_mask import brain_mask_from_input
    from mctp.dataset import MCTPDataset
    from mctp.transforms import MASK_ENVELOPE, MASK_SUPPORT, build_transforms

    args = dict(mctp_root=str(root), gan_bpm_root=str(gan_bpm), split_json=str(split))
    parts = {p: MCTPDataset_mask(part=p, **args) for p in ("train", "monitor", "test")}
    s = json.loads(split.read_text())
    tr, mo, te = (set(parts[p].case_ids) for p in ("train", "monitor", "test"))
    check("train + monitor = split.json's train_case_ids, and they don't overlap",
          tr | mo == set(s["train_case_ids"]) and not tr & mo)
    check("test = split.json's val_case_ids", te == set(s["val_case_ids"]))
    pt, pm, pe = (set(parts[p].patient_ids) for p in ("train", "monitor", "test"))
    check("no patient is in two parts", not (pt & pm or pt & pe or pm & pe))
    check("monitor: one patient from each dataset",
          sorted(p.rsplit("-", 1)[0] for p in pm) == sorted(MARKERS))
    pair = {"isles24-0", "isles24-1"}
    together = all(pair <= set(MCTPDataset_mask(part="monitor", monitor_seed=seed, **args).case_ids)
                   or pair <= set(MCTPDataset_mask(part="train", monitor_seed=seed, **args).case_ids)
                   for seed in range(6))
    check("a two-case patient goes to monitor or to train whole (6 seeds)", together)
    leak = write_split(work / "leak/split.json", [c for c in s["train_case_ids"] if c != "isles24-1"],
                       s["val_case_ids"] + ["isles24-1"])
    check("negative control: a split with one patient in train and val is refused",
          raises(lambda: MCTPDataset_mask(part="test", **dict(args, split_json=str(leak))), ValueError))
    capped = MCTPDataset_mask(part="monitor", monitor_max_slices=3, **args)
    full_rows, cap_rows = parts["monitor"].ds.rows, capped.ds.rows
    check("monitor_max_slices caps the monitor, evenly spaced from first to last slice",
          len(capped) == 3 and cap_rows.iloc[0].equals(full_rows.iloc[0])
          and cap_rows.iloc[-1].equals(full_rows.iloc[-1]), f"{len(full_rows)} -> {len(capped)}")

    # Against gan_bpm's own pipeline, built here with the bounds spelled out, and against the
    # plan's formulas (§5) computed from the raw .npy files.
    test = parts["test"]
    ref = MCTPDataset(str(root), test.case_ids, label_type="tmax", transform=build_transforms(
        {"label_lo": 0.0, "label_hi": 12.0, "label_valid_eps": -0.5}, train=False))
    keys = ["case_id", "slice_idx"]
    check("the test part has gan_bpm's slices, in its order", ref.rows[keys].equals(test.ds.rows[keys]))
    worst = dict(cond=0.0, cond_raw=0.0, gt=0.0, gt40=np.inf)
    exact = dict(loss_mask=True, loss_mask_raw=True, mask=True, mask_raw=True, shapes=True)
    whole_head = holes = False
    for i in range(len(test)):
        item, (image, label, m) = test[i], ref[i]
        row = test.ds.rows.iloc[i]
        raw_in = np.load(root / row["input_relpath"])
        raw, lab = raw_in.astype(np.float64), np.load(root / row["label_relpath"]).astype(np.float64)
        env = np.asarray(brain_mask_from_input(raw_in)).astype(bool)
        c, g = item["cond_image"].double().numpy(), item["gt_image"][0].double().numpy()
        lm, mk = item["loss_mask"][0].numpy(), item["mask"][0].numpy()
        worst["cond"] = max(worst["cond"], np.abs((2 * c - 1) - image[0].double().numpy()).max())
        worst["cond_raw"] = max(worst["cond_raw"], np.abs(c - np.clip((raw - 10) / 315, 0, 1)).max())
        worst["gt"] = max(worst["gt"], np.abs(g - np.clip(lab, 0, 12) / 12 * env).max())
        worst["gt40"] = min(worst["gt40"], np.abs(g - np.clip(lab, 0, 40) / 40 * env).max())
        exact["loss_mask"] &= np.array_equal(lm, m[MASK_SUPPORT].numpy())
        exact["loss_mask_raw"] &= np.array_equal(lm.astype(bool), env & (lab > -0.5))
        exact["mask"] &= np.array_equal(mk, m[MASK_ENVELOPE].numpy())
        exact["mask_raw"] &= np.array_equal(mk.astype(bool), env)
        exact["shapes"] &= (tuple(item["cond_image"].shape) == (3, H, H)
                            and all(tuple(item[k].shape) == (1, H, H) for k in ("gt_image", "mask", "loss_mask"))
                            and all(item[k].dtype == torch.float32 for k in ("cond_image", "gt_image", "mask", "loss_mask")))
        whole_head |= bool((c[:, ~env] > 0).any())
        holes |= bool((env & ~(lab > -0.5)).any())
    check("2·cond − 1 = gan_bpm's input over the whole frame", worst["cond"] < 1e-6, f"max |diff| {worst['cond']:.1e}")
    check("cond = clip((HU − 10)/315, 0, 1) from the raw input", worst["cond_raw"] < 1e-6, f"{worst['cond_raw']:.1e}")
    check("cond is not masked: the skull outside the envelope is there", whole_head)
    check("gt = clip(Tmax, 0, 12)/12 × envelope from the raw label", worst["gt"] < 1e-6, f"{worst['gt']:.1e}")
    check("  negative control: the 40 s scale does not match", worst["gt40"] > 1e-2, f"{worst['gt40']:.2f}")
    check("loss_mask = gan_bpm's support, exactly", exact["loss_mask"])
    check("loss_mask = envelope ∧ (raw label > −0.5), exactly", exact["loss_mask_raw"])
    check("  negative control: the fixture has holes, so loss_mask = mask would fail", holes)
    check("mask = gan_bpm's envelope = brain_mask_from_input(raw), exactly", exact["mask"] and exact["mask_raw"])
    check("shapes (3|1, H, W) and float32", exact["shapes"])
    paths = [p["path"] for part in parts.values() for p in (part[i] for i in range(len(part)))]
    check("paths end in .npy and are unique across parts",
          all(p.endswith(".npy") for p in paths) and len(set(paths)) == len(paths), paths[0])
    return parts


def check_dataset_rcbf(root, split, gan_bpm, tmax_parts):
    print("\ndataset, rCBF: MCTPDataset_mask with label_type rcbf (§18)")
    from data.dataset import MCTPDataset_mask
    from mctp.brain_mask import brain_mask_from_input

    args = dict(mctp_root=str(root), gan_bpm_root=str(gan_bpm), split_json=str(split), label_type="rcbf")
    parts = {p: MCTPDataset_mask(part=p, **args) for p in ("train", "monitor", "test")}
    keys = ["case_id", "slice_idx"]
    check("each part has Tmax's slices, in Tmax's order (same split, same monitor carve)",
          all(parts[p].ds.rows[keys].reset_index(drop=True).equals(tmax_parts[p].ds.rows[keys].reset_index(drop=True))
              for p in parts))
    test, ttest = parts["test"], tmax_parts["test"]
    worst = dict(gt=0.0, gt12=np.inf)
    exact = dict(loss_mask=True, input=True)
    holes = dead = False
    for i in range(len(test)):
        item, titem = test[i], ttest[i]
        row = test.ds.rows.iloc[i]
        lab = np.load(root / row["label_relpath"]).astype(np.float64)
        env = np.asarray(brain_mask_from_input(np.load(root / row["input_relpath"]))).astype(bool)
        g, lm = item["gt_image"][0].double().numpy(), item["loss_mask"][0].numpy().astype(bool)
        worst["gt"] = max(worst["gt"], np.abs(g - np.clip(lab, 0, 2) / 2 * env).max())
        worst["gt12"] = min(worst["gt12"], np.abs(g - np.clip(lab, 0, 12) / 12 * env).max())
        exact["loss_mask"] &= np.array_equal(lm, env & (lab > -0.5))
        exact["input"] &= torch.equal(item["cond_image"], titem["cond_image"]) and torch.equal(item["mask"], titem["mask"])
        holes |= bool((env & ~(lab > -0.5)).any())
        dead |= bool((lm & (lab == 0)).any())
    check("gt = clip(rCBF, 0, 2)/2 × envelope from the raw label", worst["gt"] < 1e-6, f"{worst['gt']:.1e}")
    check("  negative control: Tmax's 12 s scale does not match", worst["gt12"] > 1e-2, f"{worst['gt12']:.2f}")
    check("loss_mask = envelope ∧ (raw rCBF > −0.5): UCLA's −1 leaves it, exactly", exact["loss_mask"])
    check("  negative control: the fixture has rCBF holes", holes)
    check("rCBF 0 (dead tissue) stays in loss_mask", dead)
    check("cond and mask are Tmax's, exactly: the same input, the same envelope", exact["input"])
    return parts


def check_val_hook(root, split, gan_bpm, parts):
    print("\nwhich_val_dataset: data/__init__.py define_dataset")
    from core.praser import dict_to_nonedict
    from data import define_dataset

    def opt(val_part, name="ctp", debug=None, validation_split=0):
        ds = lambda part: {"name": ["data.dataset", "MCTPDataset_mask"],
                           "args": dict(mctp_root=str(root), gan_bpm_root=str(gan_bpm),
                                        split_json=str(split), part=part)}
        train = {"which_dataset": ds("train"), "dataloader": {"validation_split": validation_split, "args": {}}}
        if val_part:
            train["which_val_dataset"] = ds(val_part)
        return dict_to_nonedict({"name": name, "phase": "train", "seed": 0,
                                 "datasets": {"train": train}, "debug": debug or {}})

    tr, va = define_dataset(Log(), opt("monitor"))
    check("train and monitor come from their own blocks, no slice split",
          len(tr) == len(parts["train"]) and len(va) == len(parts["monitor"]) and va.part == "monitor")
    tr, va = define_dataset(Log(), opt("monitor", "debug_ctp", {"debug_split": 2, "debug_val_split": 1}))
    check("debug cuts them to debug_split and debug_val_split", (len(tr), len(va)) == (2, 1))
    check("negative control: a validation set sharing patients with train is refused",
          raises(lambda: define_dataset(Log(), opt("train")), ValueError))
    tr, va = define_dataset(Log(), opt(None, validation_split=3))
    check("without which_val_dataset, the slice split is as before",
          (len(tr), len(va)) == (len(parts["train"]) - 3, 3))


def check_stage(root, split, gan_bpm, work):
    print("\nstaging: scripts/ctp_only_stage_data.py (what a training job copies in)")
    import shutil
    import subprocess
    import pandas as pd
    from data.dataset import MCTPDataset_mask
    dst = work / "staged"
    shutil.rmtree(dst, ignore_errors=True)
    run = subprocess.run([sys.executable, str(REPO / "scripts/ctp_only_stage_data.py"), str(root), str(dst),
                          "--split", str(split), "--side", "train"], capture_output=True, text=True)
    check("it runs", run.returncode == 0, (run.stdout + run.stderr).strip().replace("\n", " | ")[:150])
    kw = dict(gan_bpm_root=str(gan_bpm), split_json=str(split), label_type="tmax", tmax_ceiling=12.0)
    same = run.returncode == 0
    for part in ("train", "monitor") if same else ():
        a, b = (MCTPDataset_mask(mctp_root=str(r), part=part, **kw) for r in (root, dst))
        same &= len(a) == len(b) and all(torch.equal(a[i][k], b[i][k]) for i in range(len(a))
                                         for k in ("cond_image", "gt_image", "mask", "loss_mask"))
    check("train and monitor read from the copy give the original's items, exactly", same)
    rows = pd.read_parquet(root / "tmax_index.parquet")
    held_out = rows[rows["case_id"].isin(json.loads(split.read_text())["val_case_ids"])]
    check("  negative control: the held-out cases' files were not copied",
          len(held_out) > 0 and not any((dst / f).exists() for f in held_out["input_relpath"]))
    both = work / "staged_all"
    shutil.rmtree(both, ignore_errors=True)
    cmd = [sys.executable, str(REPO / "scripts/ctp_only_stage_data.py"), str(root), str(both),
           "--split", str(split), "--side", "all"]
    first, again = (subprocess.run(cmd, capture_output=True, text=True) for _ in range(2))
    in_split = rows[rows["case_id"].isin(set(json.loads(split.read_text())["train_case_ids"]) | set(held_out["case_id"]))]
    check("--side all copies both sides' files; a rerun copies nothing but the index",
          first.returncode == 0 and again.returncode == 0
          and all((both / f).is_file() for f in list(in_split["input_relpath"]) + list(in_split["label_relpath"]))
          and "copied 0.0 GB" in again.stdout, again.stdout.strip()[-90:])

    # condor/ctp_only_stage_home.sh, target rcbf: into the root that already holds Tmax's copy.
    tmax_index = (both / "tmax_index.parquet").read_bytes()
    tmax_files = {f: (both / f).stat().st_mtime_ns for f in in_split["input_relpath"]}
    rc = subprocess.run(cmd + ["--label-type", "rcbf"], capture_output=True, text=True)
    same = rc.returncode == 0
    for label in ("tmax", "rcbf") if same else ():
        a, b = (MCTPDataset_mask(mctp_root=str(r), part="test", gan_bpm_root=str(gan_bpm), split_json=str(split),
                                 label_type=label) for r in (root, both))
        same &= len(a) == len(b) and all(torch.equal(a[i][k], b[i][k]) for i in range(len(a))
                                         for k in ("cond_image", "gt_image", "mask", "loss_mask"))
    check("staging rCBF beside Tmax: both labels' items from the copy equal the original's",
          same, (rc.stdout + rc.stderr).strip().replace("\n", " | ")[-120:])
    check("  and Tmax's index and files are untouched",
          (both / "tmax_index.parquet").read_bytes() == tmax_index
          and all((both / f).stat().st_mtime_ns == t for f, t in tmax_files.items()))


def check_predictor(ckpts, state, gan_bpm, cfg):
    print("\npredictor: models/gan_bpm_predictor.py")
    from models.gan_bpm_predictor import GanBpmInitialPredictor
    P = GanBpmInitialPredictor(gan_bpm, ckpts["good"], expect_fold=0).eval()
    check("the checkpoint's weights, loaded strict", same_weights(P.G, state))
    check("provenance: sha256 of the checkpoint file",
          P.provenance["sha256"] == hashlib.sha256(ckpts["good"].read_bytes()).hexdigest())
    torch.manual_seed(1)
    x = torch.rand(2, 1, 3, H, H) * 2 - 1
    with torch.no_grad():
        y, hs = P.forward_gan_bpm(x)
        check("y is bitwise Generator.forward's", torch.equal(y, P.G(x)))
        check("  negative control: another input gives another y", not torch.equal(y, P.G(x.clamp(max=0))))
        got, g = {}, P.G
        hooks = [m.register_forward_hook(lambda mod, i, o, n=n: got.__setitem__(n, o)) for n, m in {
            "stem": g.temporal_stem, "skip1": g.unet.encoder1, "skip2": g.unet.encoder2,
            "d4": g.unet.decoder4, "d5": g.unet.decoder5, "d6": g.unet.decoder6}.items()]
        g(x)
        for h in hooks:
            h.remove()
        want = [torch.cat([got["d6"], got["stem"]], 1), torch.cat([got["d5"], got["skip1"]], 1),
                torch.cat([got["d4"], got["skip2"]], 1)]
        check("hs = [decoder6 ⊕ stem, decoder5 ⊕ skip1, decoder4 ⊕ skip2] from G's own pass, each doubled",
              all(torch.equal(h, torch.cat([w, w], 1)) for h, w in zip(hs, want)))
        c = torch.rand(2, 3, H, H)
        y01, _ = P(c)
        check("forward(cond01) = (G(2·cond01 − 1) + 1)/2, in [0, 1]",
              torch.equal(y01, (P.forward_gan_bpm((2 * c - 1).unsqueeze(1))[0] + 1) / 2)
              and 0 <= y01.min() and y01.max() <= 1)
        y512, hs512 = P(torch.rand(1, 3, 512, 512))
        shapes = [tuple(y512.shape[1:])] + [tuple(h.shape[1:]) for h in hs512]
        check("shapes at 512: y and hs as BRAN's", shapes == [(1, 512, 512), (64, 256, 256), (128, 128, 128), (256, 64, 64)],
              str(shapes))
    for name in ("ceiling40", "phases4", "rcbf", "sqrt"):
        check(f"refuses the '{name}' checkpoint", raises(lambda: GanBpmInitialPredictor(gan_bpm, ckpts[name]), ValueError))
    check("refuses a fold-1 checkpoint when expect_fold=0",
          raises(lambda: GanBpmInitialPredictor(gan_bpm, ckpts["fold1"], expect_fold=0), ValueError))
    check("  negative control: without expect_fold it loads",
          not raises(lambda: GanBpmInitialPredictor(gan_bpm, ckpts["fold1"])))
    R = GanBpmInitialPredictor(gan_bpm, ckpts["rcbf_good"], expect_fold=0, target="rcbf")
    check("target rcbf: an rCBF checkpoint (its stored ceiling 40) loads strict, provenance says rcbf",
          same_weights(R.G, state) and R.provenance["target"] == "rcbf")
    check("  the ceiling isn't checked for rCBF: one stored at 12 loads too",
          not raises(lambda: GanBpmInitialPredictor(gan_bpm, ckpts["rcbf"], target="rcbf")))
    check("target rcbf refuses the Tmax checkpoint",
          raises(lambda: GanBpmInitialPredictor(gan_bpm, ckpts["good"], target="rcbf"), ValueError))
    check("target rcbf refuses an rCBF sqrt checkpoint",
          raises(lambda: GanBpmInitialPredictor(gan_bpm, ckpts["rcbf_sqrt"], target="rcbf"), ValueError))
    check("an all-data checkpoint (fold 'all') loads without expect_fold, for each target",
          not raises(lambda: GanBpmInitialPredictor(gan_bpm, ckpts["all_tmax"]))
          and not raises(lambda: GanBpmInitialPredictor(gan_bpm, ckpts["all_rcbf"], target="rcbf")))
    check("  negative control: expect_fold=0 refuses it",
          raises(lambda: GanBpmInitialPredictor(gan_bpm, ckpts["all_tmax"], expect_fold=0), ValueError))

    print("\npredictor inside Network: initialization order (§8.4)")
    from core.base_network import BaseNetwork
    from models import define_network
    log = Log()
    net = define_network(log, {"phase": "train"}, network_opt(cfg, gan_bpm, ckpts["good"]))
    check("after define_network (train), the predictor's weights are the checkpoint's", same_weights(net.initial_net.G, state))
    check("the experiment log names the checkpoint and its sha256",
          any("Initial predictor loaded from" in l and P.provenance["sha256"] in l for l in log.lines))
    net2 = define_network(Log(), {"phase": "test"}, network_opt(cfg, gan_bpm, ckpts["good"]))
    BaseNetwork.init_weights(net2)
    check("  negative control: BaseNetwork's init alone overwrites them", not same_weights(net2.initial_net.G, state))
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            torch.save({"bogus": torch.zeros(1)}, "tmax_net_G.pth")
            net3 = define_network(Log(), {"phase": "train"}, network_opt(cfg, gan_bpm, ckpts["good"]))
            check("a stray ./tmax_net_G.pth is skipped for this predictor", same_weights(net3.initial_net.G, state))
            check("  negative control: loading it strict, as BRAN's path would, raises",
                  raises(lambda: net3.initial_net.load_state_dict(torch.load("tmax_net_G.pth"), strict=True)))
        finally:
            os.chdir(cwd)


def check_network(parts, ckpts, gan_bpm, cfg):
    print("\nNetwork: the U-Net's condition (§7.1) and the loss mask (§7.2)")
    import models.loss
    from models import define_network
    net = define_network(Log(), {"phase": "train"}, network_opt(cfg, gan_bpm, ckpts["good"]))
    net.set_loss(models.loss.mse_loss)
    net.set_new_noise_schedule(device=torch.device("cpu"), phase="train")
    net.eval()                       # no dropout: the comparisons below need a repeatable U-Net
    b = batch_of(parts["test"])
    gt, cond, mask, lm = b["gt_image"], b["cond_image"], b["mask"], b["loss_mask"]

    seen = {"pred": [], "unet": []}
    net.initial_net.register_forward_pre_hook(lambda m, a: seen["pred"].append(a[0].clone()))
    net.denoise_fn.register_forward_pre_hook(lambda m, a: seen["unet"].append(a[0].clone()))
    with torch.no_grad():
        total = net(gt, cond, mask=mask, loss_mask=lm)
        terms = net.loss_terms
        check("forward keeps the loss's two terms for the health log; the loss is exactly diff + 0.1 × init",
              torch.equal(total, terms["loss_diff"] + terms["loss_init"] * 0.1))
        check("forward: the predictor reads the whole condition", torch.equal(seen["pred"][-1], cond))
        check("forward: the U-Net reads condition × mask", torch.equal(seen["unet"][-1][:, :3], cond * mask))
        check("  negative control: on this batch the two differ", not torch.equal(cond, cond * mask))
        check("  negative control: masking the predictor's input too would change y_initial",
              not torch.allclose(net.initial_net(cond)[0], net.initial_net(cond * mask)[0]))
        net.mask_unet_cond = False
        net(gt, cond, mask=mask, loss_mask=lm)
        check("  negative control: with mask_unet_cond off, the U-Net reads the whole condition",
              torch.equal(seen["unet"][-1][:, :3], cond))
        net.mask_unet_cond = True

        net.beta_schedule = dict(net.beta_schedule, test=dict(net.beta_schedule["test"], n_timestep=4))
        net.set_new_noise_schedule(device=torch.device("cpu"), phase="test")
        seen["pred"].clear()
        seen["unet"].clear()
        out, _ = net.restoration(cond, y_t=cond, y_0=gt, mask=mask, sample_num=2, target=gt)
        check("restoration: the predictor reads the whole condition",
              len(seen["pred"]) == 1 and torch.equal(seen["pred"][0], cond))
        check("restoration: every U-Net step reads condition × mask",
              len(seen["unet"]) == 4 and all(torch.equal(u[:, :3], cond * mask) for u in seen["unet"]))
        check("restoration: output finite, and 0 outside the mask",
              bool(torch.isfinite(out).all()) and torch.equal(out * (1 - mask), torch.zeros_like(out)))
        net.set_new_noise_schedule(device=torch.device("cpu"), phase="train")

        class Pointwise(torch.nn.Module):
            """Stands in for the U-Net with no spatial mixing: a voxel's output sees that voxel only."""
            def forward(self, x, gammas, cond):
                return 0.5 * x[:, -1:] + 0.25 * x[:, :1]

        unet, net.denoise_fn = net.denoise_fn, Pointwise()
        noise = torch.randn_like(gt)
        hole, valid = (mask - lm).bool(), lm.bool()
        gt_hole, gt_valid = gt.clone(), gt.clone()
        gt_hole[hole] = 0.9
        gt_valid[valid] = 1 - gt_valid[valid]

        def loss(g, loss_mask):
            torch.manual_seed(0)     # the same t
            return net(g, cond, mask=mask, noise=noise, loss_mask=loss_mask).item()
        check("with loss_mask, gt in the holes doesn't change the loss", loss(gt, lm) == loss(gt_hole, lm))
        check("  negative control: with the mask as the loss mask, it does", loss(gt, None) != loss(gt_hole, None))
        check("  and gt where the label is valid does", loss(gt, lm) != loss(gt_valid, lm))
        net.denoise_fn = unet

        net.mask_unet_cond = False
        torch.manual_seed(0)
        new = net(gt, cond, mask=mask)
        torch.manual_seed(0)
        old = main_forward(net, gt, cond, mask=mask)
        check("without loss_mask and mask_unet_cond, the loss is bit-identical to main's",
              torch.equal(new, old), f"{new.item():.6f}")
        torch.manual_seed(0)
        check("  negative control: with loss_mask it differs", not torch.equal(new, net(gt, cond, mask=mask, loss_mask=lm)))
        net.mask_unet_cond = True

    net.train()
    loss_value = net(gt, cond, mask=mask, loss_mask=lm)
    loss_value.backward()
    grad = lambda m: sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
    check("one training step: finite loss, gradients reach the predictor and the U-Net",
          bool(torch.isfinite(loss_value)) and grad(net.initial_net) > 0 and grad(net.denoise_fn) > 0)

    from models.health import HealthLog
    grads = {n: q.grad.clone() for n, q in net.named_parameters() if q.grad is not None}
    before = loss_value.detach().clone()
    with tempfile.TemporaryDirectory() as d:
        health = HealthLog(d, label_scale=12.0, unit="s")
        health.start_epoch()
        health.step(net, loss_value, len(gt))
        health.end_epoch(1, len(gt), 5e-5, 0)
        row = next(csv.DictReader((Path(d) / "health/train.csv").open()))
    check("health log: reading the loss and the gradients changes neither",
          torch.equal(loss_value.detach(), before)
          and all(torch.equal(q.grad, grads[n]) for n, q in net.named_parameters() if q.grad is not None))
    check("  its row has both loss terms and a positive, finite gradient norm",
          all(math.isfinite(float(row[k])) for k in ("loss_diff", "loss_init")) and 0 < float(row["grad_norm_max"]) < math.inf)


def check_mirror(parts, ckpts, gan_bpm, cfg):
    print("\nthe asymmetry prior's flip: unet.mirror_hs (ctp_only_decisions.md)")
    import models.loss
    from models import define_network
    off = network_opt(cfg, gan_bpm, ckpts["good"])
    on = copy.deepcopy(off)
    del on["args"]["unet"]["mirror_hs"]                 # upstream configs have no such key
    check("the config turns the flip off", off["args"]["unet"].get("mirror_hs") is False)
    nets = {}
    for name, opt in (("on", on), ("off", off)):
        nets[name] = define_network(Log(), {"phase": "train"}, opt).eval()
        nets[name].set_loss(models.loss.mse_loss)
        nets[name].set_new_noise_schedule(device=torch.device("cpu"), phase="train")
    shapes = lambda n: {k: v.shape for k, v in n.state_dict().items()}
    check("both settings have the same weights, so a checkpoint loads into either", shapes(nets["on"]) == shapes(nets["off"]))
    nets["off"].load_state_dict(nets["on"].state_dict())

    b = batch_of(parts["test"])
    pairs, losses = {}, {}
    for name, net in nets.items():
        seen = []
        u = net.denoise_fn
        hooks = [m.flip_sample.register_forward_pre_hook(lambda m, a: seen.append(a[0].clone()))
                 for m in (u.cond_resample1, u.cond_resample2, u.cond_resample3)]
        with torch.no_grad():
            torch.manual_seed(0)     # the same t and noise for both
            losses[name] = net(b["gt_image"], b["cond_image"], mask=b["mask"], loss_mask=b["loss_mask"]).item()
        for h in hooks:
            h.remove()
        pairs[name] = [(x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:]) for x in seen]
    check("default (no key): each of the 3 modules pairs the features with their flip along dim 2",
          len(pairs["on"]) == 3 and all(torch.equal(m, torch.flip(f, [2])) for f, m in pairs["on"]))
    check("mirror_hs false: each of the 3 modules pairs the features with themselves",
          len(pairs["off"]) == 3 and all(torch.equal(m, f) for f, m in pairs["off"]))
    check("  negative control: these features aren't mirror-symmetric",
          not any(torch.equal(f, torch.flip(f, [2])) for f, _ in pairs["on"]))
    check("  negative control: with the same weights, the switch changes the loss", losses["on"] != losses["off"],
          f"{losses['on']:.6f} vs {losses['off']:.6f}")


def check_holes(parts, ckpts, gan_bpm, cfg):
    print("\nNetwork: zero_hole_residual (§18), with the rCBF config and its data")
    import models.loss
    from torch.utils.data.dataloader import default_collate
    from models import define_network
    opt = network_opt(cfg, gan_bpm, ckpts["rcbf_good"])
    check("the rCBF config turns it on", opt["args"].get("zero_hole_residual") is True)
    net = define_network(Log(), {"phase": "train"}, opt)
    net.set_loss(models.loss.mse_loss)
    net.set_new_noise_schedule(device=torch.device("cpu"), phase="train")
    net.eval()                       # no dropout: the comparisons below need a repeatable U-Net
    test = parts["test"]
    with_holes = [i for i in range(len(test)) if bool((test[i]["mask"] > test[i]["loss_mask"]).any())][:2]
    b = default_collate([test[i] for i in with_holes])
    gt, cond, mask, lm = b["gt_image"], b["cond_image"], b["mask"], b["loss_mask"]
    hole, valid = (mask - lm).bool(), lm.bool()
    check("  the batch has holes", len(with_holes) == 2 and bool(hole.any()))
    gt_hole = gt.clone()
    gt_hole[hole] = 0.9
    noise = torch.randn_like(gt)
    seen = []
    hook = net.denoise_fn.register_forward_pre_hook(lambda m, a: seen.append((a[0][:, -1:].clone(), a[1].clone())))

    def run(g, loss_mask=lm):
        torch.manual_seed(0)         # the same t
        with torch.no_grad():
            loss = net(g, cond, mask=mask, noise=noise, loss_mask=loss_mask).item()
        return loss, seen[-1][0], seen[-1][1]

    (l_on, x_on, gam), (l_on2, x_on2, _) = run(gt), run(gt_hole)
    want = (1 - gam.view(-1, 1, 1, 1)).sqrt() * noise
    check("at a hole, the U-Net's noisy input is the noise alone, sqrt(1 − γ)·ε",
          torch.allclose(x_on[hole], want[hole], rtol=0, atol=1e-7))
    check("gt in the holes changes neither the U-Net's input nor the loss (the real U-Net)",
          torch.equal(x_on, x_on2) and l_on == l_on2, f"{l_on:.6f}")
    net.zero_hole_residual = False
    (l_off, x_off, _), (l_off2, x_off2, _) = run(gt), run(gt_hole)
    check("  negative control: off, gt in the holes reaches the input and, through the U-Net, the loss",
          not torch.equal(x_off, x_off2) and l_off != l_off2, f"{l_off:.6f} vs {l_off2:.6f}")
    check("on or off, the input where the label is valid is the same", torch.equal(x_on[valid], x_off[valid]))
    no_mask = run(gt, None)[0]
    net.zero_hole_residual = True
    check("without loss_mask it does nothing (nothing says where the holes are)", run(gt, None)[0] == no_mask)
    hook.remove()


ABSENT = object()


def config_diff(a, b, path=""):
    """The paths where two parsed configs differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            yield from config_diff(a.get(k, ABSENT), b.get(k, ABSENT), f"{path}.{k}" if path else k)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            yield from config_diff(x, y, f"{path}[{i}]")
    elif a != b:
        yield path


# What may differ between the rCBF config and Tmax's: the label's keys, nothing of the recipe (§18).
RCBF_KEYS = ["name", "datasets.*.args.split_json", "datasets.*.args.label_type",
             "model.which_model.args.health_log.label_scale", "model.which_model.args.health_log.unit",
             "model.which_networks[[]0[]].args.initial_predictor.checkpoint",
             "model.which_networks[[]0[]].args.initial_predictor.target",
             "model.which_networks[[]0[]].args.zero_hole_residual"]


def check_config(cfg, path=CONFIG, tmax_cfg=None):
    print(f"\nconfig: {path.relative_to(REPO)}")
    import fnmatch
    import inspect
    from data.dataset import MCTPDataset_mask
    from mctp.label_scale import label_bounds
    a = cfg["model"]["which_networks"][0]["args"]
    ip = a["initial_predictor"]
    tr, mo, te = (cfg["datasets"][p][k]["args"] for p, k in
                  (("train", "which_dataset"), ("train", "which_val_dataset"), ("test", "which_dataset")))
    check("parts: train, monitor, test", (tr["part"], mo["part"], te["part"]) == ("train", "monitor", "test"))
    check("the three share root, gan_bpm, split, label and ceiling",
          all(tr[k] == mo[k] == te[k] for k in ("mctp_root", "gan_bpm_root", "split_json", "label_type", "tmax_ceiling")))
    default = {k: p.default for k, p in inspect.signature(MCTPDataset_mask.__init__).parameters.items()}
    check("train and monitor carve the same monitor cases",
          all(tr.get(k, default[k]) == mo.get(k, default[k]) for k in ("monitor_cases_per_dataset", "monitor_seed")))
    split, ckpt = Path(tr["split_json"]), Path(ip["checkpoint"])
    check("the checkpoint is the split's fold: fold_k/split.json, fold_k/checkpoints/best.pt",
          ckpt.parent.parent == split.parent and ckpt.parent.name == "checkpoints")
    check("expect_fold is that fold", ip.get("expect_fold") == int(split.parent.name.rsplit("_", 1)[-1]))
    label = tr["label_type"]
    check("the predictor's target is the datasets' label, and its gan_bpm run is that label's",
          ip.get("target", "tmax") == label and split.parent.parent.name.startswith(f"gan_bpm_{label}_"),
          f"{label}, {split.parent.parent.name}")
    check("predictor and datasets read the same gan_bpm checkout", ip["gan_bpm_root"] == tr["gan_bpm_root"])
    check("unet.in_channel = 3 phases + 1, n_phases 3", a["unet"]["in_channel"] == 4 and ip["n_phases"] == 3)
    check("U-Net at the generator's 512", a["unet"]["image_size"] == 512)
    check("mask_unet_cond is on", a["mask_unet_cond"] is True)
    bs = cfg["datasets"]["train"]["dataloader"]["args"]["batch_size"]
    check("log_iter is a multiple of the batch size (iter counts samples)", cfg["train"]["log_iter"] % bs == 0)
    check("the debug cut is whole batches: the train loader drops an incomplete one",
          cfg["debug"]["debug_split"] % bs == 0 and cfg["datasets"]["train"]["dataloader"]["args"]["drop_last"])
    hl = cfg["model"]["which_model"]["args"].get("health_log") or {}
    lo, hi = label_bounds(label, tr["tmax_ceiling"])
    unit = {"tmax": "s", "rcbf": "rCBF"}[label]
    check(f"health log on, errors in {unit} (label_scale = the label's range, {hi - lo:g})",
          hl.get("label_scale") == hi - lo and hl.get("unit") == unit)
    if tmax_cfg is None:
        check("zero_hole_residual is off: the Tmax recipe as it trains", not a.get("zero_hole_residual"))
        return
    extra = [d for d in config_diff(tmax_cfg, cfg) if not any(fnmatch.fnmatchcase(d, k) for k in RCBF_KEYS)]
    check("it is Tmax's config except for the label's keys", not extra, ", ".join(extra))
    changed = copy.deepcopy(cfg)
    changed["datasets"]["train"]["dataloader"]["args"]["batch_size"] = 2
    caught = [d for d in config_diff(tmax_cfg, changed) if not any(fnmatch.fnmatchcase(d, k) for k in RCBF_KEYS)]
    check("  negative control: a changed batch size is caught", caught == ["datasets.train.dataloader.args.batch_size"])


# What may differ between an all-data config and its target's fold-0 config: the split, no monitor,
# ISLES 2018 as the test set, and the all-data generator without a fold (ctp_only_eval_plan.md §6).
ALL_KEYS = ["name", "datasets.train.which_dataset.args.split_json",
            "datasets.train.which_dataset.args.monitor_cases_per_dataset", "datasets.train.which_val_dataset",
            "datasets.test.which_dataset.args.mctp_root", "datasets.test.which_dataset.args.split_json",
            "model.which_networks[[]0[]].args.initial_predictor.checkpoint",
            "model.which_networks[[]0[]].args.initial_predictor.expect_fold"]


def check_config_all(cfg, path, fold0_cfg):
    print(f"\nconfig: {path.relative_to(REPO)} (all-data)")
    import fnmatch
    tr = cfg["datasets"]["train"]["which_dataset"]["args"]
    ip = cfg["model"]["which_networks"][0]["args"]["initial_predictor"]
    fold0_ip = fold0_cfg["model"]["which_networks"][0]["args"]["initial_predictor"]
    extra = [d for d in config_diff(fold0_cfg, cfg) if not any(fnmatch.fnmatchcase(d, k) for k in ALL_KEYS)]
    check("it is the fold-0 config except for the split, the monitor, the test set and the predictor", not extra,
          ", ".join(extra))
    changed = copy.deepcopy(cfg)
    changed["model"]["which_model"]["args"]["optimizers"][0]["lr"] = 1e-4
    caught = [d for d in config_diff(fold0_cfg, changed) if not any(fnmatch.fnmatchcase(d, k) for k in ALL_KEYS)]
    check("  negative control: a changed learning rate is caught", caught == ["model.which_model.args.optimizers[0].lr"])
    check("every case trains: no monitor carve, no monitor set",
          tr.get("monitor_cases_per_dataset") == 0 and "which_val_dataset" not in cfg["datasets"]["train"])
    split, ckpt = Path(tr["split_json"]), Path(ip["checkpoint"])
    fold_run = Path(fold0_ip["checkpoint"]).parent.parent.parent.name
    check("split and predictor are gan_bpm's all-data run of the folds' recipe: <run>_all/all/{split.json, checkpoints/epoch_<M>.pt}",
          split.parent.name == "all" and ckpt.parent.parent == split.parent and ckpt.name.startswith("epoch_")
          and split.parent.parent.name == fold_run.replace("_k5", "_all"), f"{split.parent.parent.name}, {ckpt.name}")
    check("no expect_fold: the all-data checkpoint has none", "expect_fold" not in ip)
    check("the test set is ISLES 2018, never the CV data it trained on",
          "isles2018" in cfg["datasets"]["test"]["which_dataset"]["args"]["mctp_root"])


def check_run_py_all(cfg, root, gan_bpm, ckpt, work):
    """run.py train -d with an all-data config: every case trains, no validation set."""
    import argparse as ap
    import pandas as pd
    import core.praser as Praser
    import models.network as network
    from data.dataset import MCTPDataset_mask
    try:
        import npy2img  # noqa: F401
    except ImportError:
        sys.modules["npy2img"] = types.ModuleType("npy2img")
    import run

    label = cfg["datasets"]["train"]["which_dataset"]["args"]["label_type"]
    print(f"\nrun.py train -d on the CPU, the all-data {label} config")
    cases = sorted(pd.read_parquet(root / f"{label}_index.parquet")["case_id"].unique())
    split = write_split(work / f"all_{label}/all/split.json", cases, [])
    kw = dict(mctp_root=str(root), gan_bpm_root=str(gan_bpm), split_json=str(split), label_type=label)
    whole = MCTPDataset_mask(part="train", monitor_cases_per_dataset=0, **kw)
    check("the train part holds every case of the split", whole.case_ids == cases, f"{len(whole.case_ids)} cases")
    check("  negative control: the default monitor carve would take patients out",
          len(MCTPDataset_mask(part="train", **kw).case_ids) < len(cases))

    c = copy.deepcopy(cfg)
    c["path"]["base_dir"] = str(work / "experiments")
    c["datasets"]["train"]["which_dataset"]["args"].update(mctp_root=str(root), gan_bpm_root=str(gan_bpm), split_json=str(split))
    c["datasets"]["train"]["dataloader"]["args"]["num_workers"] = 0       # macOS workers spawn
    c["model"]["which_networks"] = [network_opt(c, gan_bpm, ckpt)]
    for p in ("train", "test"):
        c["model"]["which_networks"][0]["args"]["beta_schedule"][p]["n_timestep"] = 5
    run_dir = work / f"run_{label}_all"
    run_dir.mkdir(exist_ok=True)
    train_cfg = run_dir / "train.json"
    train_cfg.write_text(json.dumps(c, indent=1))
    original = network.Network.set_new_noise_schedule
    network.Network.set_new_noise_schedule = (
        lambda self, device=torch.device("cpu"), phase="train": original(self, device=device, phase=phase))
    cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        args = ap.Namespace(config=str(train_cfg), phase="train", batch=None, gpu_ids=None, debug=True, port="21012")
        opt = Praser.parse(args)
        opt["world_size"] = 1
        run.main_worker(0, 1, opt)
        exp = Path(opt["path"]["experiments_root"])
        check("train: checkpoint 1 written (Network, EMA, state)",
              all((exp / "checkpoint" / f"1{s}").is_file() for s in ("_Network.pth", "_Network_ema.pth", ".state")))
        rows = read_csv(exp / "health/train.csv")
        check("health: train.csv has epoch 1, finite", len(rows) == 1 and all(
            math.isfinite(float(rows[0][k])) for k in ("loss", "loss_diff", "loss_init")))
        check("no validation set: no monitor pass, no monitor.csv",
              not (exp / "results/val").exists() and not (exp / "health/monitor.csv").exists())
        check("train.log names the all-data checkpoint", str(ckpt) in (exp / "train.log").read_text())
    finally:
        os.chdir(cwd)
        network.Network.set_new_noise_schedule = original


def check_run_py(cfg, root, split, ckpt, state, gan_bpm, work, full=True):
    """run.py train -d with `cfg`; with `full`, also a resume and test -d."""
    label = cfg["datasets"]["train"]["which_dataset"]["args"]["label_type"]
    unit = cfg["model"]["which_model"]["args"]["health_log"]["unit"]
    print(f"\nrun.py end to end on the CPU, {label}: train -d" + (", resume, then test -d" if full else ""))
    import argparse as ap
    import core.praser as Praser
    import models.network as network
    from models import define_network
    try:
        import npy2img  # noqa: F401  matplotlib; run.py uses it only after testing, for PNGs
    except ImportError:
        sys.modules["npy2img"] = types.ModuleType("npy2img")
    import run

    c = copy.deepcopy(cfg)
    c["path"]["base_dir"] = str(work / "experiments")
    for p, k in (("train", "which_dataset"), ("train", "which_val_dataset"), ("test", "which_dataset")):
        c["datasets"][p][k]["args"].update(mctp_root=str(root), gan_bpm_root=str(gan_bpm), split_json=str(split))
    for p in ("train", "test"):
        for k in ("args", "val_args"):
            if k in c["datasets"][p]["dataloader"]:
                c["datasets"][p]["dataloader"][k]["num_workers"] = 0     # macOS workers spawn
    c["model"]["which_networks"] = [network_opt(c, gan_bpm, ckpt)]
    for p in ("train", "test"):
        c["model"]["which_networks"][0]["args"]["beta_schedule"][p]["n_timestep"] = 5
    run_dir = work / f"run_{label}"
    run_dir.mkdir(exist_ok=True)
    train_cfg = run_dir / "train.json"
    train_cfg.write_text(json.dumps(c, indent=1))

    # Palette builds the noise schedule on 'cuda' unless told otherwise.
    original = network.Network.set_new_noise_schedule
    network.Network.set_new_noise_schedule = (
        lambda self, device=torch.device("cpu"), phase="train": original(self, device=device, phase=phase))
    cwd = os.getcwd()
    os.chdir(run_dir)                # run.py's cwd holds ./tmax_net_G.pth and ./ddim_test.txt
    try:
        args = ap.Namespace(config=str(train_cfg), phase="train", batch=None, gpu_ids=None, debug=True, port="21012")
        opt = Praser.parse(args)
        opt["world_size"] = 1
        run.main_worker(0, 1, opt)
        exp = Path(opt["path"]["experiments_root"])
        ck = exp / "checkpoint"
        check("train: checkpoint 1 written (Network, EMA, state)",
              all((ck / f"1{s}").is_file() for s in ("_Network.pth", "_Network_ema.pth", ".state")))
        val = exp / "results/val/1"
        names = sorted(p.name for p in val.glob("*.npy")) if val.is_dir() else []
        check("train: the monitor pass wrote GT_, Out_, Process_, Mask_ for 2 slices",
              [sum(n.startswith(k) for n in names) for k in ("GT_", "Out_", "Process_", "Mask_")] == [2, 2, 2, 2])
        outs = [np.load(p) for p in sorted(val.glob("Out_*.npy"))]
        masks = [np.load(p) for p in sorted(val.glob("Mask_*.npy"))]
        check("train: monitor outputs are finite, (H, W), and 0 outside the mask",
              len(outs) == 2 and all(o.shape == (H, H) and np.isfinite(o).all() and not o[m == 0].any()
                                     for o, m in zip(outs, masks)))
        log = (exp / "train.log").read_text()
        check("train.log names the predictor's checkpoint", "Initial predictor loaded from" in log and str(ckpt) in log)
        rows = read_csv(exp / "health/train.csv")
        check("health: train.csv has epoch 1 with the loss terms, gradient norm and LR, all finite",
              len(rows) == 1 and rows[0]["epoch"] == "1" and rows[0]["steps"] == "1" and all(
                  math.isfinite(float(rows[0][k])) for k in ("loss", "loss_diff", "loss_init", "grad_norm_max", "lr")),
              str(rows[0] if rows else None)[:120])
        mon = read_csv(exp / "health/monitor.csv")
        check(f"health: monitor.csv has the epoch-1 pass, errors in {unit} for MRDPM and the predictor alone",
              len(mon) == 1 and mon[0]["slices"] == "2" and all(math.isfinite(float(mon[0][k])) for k in (
                  f"mae_output_{unit}", f"mae_predictor_{unit}", "support_of_envelope")),
              str(mon[0] if mon else None)[:120])
        check("health: the monitor panel", (exp / "health/monitor_e001.png").is_file()
              or importlib.util.find_spec("matplotlib") is None)
        saved = torch.load(ck / "1_Network.pth", map_location="cpu")
        pred = {k[len("initial_net.G."):]: v for k, v in saved.items() if k.startswith("initial_net.G.")}
        check("train: the predictor was fine-tuned, its weights moved",
              pred.keys() == state.keys() and not all(torch.equal(pred[k], state[k]) for k in state))
        if not full:
            return

        # Resume from checkpoint 1 for one more epoch: the LR schedule must continue from epoch 1.
        resumed = copy.deepcopy(c)
        resumed["path"]["resume_state"] = str(ck / "1")
        resumed["debug"]["n_epoch"] = 1
        resume_cfg = run_dir / "resume.json"
        resume_cfg.write_text(json.dumps(resumed, indent=1))
        args = ap.Namespace(config=str(resume_cfg), phase="train", batch=None, gpu_ids=None, debug=True, port="21012")
        opt = Praser.parse(args)
        opt["world_size"] = 1
        run.main_worker(0, 1, opt)
        rexp = Path(opt["path"]["experiments_root"])
        rrows = read_csv(rexp / "health/train.csv")
        check("resume: epoch 2 trains from checkpoint 1 with the LR schedule at epoch 1, not restarted at 0",
              rexp != exp and (rexp / "checkpoint/2_Network.pth").is_file()
              and len(rrows) == 1 and rrows[0]["epoch"] == "2" and rrows[0]["lr_epoch"] == "1",
              str({k: rrows[0][k] for k in ("epoch", "lr", "lr_epoch")} if rrows else None))

        test_cfg = run_dir / "test.json"
        c["path"]["resume_state"] = str(ck / "1")
        test_cfg.write_text(json.dumps(c, indent=1))
        args = ap.Namespace(config=str(test_cfg), phase="test", batch=None, gpu_ids=None, debug=True, port="21012")
        opt = Praser.parse(args)
        opt["world_size"] = 1
        run.main_worker(0, 1, opt)
        test_dir = Path(opt["path"]["experiments_root"]) / "results/test/0"
        tested = sorted(p.name[len("Out_"):].split("_slice")[0] for p in test_dir.glob("Out_*.npy"))
        n_test = c["debug"]["debug_split"]          # the test phase's debug cut (it has no val set)
        check(f"test: Out_ maps for {n_test} slices (debug_split), all from the held-out fold",
              len(tested) == n_test and set(tested) <= set(json.loads(split.read_text())["val_case_ids"]), str(tested))
        check("test: ./ddim_test.txt records resume_state", str(ck / "1") in (run_dir / "ddim_test.txt").read_text())
        # What Palette.load_networks does in the test phase: strict=False over a fresh Network.
        net = define_network(Log(), {"phase": "test"}, network_opt(c, gan_bpm, ckpt))
        missing, unexpected = net.load_state_dict(saved, strict=False)
        schedule = {"gammas", "sqrt_recip_gammas", "sqrt_recipm1_gammas", "posterior_log_variance_clipped",
                    "posterior_mean_coef1", "posterior_mean_coef2"}
        check("test: the MRDPM checkpoint covers every weight (strict=False skips only the schedule buffers)",
              not missing and set(unexpected) <= schedule, f"missing {len(missing)}, unexpected {sorted(unexpected)}")
        check("test: so the predictor tested is the fine-tuned one, not gan_bpm's",
              same_weights(net.initial_net.G, pred) and not same_weights(net.initial_net.G, state))
    finally:
        os.chdir(cwd)
        network.Network.set_new_noise_schedule = original


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gan-bpm", type=Path, default=Path("/raid/colivares/github_repos/gan_bpm"))
    p.add_argument("--keep", type=Path, default=None, help="build the fixtures here and keep them")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    if str(args.gan_bpm) not in sys.path:
        sys.path.append(str(args.gan_bpm))      # appended, as MCTPDataset_mask does

    cfg, cfg_rcbf = load_config(CONFIG), load_config(CONFIG_RCBF)
    cfg_all = {t: load_config(path) for t, path in CONFIG_ALL.items()}
    tmp = None
    if args.keep is None:
        tmp = tempfile.TemporaryDirectory()
        work = Path(tmp.name)
    else:
        work = args.keep
        work.mkdir(parents=True, exist_ok=True)
    root = work / "mctp_root"
    root.mkdir(exist_ok=True)
    print(f"fixtures in {work}")
    make_root(root)
    split = write_split(work / "run_k/fold_0/split.json",
                        [f"{d}-{i}" for d in MARKERS for i in range(3)], [f"{d}-3" for d in MARKERS])
    (work / "ckpts").mkdir(exist_ok=True)
    ckpts, state = make_checkpoints(work / "ckpts", args.gan_bpm, root)

    parts = check_dataset(root, split, args.gan_bpm, work)
    parts_rcbf = check_dataset_rcbf(root, split, args.gan_bpm, parts)
    check_val_hook(root, split, args.gan_bpm, parts)
    check_stage(root, split, args.gan_bpm, work)
    check_predictor(ckpts, state, args.gan_bpm, cfg)
    check_network(parts, ckpts, args.gan_bpm, cfg)
    check_holes(parts_rcbf, ckpts, args.gan_bpm, cfg_rcbf)
    check_mirror(parts, ckpts, args.gan_bpm, cfg)
    check_config(cfg)
    check_config(cfg_rcbf, CONFIG_RCBF, tmax_cfg=cfg)
    for t, fold0 in (("tmax", cfg), ("rcbf", cfg_rcbf)):
        check_config_all(cfg_all[t], CONFIG_ALL[t], fold0)
    check_run_py(cfg, root, split, ckpts["good"], state, args.gan_bpm, work)
    check_run_py(cfg_rcbf, root, split, ckpts["rcbf_good"], state, args.gan_bpm, work, full=False)
    check_run_py_all(cfg_all["tmax"], root, args.gan_bpm, ckpts["all_tmax"], work)
    check_run_py_all(cfg_all["rcbf"], root, args.gan_bpm, ckpts["all_rcbf"], work)

    n_fail = RESULTS.count(False)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    if tmp is not None:
        tmp.cleanup()
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
