"""ctp_only_plan.md §12 on the box: the checks that need the real f13 data and gan_bpm checkpoint.

Two parts, run separately, both from the MRDPM repo root in the job image:

    python scripts/ctp_only_box_checks.py --part cpu --out /radraid2/colivares/mrdpm_ctp_only/box_checks
    python scripts/ctp_only_box_checks.py --part gpu --out ...    # condor/ctp_only_gpu_checks.sub

The config is Tmax's unless `--config` names another (rCBF: config/ctp_only_f13_rcbf_fold0.json,
ctp_only_plan.md §18).

Both parts start with 1: the real checkpoint loads into the predictor, with the sha256 the plan
recorded (§13), and was trained on this data root.

--part cpu (data only, no model passes):
2. The three parts against fold 0's split.json, and the monitor's size.
3. The test part, slice for slice, against gan_bpm's own validation loader, built from the
   checkpoint's config as scripts/masked_input_check.py does: same slices, same values.
4. Orientation (§6): which array axis the brain mirrors across, and what MRDPM's flip
   along dim 2 does to the envelope. Also orientation.png, one slice and its flip.

--part gpu (model passes, CUDA required):
2. Control: the predictor alone, through MRDPM's dataset and wrapper, must score the
   checkpoint's stored val_L_pix_brain on the test part.
3. Network from the real config at 512: the predictor's weights after init, one forward, its
   time and the peak memory.
"""
import argparse
import json
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))       # MRDPM's `models` and `data` must win over gan_bpm's
CONFIG = REPO / "config/ctp_only_f13_tmax_fold0.json"
# Each config's fold-0 generator, as the plan recorded it (§13 step 0 for Tmax, §18 for rCBF).
PLAN_SHA256 = {"ctp_only_f13_tmax_fold0": "2428e88482ce0994229a086ac4bb4a56b7b37ada5494156ae3e2ea0eba3a4731",
               "ctp_only_f13_rcbf_fold0": "bb180198b014a459f21353ac72f2bc741af7ab8fbf329aaef6bb882486505c41"}

RESULTS = []                                    # (name, ok), in order


def check(name, ok, detail=""):
    """One line per check, as scripts/ctp_only_checks.py prints them: FAIL always means
    something is wrong."""
    RESULTS.append((name.strip(), bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""), flush=True)


class Log:
    def __init__(self):
        self.lines = []

    def info(self, msg, *args, **kwargs):
        self.lines.append(str(msg))

    warning = info


def load_config(path):
    """The config as core/praser.py reads it: everything from `//` to the end of a line goes."""
    return json.loads("".join(line.split("//")[0] + "\n" for line in Path(path).open()))


def gan_bpm_val_dataset(ckpt, gan_bpm):
    """gan_bpm's validation set for the checkpoint's fold, built from its stored config."""
    from training import TrainingConfig, build_dataloaders
    cfg = dict(ckpt["config"])
    if cfg.get("split_file") and not Path(cfg["split_file"]).is_absolute():
        cfg["split_file"] = str(gan_bpm / cfg["split_file"])
    cfg["num_workers"] = 0
    known = {f.name for f in fields(TrainingConfig)}
    # No .validate(): it only raises (the run passed it already) and it imports gan_bpm's
    # `models`, which MRDPM's own `models` shadows in this process.
    _, val_loader = build_dataloaders(TrainingConfig(**{k: v for k, v in cfg.items() if k in known}))
    return val_loader.dataset


def dice(a, b):
    return 2.0 * (a & b).sum() / max(int(a.sum() + b.sum()), 1)


def recentre(m, axis):
    """Shift `m` along `axis` so its centroid sits on the frame's centre (what a centred brain would be)."""
    idx = np.nonzero(m.any(axis=1 - axis))[0]
    if not len(idx):
        return m
    centroid = np.average(np.arange(m.shape[axis]), weights=m.sum(axis=1 - axis))
    return np.roll(m, int(round((m.shape[axis] - 1) / 2 - centroid)), axis=axis)


def cpu_part(args, ds_args, ckpt, gan_bpm, summary):
    print("\n2. parts against split.json")
    from data.dataset import MCTPDataset_mask
    parts = {p: MCTPDataset_mask(**a) for p, a in ds_args.items()}
    split = json.loads(Path(ds_args["train"]["split_json"]).read_text())
    tr, mo, te = (set(parts[p].case_ids) for p in ("train", "monitor", "test"))
    check("train + monitor = train_case_ids, disjoint", tr | mo == set(split["train_case_ids"]) and not tr & mo,
          f"{len(tr)} + {len(mo)} cases")
    check("test = val_case_ids", te == set(split["val_case_ids"]), f"{len(te)} cases")
    pt, pm, pe = (set(parts[p].patient_ids) for p in ("train", "monitor", "test"))
    check("no patient in two parts", not (pt & pm or pt & pe or pm & pe))
    datasets = parts["monitor"].ds.rows.groupby("case_id")["dataset"].first().to_dict()
    check("monitor: one patient per dataset", len(pm) == len(set(datasets.values())), str(datasets))
    for name, part in parts.items():
        print(f"     {name:<8} {len(part.case_ids):>4} cases {len(part):>5} slices")
    summary["parts"] = {n: {"cases": len(p.case_ids), "slices": len(p)} for n, p in parts.items()}
    summary["monitor_cases"] = datasets

    print("\n3. the test part against gan_bpm's validation loader")
    from mctp.transforms import MASK_ENVELOPE, MASK_SUPPORT
    ref = gan_bpm_val_dataset(ckpt, gan_bpm)
    test = parts["test"]
    keys = ["case_id", "slice_idx"]
    same_rows = ref.rows[keys].astype(str).reset_index(drop=True).equals(test.ds.rows[keys].astype(str).reset_index(drop=True))
    check("the same slices in the same order", same_rows, f"{len(test)} vs {len(ref)}")
    worst_cond = worst_gt = 0.0
    exact = {"loss_mask": True, "mask": True}
    sizes, rows = [], []                # envelope sizes and mirror dice, for section 4
    loader = lambda ds: DataLoader(ds, batch_size=None, num_workers=args.workers)
    pairs = zip(loader(test), loader(ref)) if same_rows else ((item, None) for item in loader(test))
    t0 = time.time()
    for item, r in pairs:
        env = item["mask"][0].numpy().astype(bool)
        sizes.append(int(env.sum()))
        if env.sum() >= 1000:
            rows.append([dice(env, env[::-1, :]), dice(env, env[:, ::-1]),
                         dice(recentre(env, 0)[::-1, :], recentre(env, 0)),
                         dice(recentre(env, 1)[:, ::-1], recentre(env, 1))])
        if r is None:
            continue
        image, label, m = r
        worst_cond = max(worst_cond, float((2 * item["cond_image"] - 1 - image[0]).abs().max()))
        worst_gt = max(worst_gt, float((item["gt_image"] - (label + 1) / 2 * m[MASK_ENVELOPE:MASK_ENVELOPE + 1]).abs().max()))
        exact["loss_mask"] &= torch.equal(item["loss_mask"], m[MASK_SUPPORT:MASK_SUPPORT + 1])
        exact["mask"] &= torch.equal(item["mask"], m[MASK_ENVELOPE:MASK_ENVELOPE + 1])
    print(f"     read {len(sizes)} slices{' from each' if same_rows else ''} in {time.time() - t0:.0f} s")
    check("2·cond − 1 = gan_bpm's input, whole frame", same_rows and worst_cond < 1e-6, f"max {worst_cond:.1e}")
    check("gt = (gan_bpm label + 1)/2 × envelope", same_rows and worst_gt < 1e-6, f"max {worst_gt:.1e}")
    check("loss_mask = gan_bpm's support, exactly", same_rows and exact["loss_mask"])
    check("mask = gan_bpm's envelope, exactly", same_rows and exact["mask"])

    print("\n4. orientation: MRDPM flips dim 2, the array rows (axis 0)")
    med = np.median(np.array(rows), axis=0)
    print(f"     {len(rows)} slices, median envelope dice with its mirror image:")
    print(f"       across axis 0 (MRDPM's flip) {med[0]:.3f} as stored, {med[2]:.3f} re-centred")
    print(f"       across axis 1                {med[1]:.3f} as stored, {med[3]:.3f} re-centred")
    check("the brain mirrors best across axis 0, the axis MRDPM flips (re-centred)", med[2] > med[3],
          f"{med[2]:.3f} vs {med[3]:.3f}")
    summary["orientation_median_dice"] = dict(zip(["axis0", "axis1", "axis0_recentred", "axis1_recentred"],
                                                  map(float, med)))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        item = test[int(np.argmax(sizes))]                                # the slice with the most brain
        whole = item["cond_image"][None]                                  # (1, 3, H, W), as the U-Net sees it
        brain = whole * item["mask"][None]
        show = lambda a: np.clip(a / max(float(np.percentile(a[a > 0], 99)) if (a > 0).any() else 1.0, 1e-6), 0, 1)
        x, xf = show(brain[0, 0].numpy()), show(torch.flip(brain, [2])[0, 0].numpy())
        fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
        ax[0].imshow(whole[0, 0].numpy(), cmap="gray", vmax=0.35)
        ax[0].set_title(f"{item['path']}, phase 1\n(image rows = array axis 0 = tensor dim 2)")
        ax[1].imshow(torch.flip(whole, [2])[0, 0].numpy(), cmap="gray", vmax=0.35)
        ax[1].set_title("torch.flip(x, [2]): MRDPM's mirror")
        ax[2].imshow(np.stack([x, xf, np.zeros_like(x)], -1))
        ax[2].set_title("brain only: red = slice, green = mirror\n(yellow where they agree)")
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
        fig.tight_layout()
        fig.savefig(args.out / "orientation.png", dpi=80)
        print(f"     wrote {args.out / 'orientation.png'} ({item['path']})")
    except ImportError as exc:
        print(f"     no PNG: {exc}")


def gpu_part(args, cfg, ds_args, P, ckpt, summary):
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    print(f"\n     {props.name}, {props.total_memory / 2**30:.1f} GiB")
    from data.dataset import MCTPDataset_mask
    test = MCTPDataset_mask(**ds_args["test"])

    print("\n2. control: the predictor alone scores gan_bpm's val_L_pix_brain")
    P.to(dev)
    err = sup = 0.0
    t0 = time.time()
    with torch.no_grad():
        for bi, b in enumerate(DataLoader(test, batch_size=args.batch_size, num_workers=args.workers)):
            gt, loss_mask = b["gt_image"].to(dev), b["loss_mask"].to(dev)
            y01, _ = P(b["cond_image"].to(dev))
            err += float(((y01 - gt).abs() * loss_mask).sum())
            sup += float(loss_mask.sum())
            if bi % 20 == 0:
                done = min((bi + 1) * args.batch_size, len(test))
                print(f"     {done}/{len(test)} slices, {done / (time.time() - t0):.2f} slices/s", flush=True)
    stored = ckpt.get("metrics", {}).get("val_L_pix_brain", ckpt.get("select_value"))
    got = 2 * err / sup             # [0, 1] units -> gan_bpm's [-1, 1] units
    health = cfg["model"]["which_model"]["args"]["health_log"]
    physical = got * health["label_scale"] / 2      # [-1, 1] spans the label's range: 12 s, or rCBF 2
    check("pooled L1 over loss_mask = the checkpoint's val_L_pix_brain", abs(got - stored) < 1e-4 * stored,
          f"{got:.6f} vs {stored:.6f}, {physical:.3f} {health['unit']}")
    summary["predictor_alone"] = {"val_L_pix_brain": got, "stored": stored, "physical": physical, "unit": health["unit"]}
    P.cpu()

    print("\n3. Network from the real config, at 512")
    import models.loss
    from models import define_network
    net = define_network(Log(), {"phase": "train"}, cfg["model"]["which_networks"][0])
    own = net.initial_net.G.state_dict()
    check("after define_network (train), the predictor's weights are the checkpoint's",
          all(torch.equal(own[k], ckpt["generator"][k]) for k in own) and own.keys() == ckpt["generator"].keys())
    net.to(dev)
    net.set_loss(models.loss.mse_loss)
    net.set_new_noise_schedule(device=dev, phase="train")
    b = {k: v.to(dev) for k, v in next(iter(DataLoader(test, batch_size=1))).items() if torch.is_tensor(v)}
    torch.cuda.reset_peak_memory_stats(dev)
    seconds = []
    with torch.no_grad():
        for _ in range(2):          # the first includes CUDA's warm-up
            t0 = time.time()
            loss = net(b["gt_image"], b["cond_image"], mask=b["mask"], loss_mask=b["loss_mask"])
            torch.cuda.synchronize(dev)
            seconds.append(time.time() - t0)
    check("one forward on a real 512 slice gives a finite loss", bool(torch.isfinite(loss)), f"{float(loss):.4f}")
    peak = torch.cuda.max_memory_allocated(dev) / 2**30
    print(f"     forward without gradients, batch 1: {seconds[0]:.2f} s cold, {seconds[1]:.3f} s warm, "
          f"peak {peak:.2f} GiB allocated")
    n_unet = sum(p.numel() for p in net.denoise_fn.parameters())
    print(f"     U-Net {n_unet:,} parameters, predictor {sum(p.numel() for p in net.initial_net.parameters()):,}")
    summary["network_forward"] = {"device": props.name, "seconds_cold": seconds[0], "seconds_warm": seconds[1],
                                  "peak_gib": peak}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--part", choices=["cpu", "gpu"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers (0 on macOS: they must fork)")
    p.add_argument("--batch-size", type=int, default=8, help="for the gpu part's control")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.part == "gpu" and not torch.cuda.is_available():
        print("CUDA is not available in this process: the gpu part needs a GPU (condor/ctp_only_gpu_checks.sub)")
        sys.exit(2)

    cfg = load_config(args.config)
    net_args = cfg["model"]["which_networks"][0]["args"]
    ip = net_args["initial_predictor"]
    ds_args = {p: cfg["datasets"][s][k]["args"] for p, (s, k) in
               {"train": ("train", "which_dataset"), "monitor": ("train", "which_val_dataset"),
                "test": ("test", "which_dataset")}.items()}
    gan_bpm = Path(ip["gan_bpm_root"])
    if str(gan_bpm) not in sys.path:
        sys.path.append(str(gan_bpm))
    summary = {}

    print(f"--part {args.part}\n\n1. the gan_bpm checkpoint")
    from models.gan_bpm_predictor import GanBpmInitialPredictor
    P = GanBpmInitialPredictor(**{k: v for k, v in ip.items() if k != "type"}).eval()
    prov = P.provenance
    print(f"     {prov}")
    check("loads into the predictor (config guards, strict weights)", True)
    check("sha256 is the one recorded in the plan", prov["sha256"] == PLAN_SHA256.get(cfg["name"]), prov["sha256"][:16])
    check("it was trained on this data root", Path(str(prov["data_root"])) == Path(ds_args["train"]["mctp_root"]),
          str(prov["data_root"]))
    ckpt = torch.load(ip["checkpoint"], map_location="cpu", weights_only=False)

    if args.part == "cpu":
        cpu_part(args, ds_args, ckpt, gan_bpm, summary)
    else:
        gpu_part(args, cfg, ds_args, P, ckpt, summary)

    failed = [name for name, ok in RESULTS if not ok]
    summary["checks"] = {"passed": len(RESULTS) - len(failed), "total": len(RESULTS), "failed": failed}
    out = args.out / f"summary_{args.part}.json"
    out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"\nwrote {out}")
    print("=" * 72)
    if failed:
        print(f"RESULT: {len(failed)} OF {len(RESULTS)} box checks ({args.part}) FAILED")
        for name in failed:
            print(f"  FAIL  {name}")
    else:
        print(f"RESULT: ALL {len(RESULTS)} box checks ({args.part}) PASSED")
    print("=" * 72)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
