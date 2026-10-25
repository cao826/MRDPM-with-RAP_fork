"""bran_ext_plan.md §6: CPU checks of the BRAN and extreme-loss pieces, on synthetic data.

Builds scripts/ctp_only_checks.py's fake f13 root, then checks BRAN at 3 channels, the extreme
loss against gan_bpm's own loss code, scripts/pretrain_bran.py end to end (with a resume), the
BRAN predictor's checkpoint guard, Network's masked predictor input and extreme-loss terms, the
EMA of BatchNorm statistics, run.py train and test with a BRAN arm, the five arm configs and the
DAG, the run-name patterns, and the Python inside the three H100 jobs that rewrites the config.
Checks marked "negative control" show that the check above them can fail.

    python scripts/bran_ext_checks.py [--gan-bpm PATH] [--keep DIR]

scripts/ctp_only_checks.py must still pass too: the code it covers changed (Network.initial,
the EMA, the health log), and every new key is off in its configs.
"""
import argparse
import copy
import json
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ctp_only_checks as C                     # noqa: E402  hides the GPUs, puts MRDPM first on sys.path

import numpy as np                              # noqa: E402
import torch                                    # noqa: E402
import torch.nn.functional as F                 # noqa: E402

check, raises, H, REPO = C.check, C.raises, C.H, C.REPO
VARIANTS = ("bran_l1", "bran_ext", "bran_ext_diffext", "bran_l1_diffext", "bran_extlow")
CONFIGS = {v: REPO / f"config/ctp_only_f13_rcbf_{v}_fold0.json" for v in VARIANTS}
BOTH, LOW = {"weight": 1.0, "side": "both"}, {"weight": 1.0, "side": "low"}
# arm: (the predictor's ext, diffusion_ext, its BRAN's folder)
EXPECT = {"bran_l1": (None, None, "rcbf_l1"), "bran_ext": (BOTH, None, "rcbf_ext"),
          "bran_ext_diffext": (BOTH, BOTH, "rcbf_ext"), "bran_l1_diffext": (None, BOTH, "rcbf_l1"),
          "bran_extlow": (LOW, None, "rcbf_extlow")}
SMALL = {"epochs": 2, "batch_size": 2}          # the checks' pretraining recipe; the rest is the default
SCHEDULE = {"gammas", "sqrt_recip_gammas", "sqrt_recipm1_gammas", "posterior_log_variance_clipped",
            "posterior_mean_coef1", "posterior_mean_coef2"}


def module_from(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def point_at(cfg, root, split, gan_bpm, recipe=None, ckpt=None):
    """The config on the fixtures: the three data blocks, no workers, the network at H."""
    c = copy.deepcopy(cfg)
    for p, k in (("train", "which_dataset"), ("train", "which_val_dataset"), ("test", "which_dataset")):
        c["datasets"][p][k]["args"].update(mctp_root=str(root), gan_bpm_root=str(gan_bpm), split_json=str(split))
    for p in ("train", "test"):
        for k in ("args", "val_args"):
            if k in c["datasets"][p]["dataloader"]:
                c["datasets"][p]["dataloader"][k]["num_workers"] = 0
    a = c["model"]["which_networks"][0]["args"]
    a["unet"]["image_size"] = H
    ip = a["initial_predictor"]
    if recipe is not None:
        ip["pretrain"] = recipe
    if ckpt is not None:
        ip["checkpoint"] = str(ckpt)
    return c


def holed_batch(dataset, n=2):
    """n items with no-data voxels inside the envelope (ISLES'24 rCBF has none)."""
    from torch.utils.data.dataloader import default_collate
    idx = [i for i in range(len(dataset)) if float((dataset[i]["mask"] - dataset[i]["loss_mask"]).sum()) > 0][:n]
    return default_collate([dataset[i] for i in idx])


# ---------------------------------------------------------------- checks

def check_arch():
    print("\nBRAN at 3 channels: models/region_net.py, models/bran_predictor.py")
    from models.bran_predictor import build_bran
    from models.region_net import Unetgenerator_256
    torch.manual_seed(0)
    up, bran = Unetgenerator_256(ngf=32), build_bran(3)
    check("upstream's BRAN still reads 4 channels (NCCT + 3 phases)",
          up.encoder.down1x.conv_guide.weight.shape[1] == 4 and up.encoder.down1.model[0].weight.shape[1] == 8)
    check("ours reads 3, with upstream's state_dict keys",
          bran.encoder.down1x.conv_guide.weight.shape[1] == 3 and bran.state_dict().keys() == up.state_dict().keys())
    bran.eval()
    with torch.no_grad():
        y, hs = bran(torch.rand(2, 3, H, H))
        y2, _ = bran(torch.rand(2, 3, H, H) * 0 + 0.5)
    check(f"at {H}: y (2, 1, {H}, {H}); hs 64/128/256 channels at H/2, H/4, H/8 (cond_resample's widths)",
          tuple(y.shape) == (2, 1, H, H)
          and [tuple(h.shape) for h in hs] == [(2, 64, H // 2, H // 2), (2, 128, H // 4, H // 4), (2, 256, H // 8, H // 8)])
    check("  its output is tanh-bounded and finite", bool(torch.isfinite(y).all()) and float(y.abs().max()) <= 1
          and bool(torch.isfinite(y2).all()))


def check_ext(gan_bpm):
    print("\nthe extreme loss: models/loss.py against gan_bpm's models/losses.py")
    from models.loss import ext_term, ext_weight
    sys.path.insert(0, str(HERE))
    import pretrain_bran as P
    g = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    check("W = (2 gt − 1)²: 1, 0.25, 0, 0.25, 1", torch.allclose(ext_weight(g), torch.tensor([1.0, 0.25, 0.0, 0.25, 1.0])))
    check("  low + high = both; low is 0 above the middle, high 0 below it",
          torch.equal(ext_weight(g, "low") + ext_weight(g, "high"), ext_weight(g))
          and not ext_weight(g, "low")[2:].any() and not ext_weight(g, "high")[:3].any())
    check("  an unknown side is refused", raises(lambda: ext_weight(g, "middle"), ValueError))

    L = module_from(Path(gan_bpm) / "models" / "losses.py", "gan_bpm_losses")
    torch.manual_seed(1)
    pred, gt = torch.rand(3, 1, 32, 32), torch.rand(3, 1, 32, 32)
    support = (torch.rand(3, 1, 32, 32) > 0.3).float()
    pm = lambda x: 2 * x - 1                                                    # noqa: E731
    for lam in (50.0, 200.0, 500.0):
        theirs = 100 * L.l_pix(pm(pred), pm(gt), support) + lam * L.l_ext(pm(pred), pm(gt), support)
        ours, _ = P.predictor_loss(pred, gt, support, {"weight": lam / 50, "side": "both"})
        check(f"λ_ext = {lam:g}: gan_bpm's 100·L_pix + λ_ext·L_ext = 200 × (L1 + (λ_ext / 50)·L_ext) on [0, 1] labels",
              torch.allclose(theirs, 200 * ours, rtol=1e-5), f"{float(theirs):.5f} vs {float(200 * ours):.5f}")
    wrong, _ = P.predictor_loss(pred, gt, support, {"weight": 2.0, "side": "both"})
    theirs50 = 100 * L.l_pix(pm(pred), pm(gt), support) + 50 * L.l_ext(pm(pred), pm(gt), support)
    check("  negative control: weight 2 is not λ_ext = 50", not torch.allclose(theirs50, 200 * wrong, rtol=1e-5))
    _, terms = P.predictor_loss(pred, gt, support, None)
    check("  without ext the loss is L1 alone, and ext_low + ext_high = ext_both",
          torch.equal(P.predictor_loss(pred, gt, support, None)[0], terms["l1"])
          and torch.allclose(terms["ext_low"] + terms["ext_high"], terms["ext_both"]))
    ratio_ours = ext_term(pred, gt, ext_weight(gt), support) / F.l1_loss(support * pred, support * gt)
    ratio_pre = terms["ext_both"] / terms["l1"]
    check("Network's L_init terms (means over every voxel) keep the pretraining's ext : L1 ratio (means over support)",
          torch.allclose(ratio_ours, ratio_pre, rtol=1e-5))


def check_pretrain(cfg_ext, root, split, gan_bpm, work, parts):
    print("\nscripts/pretrain_bran.py end to end (CPU, 2 epochs, stopped after 1 and resumed)")
    import pretrain_bran as P
    from models.bran_predictor import build_bran, normalize_ext, pretrain_recipe
    recipe = pretrain_recipe(SMALL)
    c = point_at(cfg_ext, root, split, gan_bpm, recipe)
    path = work / "pretrain_ext.json"
    path.write_text(json.dumps(c, indent=1))
    out = work / "bran_ext_fold0"
    argv = ["-c", str(path), "--out", str(out), "--workers", "0", "--device", "cpu"]
    status = P.main(argv + ["--stop-after", "1"])
    log = C.read_csv(out / "pretrain_log.csv")
    check("stopped after epoch 1: a log row and a resume file, no checkpoint",
          status == 3 and len(log) == 1 and (out / "last_state.pt").is_file() and not (out / "bran_G.pth").exists())
    status = P.main(argv)
    log = C.read_csv(out / "pretrain_log.csv")
    check("resumed: epoch 2 only (the log has epochs 1, 2), then the checkpoint",
          status == 0 and [r["epoch"] for r in log] == ["1", "2"] and (out / "bran_G.pth").is_file(), str([r["epoch"] for r in log]))
    check("  the LR falls linearly to 0: 2e-4, then 1e-4",
          len(log) == 2 and abs(float(log[0]["lr"]) - 2e-4) < 1e-12 and abs(float(log[1]["lr"]) - 1e-4) < 1e-12)
    check("  its columns are finite; the extreme loss's halves add up",
          all(np.isfinite(float(r[k])) for r in log for k in ("train_loss", "train_l1", "mon_l1", "mon_mae_label"))
          and all(abs(float(r["train_ext_low"]) + float(r["train_ext_high"]) - float(r["train_ext_both"])) < 1e-6 for r in log))
    check("  negative control: a third call refuses (the pretraining is done)", raises(lambda: P.main(argv), SystemExit))
    other = copy.deepcopy(c)
    other["model"]["which_networks"][0]["args"]["initial_predictor"]["ext"] = {"weight": 2.0, "side": "both"}
    (work / "pretrain_other.json").write_text(json.dumps(other, indent=1))
    (work / "other").mkdir(exist_ok=True)
    (work / "other" / "last_state.pt").write_bytes((out / "last_state.pt").read_bytes())
    check("  negative control: another config's resume file is refused",
          raises(lambda: P.main(["-c", str(work / "pretrain_other.json"), "--out", str(work / "other"),
                                 "--workers", "0", "--device", "cpu"]), SystemExit))

    ck = torch.load(out / "bran_G.pth", map_location="cpu", weights_only=False)
    meta = ck["meta"]
    check("meta: rCBF, fold 0, 3 channels, masked input, the config's ext and recipe, 2 epochs, not debug",
          (meta["target"], meta["fold"], meta["in_channels"], meta["mask_cond"], meta["ext"], meta["pretrain"],
           meta["epochs_done"], meta["debug"]) == ("rcbf", 0, 3, True, normalize_ext(BOTH), recipe, 2, False))
    check("  it trained on the train part and mapped the test part, which don't overlap",
          meta["train_case_ids"] == parts["train"].case_ids and meta["heldout_case_ids"] == parts["test"].case_ids
          and not set(meta["train_case_ids"]) & set(meta["heldout_case_ids"]))
    held = out / "heldout"
    want = {parts["test"][i]["path"] for i in range(len(parts["test"]))}
    have = {p.name[len("Init_"):] for p in held.glob("Init_*.npy")}
    check("held-out maps: Init_, GT_ and Mask_ for every test slice, (H, W) float32",
          have == want and all((held / f"{pre}{n}").is_file() for n in want for pre in ("GT_", "Mask_"))
          and np.load(held / f"Init_{sorted(want)[0]}").shape == (H, H))
    G = build_bran(3)
    G.load_state_dict(ck["generator"])
    G.eval()
    worst_init = worst_gt = 0.0
    zero_out = masked_input = True
    with torch.no_grad():
        for i in range(len(parts["test"])):
            item = parts["test"][i]
            cond, mask = item["cond_image"][None], item["mask"][None]
            y = (G(cond * mask)[0] * mask)[0, 0].numpy()
            saved = np.load(held / f"Init_{item['path']}")
            worst_init = max(worst_init, float(np.abs(saved - y).max()))
            worst_gt = max(worst_gt, float(np.abs(np.load(held / f"GT_{item['path']}") - item["gt_image"][0].numpy()).max()))
            zero_out &= not saved[np.load(held / f"Mask_{item['path']}") != 1].any()
            if i == 0:
                masked_input = not np.allclose((G(cond)[0] * mask)[0, 0].numpy(), saved, atol=1e-5)
    check("Init_ = BRAN(condition × mask) × mask with the saved weights; GT_ = the dataset's gt; 0 off the mask",
          worst_init < 1e-5 and worst_gt == 0 and zero_out, f"Init_ {worst_init:.1e}, GT_ {worst_gt:.1e}")
    check("  negative control: BRAN on the unmasked condition gives another map", masked_input)

    dbg = work / "bran_ext_debug"
    status = P.main(["-c", str(path), "--out", str(dbg), "--workers", "0", "--device", "cpu", "--debug"])
    m = torch.load(dbg / "bran_G.pth", map_location="cpu", weights_only=False)["meta"]
    check("--debug: 1 epoch, 4 held-out maps, marked debug",
          status == 0 and m["debug"] and m["epochs_done"] == 1 and len(list((dbg / "heldout").glob("Init_*"))) == 4)
    return out / "bran_G.pth", dbg / "bran_G.pth", recipe


def check_guard(ckpt, debug_ckpt, recipe, work):
    print("\nthe BRAN predictor's checkpoint guard: models/bran_predictor.py")
    from models.bran_predictor import BranInitialPredictor
    good = torch.load(ckpt, map_location="cpu", weights_only=False)
    make = lambda p, **kw: BranInitialPredictor(**{**dict(checkpoint=p, in_channels=3, target="rcbf",  # noqa: E731
                                                         expect_fold=0, mask_cond=True, ext=BOTH, pretrain=recipe), **kw})
    pred = make(ckpt)
    prov = pred.load_pretrained()
    check("the good checkpoint loads strict, BatchNorm statistics included; provenance has its sha256",
          C.same_weights(pred, good["generator"]) and len(prov["sha256"]) == 64 and prov["epochs_done"] == 2)
    bads = {"target": {"target": "tmax"}, "fold": {"fold": 1}, "masked input": {"mask_cond": False},
            "loss (no ext)": {"ext": None}, "ext side": {"ext": LOW},
            "recipe": {"pretrain": dict(recipe, lr=1e-4)}, "channel count": {"in_channels": 4},
            "unfinished run": {"epochs_done": 1}, "debug run": {"debug": True}}
    for name, change in bads.items():
        p = work / f"bran_bad_{name.split()[0]}.pth"
        torch.save({"generator": good["generator"], "meta": dict(good["meta"], **change)}, p)
        check(f"  refuses another {name}", raises(make(p).load_pretrained, ValueError))
    check("  a debug run loads with allow_debug (the debug training job's)",
          not raises(make(debug_ckpt, allow_debug=True, pretrain=recipe).load_pretrained))
    check("  no checkpoint at all is refused, not trained from random weights",
          raises(make(None).load_pretrained, ValueError))


def check_network(cfg_both, ckpt, recipe, parts, root, split, gan_bpm):
    print("\nNetwork with BRAN: the masked predictor input and the extreme-loss terms")
    import models.loss
    from core.base_network import BaseNetwork
    from models import define_network
    from models.loss import ext_term, ext_weight
    from models.model import EMA
    from models.network import Network
    c = point_at(cfg_both, root, split, gan_bpm, recipe, ckpt)
    opt = c["model"]["which_networks"][0]
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["generator"]
    net = define_network(C.Log(), {"phase": "train"}, opt)
    check("after define_network, BRAN's weights and statistics are the pretrained ones",
          C.same_weights(net.initial_net, state))
    fresh = Network(**opt["args"])
    BaseNetwork.init_weights(fresh)
    check("  negative control: BaseNetwork's init alone overwrites them", not C.same_weights(fresh.initial_net, state))
    net.set_loss(models.loss.mse_loss)
    net.set_new_noise_schedule(device=torch.device("cpu"), phase="train")
    net.eval()
    b = holed_batch(parts["test"])
    gt, cond, mask, lm = b["gt_image"], b["cond_image"], b["mask"], b["loss_mask"]
    check("the batch has no-data holes in the envelope", bool((mask - lm).sum() > 0))
    seen = {"pred": [], "unet": []}
    net.initial_net.register_forward_pre_hook(lambda m, a: seen["pred"].append(a[0].clone()))
    net.denoise_fn.register_forward_pre_hook(lambda m, a: seen["unet"].append(a[0].clone()))
    captured = {}
    net.denoise_fn.register_forward_hook(lambda m, a, o: captured.__setitem__("noise_hat", o.detach().clone()))
    noise = torch.randn_like(gt)
    with torch.no_grad():
        torch.manual_seed(0)
        total = net(gt, cond, mask=mask, noise=noise, loss_mask=lm)
        terms = dict(net.loss_terms)
        check("forward: BRAN reads condition × mask; so does the U-Net",
              torch.equal(seen["pred"][-1], cond * mask) and torch.equal(seen["unet"][-1][:, :3], cond * mask))
        y_init = net.initial(cond, mask)[0]
        check("Network.initial = BRAN(condition × mask) × mask",
              torch.equal(y_init, net.initial_net(cond * mask)[0] * mask))
        net.initial_net.mask_cond = False
        net.initial(cond, mask)
        check("  negative control: with mask_cond off, BRAN reads the whole condition", torch.equal(seen["pred"][-1], cond))
        net.initial_net.mask_cond = True

        pe, de = net.predictor_ext, net.diffusion_ext
        nh = captured["noise_hat"]
        l1, mse = F.l1_loss(lm * gt, lm * y_init), F.mse_loss(lm * noise, lm * nh)
        ei = ext_term(y_init, gt, ext_weight(gt, pe["side"]), lm)
        ed = ext_term(nh, noise, ext_weight(gt, de["side"]), lm)
        check("the loss is loss_diff + 0.1 × loss_init, and the health log gets both extreme terms",
              torch.equal(total, terms["loss_diff"] + terms["loss_init"] * 0.1)
              and set(terms) == {"loss_diff", "loss_init", "loss_init_ext", "loss_diff_ext"})
        check("  loss_init = L1 + w · the extreme loss of BRAN's map",
              torch.allclose(terms["loss_init"], l1 + pe["weight"] * ei) and torch.allclose(terms["loss_init_ext"], ei))
        check("  loss_diff = MSE + w · mean(loss_mask · W(gt) · (ε − ε̂)²)",
              torch.allclose(terms["loss_diff"], mse + de["weight"] * ed) and torch.allclose(terms["loss_diff_ext"], ed))
        net.predictor_ext = net.diffusion_ext = None
        torch.manual_seed(0)
        plain = net(gt, cond, mask=mask, noise=noise, loss_mask=lm)
        check("with both off, the loss is MSE + 0.1 × L1 exactly, with no extra terms",
              torch.equal(plain, mse + l1 * 0.1) and set(net.loss_terms) == {"loss_diff", "loss_init"})
        check("  negative control: the extreme terms change the loss", not torch.equal(plain, total))
        net.predictor_ext, net.diffusion_ext = pe, de

        class Pointwise(torch.nn.Module):
            def forward(self, x, gammas, cond):
                return 0.5 * x[:, -1:] + 0.25 * x[:, :1]
        unet, net.denoise_fn = net.denoise_fn, Pointwise()
        hole, valid = (mask - lm).bool(), lm.bool()
        gt_hole, gt_valid = gt.clone(), gt.clone()
        gt_hole[hole], gt_valid[valid] = 0.9, 1 - gt_valid[valid]

        def loss(g):
            torch.manual_seed(0)
            return net(g, cond, mask=mask, noise=noise, loss_mask=lm).item()
        check("with the extreme terms on, gt in the no-data holes still doesn't change the loss", loss(gt) == loss(gt_hole))
        check("  and gt where the label is valid does", loss(gt) != loss(gt_valid))
        net.denoise_fn = unet

    net.train()
    ma = copy.deepcopy(net)
    before = {n: t.clone() for n, t in ma.named_buffers()}
    with torch.no_grad():
        net.initial_net(cond * mask)                 # train mode: BatchNorm's running statistics move
    EMA(beta=0.5).update_model_average(ma, net)
    cur, new = dict(net.named_buffers()), dict(ma.named_buffers())
    means = [n for n in cur if n.endswith("running_mean")]
    check("EMA: BRAN's BatchNorm statistics are averaged as the weights are",
          means and any(not torch.equal(before[n], cur[n]) for n in means)
          and all(torch.allclose(new[n], 0.5 * before[n] + 0.5 * cur[n]) for n in means))
    check("  the step counts are copied; buffers that match (the noise schedule) stay as they were",
          all(torch.equal(new[n], cur[n]) for n in new if n.endswith("num_batches_tracked"))
          and all(torch.equal(new[n], before[n]) for n in new if n in SCHEDULE))


def check_run_py(cfg_both, root, split, gan_bpm, ckpt, recipe, work, parts):
    print("\nrun.py end to end on the CPU, a BRAN arm with both extreme terms: train -d, then test -d")
    import argparse as ap
    import core.praser as Praser
    import models.network as network
    from models import define_network
    try:
        import npy2img  # noqa: F401
    except ImportError:
        sys.modules["npy2img"] = types.ModuleType("npy2img")
    import run
    import os

    c = point_at(cfg_both, root, split, gan_bpm, recipe, ckpt)
    c["path"]["base_dir"] = str(work / "experiments")
    for p in ("train", "test"):
        c["model"]["which_networks"][0]["args"]["beta_schedule"][p]["n_timestep"] = 5
    run_dir = work / "run_bran"
    run_dir.mkdir(exist_ok=True)
    original = network.Network.set_new_noise_schedule
    network.Network.set_new_noise_schedule = (
        lambda self, device=torch.device("cpu"), phase="train": original(self, device=device, phase=phase))
    cwd = os.getcwd()
    os.chdir(run_dir)

    def go(cfg, phase, name):
        path = run_dir / f"{name}.json"
        path.write_text(json.dumps(cfg, indent=1))
        o = Praser.parse(ap.Namespace(config=str(path), phase=phase, batch=None, gpu_ids=None, debug=True, port="21012"))
        o["world_size"] = 1
        run.main_worker(0, 1, o)
        return Path(o["path"]["experiments_root"])
    try:
        exp = go(c, "train", "train")
        ck = exp / "checkpoint"
        check("train: checkpoint 1 (Network, EMA, state)",
              all((ck / f"1{s}").is_file() for s in ("_Network.pth", "_Network_ema.pth", ".state")))
        log = (exp / "train.log").read_text()
        check("train.log names BRAN's checkpoint", "Initial predictor loaded from" in log and str(ckpt) in log)
        rows = C.read_csv(exp / "health/train.csv")
        check("health: train.csv has loss_init_ext and loss_diff_ext, finite",
              len(rows) == 1 and all(np.isfinite(float(rows[0][k])) for k in ("loss_diff", "loss_init", "loss_init_ext", "loss_diff_ext")),
              str(list(rows[0]) if rows else None)[:160])
        mon = C.read_csv(exp / "health/monitor.csv")
        check("health: the monitor pass scores the predictor alone too",
              len(mon) == 1 and np.isfinite(float(mon[0]["mae_predictor_rCBF"])))
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["generator"]
        saved = torch.load(ck / "1_Network.pth", map_location="cpu")
        ema = torch.load(ck / "1_Network_ema.pth", map_location="cpu")
        pred = {k[len("initial_net."):]: v for k, v in saved.items() if k.startswith("initial_net.")}
        check("train: BRAN was fine-tuned, its weights moved",
              pred.keys() == state.keys() and not all(torch.equal(pred[k], state[k]) for k in state))
        bn = [k for k in ema if k.startswith("initial_net.") and k.endswith("running_mean")]
        check("train: the EMA copy's BatchNorm statistics moved off the pretrained ones",
              bool(bn) and any(not torch.equal(ema[k], state[k[len("initial_net."):]]) for k in bn))

        t = copy.deepcopy(c)
        t["path"]["resume_state"] = str(ck / "1")
        t["model"]["which_networks"][0]["args"]["initial_predictor"]["checkpoint"] = None   # as the H100 test job sets it
        texp = go(t, "test", "test")
        res = texp / "results/test/0"
        outs = sorted(p.name[len("Out_"):] for p in res.glob("Out_*.npy"))
        inits = {p.name[len("Init_"):]: np.load(p) for p in res.glob("Init_*.npy")}
        masks = {p.name[len("Mask_"):]: np.load(p) for p in res.glob("Mask_*.npy")}
        n_test = c["debug"]["debug_split"]
        check(f"test, with no pretraining checkpoint: Out_ and Init_ for {n_test} slices, Init_ 0 off the mask",
              len(outs) == n_test and sorted(inits) == outs and all(not inits[k][masks[k] != 1].any() for k in inits))
        tnet = define_network(C.Log(), {"phase": "test"}, t["model"]["which_networks"][0])
        missing, unexpected = tnet.load_state_dict(ema, strict=False)
        check("test: the checkpoint covers every weight and statistic (strict=False skips only the schedule)",
              not missing and set(unexpected) <= SCHEDULE, f"missing {len(missing)}, unexpected {sorted(unexpected)[:4]}")
        tnet.eval()
        index = {parts["test"][i]["path"]: i for i in range(len(parts["test"]))}
        name = outs[0]
        item = parts["test"][index[name]]
        with torch.no_grad():
            mine = tnet.initial(item["cond_image"][None], item["mask"][None])[0][0, 0].numpy()
            whole = (tnet.initial_net(item["cond_image"][None])[0] * item["mask"][None])[0, 0].numpy()
        check("test: Init_ is the EMA BRAN on condition × mask", np.allclose(inits[name], mine, atol=1e-5),
              f"{float(np.abs(inits[name] - mine).max()):.1e}")
        check("  negative control: not BRAN on the whole condition", not np.allclose(inits[name], whole, atol=1e-5))
    finally:
        os.chdir(cwd)
        network.Network.set_new_noise_schedule = original


def check_configs():
    print("\nthe five arms' configs and condor/ctp_only_bran_ext.dag")
    import fnmatch
    from models.bran_predictor import PRETRAIN_DEFAULTS, normalize_ext, pretrain_recipe
    base = C.load_config(C.CONFIG_RCBF)
    allowed = ["name", "model.which_networks[[]0[]].args.initial_predictor*",
               "model.which_networks[[]0[]].args.diffusion_ext"]
    cfgs = {v: C.load_config(p) for v, p in CONFIGS.items()}
    blocks = {}
    for v, cfg in cfgs.items():
        other = [d for d in C.config_diff(base, cfg) if not any(fnmatch.fnmatchcase(d, a) for a in allowed)]
        check(f"{v}: differs from ctp_only_f13_rcbf_fold0.json only in name, initial_predictor and diffusion_ext",
              not other and cfg["name"] == f"ctp_only_f13_rcbf_{v}_fold0", str(other))
        a = cfg["model"]["which_networks"][0]["args"]
        ip = a["initial_predictor"]
        m = re.fullmatch(r"/radraid2/colivares/mrdpm_ctp_only/bran/(rcbf_[a-z0-9]+)/fold_0/bran_G\.pth", ip["checkpoint"])
        ext, diff, folder = EXPECT[v]
        check(f"  BRAN, 3 channels, rCBF, fold 0, masked input, the default recipe; BRAN {folder}, ext {ext}, diffusion {diff}",
              (ip["type"], ip["in_channels"], ip["target"], ip["expect_fold"], ip["mask_cond"]) == ("bran", 3, "rcbf", 0, True)
              and pretrain_recipe(ip.get("pretrain")) == pretrain_recipe(PRETRAIN_DEFAULTS)
              and m is not None and m.group(1) == folder
              and normalize_ext(ip.get("ext")) == normalize_ext(ext) and normalize_ext(a.get("diffusion_ext")) == normalize_ext(diff)
              and a["mask_unet_cond"] and a["zero_hole_residual"] and not a["unet"]["mirror_hs"])
        blocks.setdefault(folder, []).append((v, ip))
    check("arms that share a BRAN have the same initial_predictor block (one pretraining serves both)",
          all(all(ip == group[0][1] for _v, ip in group) for group in blocks.values()))
    dag = (REPO / "condor/ctp_only_bran_ext.dag").read_text()
    variants = set(re.findall(r'variant="([a-z0-9_]+)"', dag))
    brans = set(re.findall(r"--bran /radraid2/colivares/mrdpm_ctp_only/bran/([a-z0-9_]+)", dag))
    runs = set(re.findall(r'run="/radraid2/colivares/mrdpm_ctp_only/runs/train_ctp_only_f13_rcbf_([a-z0-9_]*?)_?fold0"', dag))
    check("the DAG's arms have configs; its --bran folders are the configs' BRANs; it tests every arm it trains",
          variants == set(VARIANTS) and brans == set(blocks)
          and runs == {v for v in VARIANTS if v != "bran_extlow"} | {""}, f"{sorted(variants)} {sorted(brans)} {sorted(runs)}")


def check_names():
    print("\nrun names with an arm: the core check's patterns and the test job's")
    import ctp_only_core_check as K
    tests = {"test_ctp_only_f13_rcbf_fold0_cv_261025_101010": ("rcbf", None, "0"),
             "test_ctp_only_f13_rcbf_bran_l1_fold0_cv_261025_101010": ("rcbf", "bran_l1", "0"),
             "test_ctp_only_f13_tmax_bran_ext_diffext_fold3_cv_261025_101010": ("tmax", "bran_ext_diffext", "3")}
    check("TEST_NAME: target, arm (None for today's MRDPM), fold",
          all(K.TEST_NAME.match(n) and K.TEST_NAME.match(n).groups() == g for n, g in tests.items()))
    check("  negative control: an ISLES 2018 test is not a CV test",
          K.TEST_NAME.match("test_ctp_only_f13_rcbf_bran_l1_fold0_isles18_261025_101010") is None)
    trains = {"train_ctp_only_f13_rcbf_fold0_261020_101010": ("rcbf", None, "fold0"),
              "train_ctp_only_f13_rcbf_bran_l1_diffext_fold0_261026_101010": ("rcbf", "bran_l1_diffext", "fold0")}
    check("TRAIN_NAME: the same for training runs",
          all(K.TRAIN_NAME.match(n) and K.TRAIN_NAME.match(n).groups() == g for n, g in trains.items()))
    check("map names: mrdpm for today's MRDPM, mrdpm_<arm> for an arm's", (K.map_name(None), K.map_name("bran_l1")) == ("mrdpm", "mrdpm_bran_l1"))
    line = next(l for l in (REPO / "condor/ctp_only_test_h100.sh").read_text().splitlines() if l.startswith("RUN_NAME="))
    got = {}
    for n in ("train_ctp_only_f13_rcbf_fold0_261020_101010", "train_ctp_only_f13_rcbf_bran_l1_fold0_261026_101010",
              "debug_ctp_only_f13_rcbf_bran_ext_diffext_fold2_261026_101010", "train_ctp_only_f13_rcbf_all_261020_101010"):
        r = subprocess.run(["bash", "-c", line + '\n[[ "$1" =~ $RUN_NAME ]] && echo "${BASH_REMATCH[2]}|${BASH_REMATCH[4]}|${BASH_REMATCH[5]}"',
                            "_", n], capture_output=True, text=True)
        got[n] = r.stdout.strip()
    check("the test job's RUN_NAME reads target | arm | fold",
          list(got.values()) == ["rcbf||fold0", "rcbf|bran_l1|fold0", "rcbf|bran_ext_diffext|fold2", "rcbf||all"], str(got))


def heredoc(script):
    """The job script's Python that writes the job config (the train job has another, before it)."""
    blocks = re.findall(r"<<'PYEOF'[^\n]*\n(.*?)\nPYEOF", (REPO / "condor" / script).read_text(), re.S)
    return next(b for b in blocks if "initial_predictor" in b)


def check_jobs(work, ckpt):
    print("\nthe H100 jobs' config rewriting, on fake stage and BRAN folders")
    stage = work / "stage"
    run = stage / "gan_run_rcbf" / "fold_1"
    run.mkdir(parents=True, exist_ok=True)
    (run / "split.json").write_text("{}")
    bran_root = work / "bran"
    for name in ("rcbf_l1", "debug_rcbf_l1"):
        (bran_root / name / "fold_1").mkdir(parents=True, exist_ok=True)
        (bran_root / name / "fold_1" / "bran_G.pth").write_bytes(name.encode())
    cfg = C.load_config(CONFIGS["bran_l1"])
    cfg["model"]["which_networks"][0]["args"]["initial_predictor"]["checkpoint"] = str(bran_root / "rcbf_l1/fold_0/bran_G.pth")
    src = work / "job_src.json"
    src.write_text(json.dumps(cfg))
    py = lambda code, *a: subprocess.run([sys.executable, "-", *map(str, a)], input=code, text=True,  # noqa: E731
                                         capture_output=True)
    train_py = heredoc("ctp_only_train_h100.sh")
    for mode, want in (("full", "rcbf_l1"), ("debug", "debug_rcbf_l1")):
        bran_in = work / f"bran_in_{mode}"
        bran_in.mkdir(exist_ok=True)
        dst = work / f"job_train_{mode}.json"
        r = py(train_py, src, dst, stage, work / "exp", 1, "", "rcbf", run, "bran_l1", mode, bran_in)
        out = json.loads(dst.read_text()) if r.returncode == 0 else {}
        ip = out.get("model", {}).get("which_networks", [{}])[0].get("args", {}).get("initial_predictor", {})
        check(f"train job, {mode}: copies {want}/fold_1's BRAN to /home, fold 1, allow_debug {mode == 'debug'}, the arm's name",
              r.returncode == 0 and ip.get("checkpoint") == str(bran_in / "bran_G.pth")
              and (bran_in / "bran_G.pth").read_bytes() == want.encode() and ip.get("expect_fold") == 1
              and ip.get("allow_debug") == (mode == "debug") and out.get("name") == "ctp_only_f13_rcbf_bran_l1_fold1"
              and out["datasets"]["train"]["which_dataset"]["args"]["split_json"] == f"{run}/split.json", r.stderr[-300:])
    r = py(train_py, src, work / "job_x.json", stage, work / "exp", 2, "", "rcbf", stage / "gan_run_rcbf/fold_2",
           "bran_l1", "full", work / "bran_in_full")
    check("  negative control: a fold with no pretrained BRAN stops the job", r.returncode != 0 and "no pretrained BRAN" in r.stderr)
    test_py = heredoc("ctp_only_test_h100.sh")
    dst = work / "job_test.json"
    r = py(test_py, src, dst, stage, work / "exp", 1, "/home/x/resume/200", "rcbf", run, "cv", "bran_l1")
    out = json.loads(dst.read_text()) if r.returncode == 0 else {}
    ip = out.get("model", {}).get("which_networks", [{}])[0].get("args", {}).get("initial_predictor", {})
    check("test job: BRAN's pretraining checkpoint isn't needed (None), fold 1, the arm's test name",
          r.returncode == 0 and ip.get("checkpoint", "x") is None and ip.get("expect_fold") == 1
          and out.get("name") == "ctp_only_f13_rcbf_bran_l1_fold1_cv", r.stderr[-300:])
    pre_py = heredoc("ctp_only_bran_pretrain.sh")
    for mode, want in (("full", "rcbf_l1"), ("debug", "debug_rcbf_l1")):
        dst = work / f"job_pre_{mode}.json"
        r = py(pre_py, src, dst, stage, 1, mode, run)
        out = json.loads(dst.read_text()) if r.returncode == 0 else {}
        ip = out.get("model", {}).get("which_networks", [{}])[0].get("args", {}).get("initial_predictor", {})
        check(f"pretrain job, {mode}: writes to bran/{want}/fold_1, the fold's split, fold 1",
              r.returncode == 0 and r.stdout.strip() == str(bran_root / want / "fold_1")
              and ip.get("checkpoint") == str(bran_root / want / "fold_1/bran_G.pth") and ip.get("expect_fold") == 1
              and out["datasets"]["test"]["which_dataset"]["args"]["split_json"] == f"{run}/split.json", r.stderr[-300:])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gan-bpm", type=Path, default=Path("/raid/colivares/github_repos/gan_bpm"))
    p.add_argument("--keep", type=Path, default=None, help="build the fixtures here and keep them")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    if str(args.gan_bpm) not in sys.path:
        sys.path.append(str(args.gan_bpm))      # appended, as MCTPDataset_mask does
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
    C.make_root(root)
    split = C.write_split(work / "run_k/fold_0/split.json",
                          [f"{d}-{i}" for d in C.MARKERS for i in range(3)], [f"{d}-3" for d in C.MARKERS])
    from data.dataset import MCTPDataset_mask
    data = dict(mctp_root=str(root), gan_bpm_root=str(args.gan_bpm), split_json=str(split), label_type="rcbf")
    parts = {q: MCTPDataset_mask(part=q, **data) for q in ("train", "monitor", "test")}

    check_arch()
    check_ext(args.gan_bpm)
    ckpt, debug_ckpt, recipe = check_pretrain(C.load_config(CONFIGS["bran_ext"]), root, split, args.gan_bpm, work, parts)
    check_guard(ckpt, debug_ckpt, recipe, work)
    check_network(C.load_config(CONFIGS["bran_ext_diffext"]), ckpt, recipe, parts, root, split, args.gan_bpm)
    check_run_py(C.load_config(CONFIGS["bran_ext_diffext"]), root, split, args.gan_bpm, ckpt, recipe, work, parts)
    check_configs()
    check_names()
    check_jobs(work, ckpt)

    n_fail = C.RESULTS.count(False)
    print(f"\n{len(C.RESULTS) - n_fail}/{len(C.RESULTS)} checks passed")
    if tmp is not None:
        tmp.cleanup()
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
