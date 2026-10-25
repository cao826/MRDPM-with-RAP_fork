"""How much of gan_bpm's generator gradient the extreme-value loss L_ext carries, at the trained
fold models (best.pt) and at initialization. Carlos asked on 2026-10-25, after the core check
(ctp_only_decisions.md #28, #29) found gan_bpm calling about half the label's core with
λ_ext = 50 next to λ_pix = 100.

gan_bpm's scripts/measure_grad_scope.py measures the same norms, but it builds its own default
config, and on f13uclav2 it was only run at initialization (to set WGAN-GP's λ_adv). This
script rebuilds each fold's training setup from the run's own config.json (losses, masks,
split, batch size), loads that fold's best.pt, and takes the supervised terms the way
measure_grad_scope.measure_batch does, with gan_bpm's own loss functions and its `flat_grad`,
read-only. Per training batch, over the generator's parameters:

  ||λ_i g_i||    each term's weighted gradient norm (pix, ext, bg)
  norm share     ||λ_i g_i|| / ||g||, with g = λ_pix g_pix + λ_ext g_ext + λ_bg g_bg, the step G
                 takes (a λ_adv = 0 run has no other term)
  step share     (λ_i g_i · g) / ||g||²: how much of the step along g term i supplies. The three
                 add up to 1; a negative share pulls against the step.
  cos(pix, ext)  1: L_ext asks for what L_pix asks for; 0: something else; −1: the opposite
  loss share     λ_i L_i over the weighted total: what the loss values alone would suggest

Medians over --batches batches per fold. Check, once per setting: the assembled g equals a
direct backward of the weighted sum (measure_grad_scope's own linearity check).

    python scripts/ctp_only_ext_share.py [--target rcbf] [--recipe adv0] [--fold 0] [--device cpu]
    python scripts/ctp_only_ext_share.py --run-dir <a run_cv folder> --checkpoint last.pt --fold 0 --no-init
"""
import argparse
import dataclasses
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctp_only_isles18_maps as M                                       # noqa: E402

CV_ROOT = Path("/raid/colivares/mctp_dataset_f13_isles24_ucla_v2")
OUT = Path("/radraid2/colivares/mrdpm_ctp_only/ext_share")
TERMS = ("pix", "ext", "bg")
LINEARITY_TOL = 1e-3         # measure_grad_scope's own threshold


def measure(experiment, batch, flat_grad, *, check=False):
    """One batch: every supervised term's gradient at G from one forward pass."""
    import torch                                                         # noqa: PLC0415
    import training                                                      # noqa: PLC0415
    from mctp.transforms import split_mask                               # noqa: PLC0415
    from models.losses import l_bg, l_ext, l_pix                         # noqa: PLC0415

    a, b, m = training._to_device(batch, experiment.device)
    model = experiment.model
    support, ext_gate, envelope = split_mask(m)
    if not getattr(model, "use_ext_gate", True):
        ext_gate = None                       # as GanBpm.generator_losses does
    params = [p for p in model.generator.parameters() if p.requires_grad]
    fake = model.generate(a)
    lam = {"pix": float(model.lambda_pix), "ext": float(model.lambda_ext),
           "bg": float(getattr(model, "lambda_bg", 0.0))}
    terms = {"pix": l_pix(fake, b, support=support),
             "ext": l_ext(fake=fake, real=b, support=support, ext_gate=ext_gate)}
    if envelope is not None and lam["bg"]:   # as generator_losses: only when weighted
        terms["bg"] = l_bg(fake, envelope)
    g = {k: lam[k] * flat_grad(v, params).double() for k, v in terms.items()}
    total = sum(g.values())
    tt = float(total @ total)
    norm = tt ** 0.5
    weighted_loss = {k: lam[k] * float(v.detach()) for k, v in terms.items()}
    row = {"n": int(a.shape[0]), "norm_total": norm}
    for k in TERMS:
        if k not in g:
            continue
        n_k = float(torch.linalg.vector_norm(g[k]))
        row[f"norm_{k}"] = n_k
        row[f"norm_share_{k}"] = n_k / max(norm, 1e-30)
        row[f"step_share_{k}"] = float(g[k] @ total) / max(tt, 1e-30)
        row[f"loss_share_{k}"] = weighted_loss[k] / max(sum(weighted_loss.values()), 1e-30)
    row["cos_pix_ext"] = float(g["pix"] @ g["ext"]) / max(
        float(torch.linalg.vector_norm(g["pix"]) * torch.linalg.vector_norm(g["ext"])), 1e-30)
    if check:
        combined = sum(lam[k] * v for k, v in terms.items())
        direct = float(torch.linalg.vector_norm(flat_grad(combined, params).double()))
        row["linearity_rel_error"] = abs(direct - norm) / max(direct, 1e-30)
    return row


def measure_setting(experiment, flat_grad, batches, label, fold):
    rows = []
    for batch in experiment.train_loader:
        if len(rows) >= batches:
            break
        if batch is None:                     # gan_bpm's collate dropped every sample
            continue
        row = measure(experiment, batch, flat_grad, check=not rows)
        rows.append({"at": label, "fold": fold, "batch": len(rows) + 1, **row})
        print(f"  {label} fold {fold} batch {len(rows)}: norm share ext {row['norm_share_ext']:.3f}, "
              f"step share ext {row['step_share_ext']:.3f}, cos(pix, ext) {row['cos_pix_ext']:+.3f}", flush=True)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--target", choices=M.TARGETS, default="rcbf")
    p.add_argument("--recipe", choices=M.RECIPES, default="adv0")
    p.add_argument("--run-dir", type=Path, help="any gan_bpm run_cv folder instead of the matrix's "
                   "<target>/<recipe> run (e.g. gan_bpm's λ_ext pilots)")
    p.add_argument("--checkpoint", default="best.pt", help="each fold's checkpoint (best.pt; last.pt)")
    p.add_argument("--fold", action="append", type=int, choices=M.FOLDS, help="default: all five")
    p.add_argument("--batches", type=int, default=8, help="batches per fold (measure_grad_scope's 8)")
    p.add_argument("--no-init", action="store_true", help="skip the reading at initialization")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--gan-bpm", type=Path, default=M.GAN_BPM, help="gan_bpm's checkout, read only")
    p.add_argument("--cv-root", type=Path, default=CV_ROOT)
    p.add_argument("--matrix", type=Path, default=M.MATRIX, help="gan_bpm's runs")
    p.add_argument("--out", type=Path, help=f"default: {OUT}/<date_time>_<target>_<recipe>/")
    args = p.parse_args(argv)
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    tag = args.run_dir.name if args.run_dir else f"{args.target}_{args.recipe}"
    out = args.out or OUT / f"{stamp}_{tag}"
    try:
        H, _S = M.load_gan_bpm(args.gan_bpm)
    except M.Blocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    import torch                                                         # noqa: PLC0415
    import training                                                      # noqa: PLC0415
    spec = importlib.util.spec_from_file_location("measure_grad_scope",
                                                  args.gan_bpm / "scripts" / "measure_grad_scope.py")
    scope = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scope)                                       # gan_bpm's flat_grad, unchanged

    run_dir = args.run_dir.resolve() if args.run_dir else args.matrix / M.gan_run(args.target, args.recipe)
    base = training.TrainingConfig.from_json(run_dir / "config.json")
    found = H.recipe_of(json.loads((run_dir / "config.json").read_text()))
    if args.run_dir is None and (base.target, found) != (args.target, args.recipe):
        print(f"BLOCKED: {run_dir.name} is {base.target}/{found}, not {args.target}/{args.recipe}", file=sys.stderr)
        return 2
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
                          else "cpu")
    print(f"{run_dir.name}: λ_pix {base.lambda_pix:g}, λ_ext {base.lambda_ext:g}, λ_bg {base.lambda_bg:g}, "
          f"λ_adv {base.lambda_adv:g}; ext_gate {base.ext_gate}, use_loss_mask {base.use_loss_mask}, "
          f"batch {base.batch_size}, split {base.split_file or '<from seed>'}; data {args.cv_root}; device {device}")

    rows = []
    settings = [(k, "trained") for k in (args.fold or M.FOLDS)] + ([] if args.no_init else [(0, "init")])
    for k, at in settings:
        cfg = dataclasses.replace(base, data_root=str(args.cv_root), fold=k,
                                  num_workers=args.num_workers).validate()
        experiment = training.build_experiment(cfg, device)
        label = "initialization"
        if at == "trained":
            epoch = training.load_checkpoint(run_dir / f"fold_{k}" / "checkpoints" / args.checkpoint, experiment)
            label = f"{args.checkpoint} (epoch {epoch})"
        experiment.model.train()             # as measure_grad_scope: the mode G trains in
        rows += measure_setting(experiment, scope.flat_grad, args.batches, label, k)

    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_batch.csv", index=False)
    cols = [c for c in ("norm_share_pix", "norm_share_ext", "norm_share_bg", "step_share_pix",
                        "step_share_ext", "step_share_bg", "cos_pix_ext", "loss_share_ext") if c in df]
    table = df.groupby(["at", "fold"], sort=False)[cols].median().round(3)
    lines = [f"{run_dir.name}, {stamp}: medians over {args.batches} training batches per row",
             f"λ_pix {base.lambda_pix:g}, λ_ext {base.lambda_ext:g}, λ_bg {base.lambda_bg:g}, λ_adv {base.lambda_adv:g}; "
             f"device {device}", "", table.to_string(), "",
             "norm share: ||λ_i g_i|| / ||g||; step share: (λ_i g_i · g) / ||g||², the three add up to 1;",
             "cos(pix, ext): do L_pix and L_ext ask for the same change; loss share: λ_ext L_ext / weighted total"]
    worst = float(df["linearity_rel_error"].max())
    sums = df[[c for c in df if c.startswith("step_share_")]].sum(axis=1)
    ok = worst <= LINEARITY_TOL and float((sums - 1).abs().max()) < 1e-6
    lines += ["", f"{'PASS' if worst <= LINEARITY_TOL else 'FAIL'}: assembled g = a direct backward of the "
                  f"weighted sum, worst relative error {worst:.2e} (≤ {LINEARITY_TOL:g})",
              f"{'PASS' if float((sums - 1).abs().max()) < 1e-6 else 'FAIL'}: step shares add up to 1 "
              f"(worst |sum − 1| {float((sums - 1).abs().max()):.1e})"]
    text = "\n".join(lines)
    print("\n" + text)
    (out / "summary.txt").write_text(text + "\n")
    print(f"\nwrote {out}/summary.txt and per_batch.csv")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
