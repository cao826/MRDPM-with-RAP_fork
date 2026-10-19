#!/bin/bash
# Invoked by probe_box.sub. Read-only: prints what the image and the box have, writes nothing.
set -u

# The execute node's $HOME is not writable inside the container (gan_bpm condor/run_cv.sh).
export HOME="${_CONDOR_SCRATCH_DIR:-/tmp}"
export MPLCONFIGDIR="$HOME/.matplotlib"

GAN_BPM="/raid/colivares/github_repos/gan_bpm"
RUN="/radraid2/colivares/matrix/gan_bpm_tmax_f13uclav2_t12_adv0_k5"
DATA_ROOT="/raid/colivares/mctp_dataset_f13_isles24_ucla_v2"

echo "host: $(hostname)   date: $(date -Is)"
echo "gan_bpm checkout: $(git -C "$GAN_BPM" rev-parse --short HEAD 2>/dev/null || echo '<missing>')"
echo "==================================================================="

GAN_BPM="$GAN_BPM" RUN="$RUN" DATA_ROOT="$DATA_ROOT" python - <<'PY'
import hashlib, importlib, json, os, sys
from pathlib import Path

print(f"python {sys.version.split()[0]}")

print("\n--- imports (MRDPM needs these at import time unless marked) ---")
MODULES = [
    ("torch", "everything"),
    ("torchvision", "data/dataset.py"),
    ("numpy", "everything"),
    ("PIL", "data/dataset.py"),
    ("tqdm", "models/model.py, models/network.py"),
    ("einops", "models/region_net.py"),
    ("dropblock", "models/region_net.py"),
    ("torchviz", "models/network.py (unused import)"),
    ("pydot_ng", "models/network.py (unused import)"),
    ("cv2", "data/util/mask.py, via data/dataset.py"),
    ("matplotlib", "npy2img.py"),
    ("tensorboardX", "optional: core/logger.py, falls back to torch.utils.tensorboard"),
    ("torch.utils.tensorboard", "optional: the fallback"),
    ("scipy", "gan_bpm mctp/brain_mask.py"),
    ("pandas", "gan_bpm mctp/dataset.py"),
    ("pyarrow", "gan_bpm mctp/dataset.py"),
]
for name, why in MODULES:
    try:
        m = importlib.import_module(name)
        print(f"  OK       {name:<24} {getattr(m, '__version__', '')}")
    except Exception as exc:
        print(f"  MISSING  {name:<24} {type(exc).__name__}: {exc}  [{why}]")

import torch
print(f"  torch {torch.__version__}, CUDA build {torch.version.cuda}")

gan_bpm, run, data_root = (Path(os.environ[k]) for k in ("GAN_BPM", "RUN", "DATA_ROOT"))

print("\n--- gan_bpm checkout ---")
for rel in ("models/generator.py", "mctp/dataset.py", "mctp/brain_mask.py", "mctp/transforms.py"):
    print(f"  {'OK     ' if (gan_bpm / rel).is_file() else 'MISSING'}  {gan_bpm / rel}")

print("\n--- fold-0 checkpoint ---")
ckpt_path = run / "fold_0/checkpoints/best.pt"
if not ckpt_path.is_file():
    print(f"  MISSING  {ckpt_path}")
    print(f"  {run}/fold_0 has: {sorted(p.name for p in (run / 'fold_0').glob('*')) if (run / 'fold_0').is_dir() else '<no dir>'}")
else:
    h = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    print(f"  path     {ckpt_path}")
    print(f"  size     {ckpt_path.stat().st_size / 2**20:.1f} MiB   sha256 {h.hexdigest()}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"  keys     {sorted(ckpt)}")
    print(f"  epoch    {ckpt.get('epoch')}   {ckpt.get('select_metric')} = {ckpt.get('select_value')}")
    cfg = ckpt.get("config") or {}
    for k in ("target", "tmax_ceiling", "n_phases", "label_transform", "lambda_adv",
              "fold", "seed", "datasets", "split_file", "data_root"):
        print(f"  config.{k:<16} {cfg.get(k, '<absent>')!r}")
    G = ckpt["generator"]
    prefixes = {}
    for key, t in G.items():
        p = ".".join(key.split(".")[:2])
        prefixes[p] = prefixes.get(p, 0) + t.numel()
    print(f"  generator: {len(G)} tensors, {sum(prefixes.values()):,} params, by module:")
    for p, n in prefixes.items():
        print(f"    {p:<36} {n:>10,}")

print("\n--- fold-0 split.json ---")
split_path = run / "fold_0/split.json"
if not split_path.is_file():
    print(f"  MISSING  {split_path}")
else:
    split = json.loads(split_path.read_text())
    for k, v in split.items():
        print(f"  {k:<24} {f'list of {len(v)}' if isinstance(v, list) else repr(v)[:100]}")
    tr, va = set(split.get("train_case_ids", [])), set(split.get("val_case_ids", []))
    print(f"  train ∩ val: {len(tr & va)} cases")

print("\n--- data root ---")
if not data_root.is_dir():
    print(f"  MISSING  {data_root}")
else:
    entries = sorted(data_root.iterdir())
    print(f"  {data_root}: {len(entries)} entries")
    for p in entries[:15]:
        print(f"    {p.name}{'/' if p.is_dir() else ''}")
PY
status=$?
echo "==================================================================="
echo "probe exited $status"
exit "$status"
