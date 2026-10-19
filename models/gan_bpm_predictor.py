"""gan_bpm's generator as MRDPM's initial predictor (ctp_only_plan.md §8).

`GanBpmInitialPredictor` puts gan_bpm's `Generator` (3 CTP phases -> Tmax) behind BRAN's
interface, in MRDPM's [0, 1] space:

    cond01 (N, 3, H, W) -> y01 (N, 1, H, W),
                           hs = [(N, 64, H/2, W/2), (N, 128, H/4, W/4), (N, 256, H/8, W/8)]

The generator reads the whole head, unmasked, as it did in gan_bpm (§7.1). `Network` masks
only the diffusion U-Net's copy of the condition (`mask_unet_cond`).
"""
import hashlib
import importlib.util
import io
from pathlib import Path

import torch
import torch.nn as nn

# What the checkpoint's own config must say. Under anything else `(y + 1) / 2` is not
# Tmax / 12 s, e.g. a 40 s ceiling (`tmax_ceiling: None`) would make it Tmax / 40 (§8.3).
EXPECTED_CONFIG = {"target": "tmax", "tmax_ceiling": 12.0, "n_phases": 3, "label_transform": "linear"}


def load_generator_module(gan_bpm_root):
    """gan_bpm's `models/generator.py`, imported by file path.

    MRDPM has its own `models` package, so `import models.generator` would find the wrong
    one. generator.py imports only torch.
    """
    path = Path(gan_bpm_root) / "models" / "generator.py"
    if not path.is_file():
        raise FileNotFoundError(f"no gan_bpm generator at {path}")
    spec = importlib.util.spec_from_file_location("gan_bpm_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_checkpoint(path, expect=EXPECTED_CONFIG):
    """`(generator state_dict, provenance)` from a gan_bpm checkpoint, refusing a mismatched one.

    The sha256 is of the bytes that were loaded, not of a second read of the file.
    """
    path = Path(path)
    raw = path.read_bytes()
    # weights_only=False, as gan_bpm's own load_checkpoint: the checkpoint holds its config dict.
    ckpt = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or {}
    bad = {k: cfg.get(k) for k, v in expect.items() if cfg.get(k) != v}
    if bad:
        raise ValueError(f"refusing gan_bpm checkpoint {path}: its config has {bad}, "
                         f"the initial predictor needs {expect}")
    provenance = {"checkpoint": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                  "epoch": ckpt.get("epoch"),
                  **{k: cfg.get(k) for k in ("target", "tmax_ceiling", "fold", "lambda_adv", "data_root")}}
    return ckpt["generator"], provenance


class GanBpmInitialPredictor(nn.Module):
    def __init__(self, gan_bpm_root, checkpoint, n_phases=3, expect_fold=None):
        super().__init__()
        self.checkpoint = str(checkpoint)
        self.expect = dict(EXPECTED_CONFIG, n_phases=n_phases)
        if expect_fold is not None:
            self.expect["fold"] = expect_fold
        self.G = load_generator_module(gan_bpm_root).Generator(n_phases=n_phases)
        self.load_pretrained()

    def load_pretrained(self):
        """Strict-load the checkpoint's generator weights and return its provenance.

        `Network.init_weights` calls this again after `BaseNetwork.init_weights` has
        re-initialized every conv, this one's included (§8.4). It is deliberately not
        called `init_weights`: BaseNetwork calls a child's `init_weights(init_type, gain)`.
        """
        state, self.provenance = read_checkpoint(self.checkpoint, self.expect)
        self.G.load_state_dict(state, strict=True)
        return self.provenance

    def forward_gan_bpm(self, x):
        """`Generator.forward`, step for step, keeping three decoder features.

        x (N, 1, 3, H, W) in [-1, 1] -> (y (N, 1, H, W) in [-1, 1], hs). `y` must stay
        bitwise equal to `self.G(x)` (scripts/ctp_only_checks.py).
        """
        u = self.G.unet
        stem = self.G.temporal_stem(x)                       # 16 @ H/2
        skip1 = u.encoder1(stem)
        skip2 = u.encoder2(skip1)
        skip3 = u.encoder3(skip2)
        skip4 = u.encoder4(skip3)
        skip5 = u.encoder5(skip4)
        transition = u.encoder6(skip5)
        up1 = torch.cat([u.decoder1(transition), skip5], 1)
        up2 = torch.cat([u.decoder2(up1), skip4], 1)
        up3 = torch.cat([u.decoder3(up2), skip3], 1)
        up4 = torch.cat([u.decoder4(up3), skip2], 1)          # 128 @ H/8
        up5 = torch.cat([u.decoder5(up4), skip1], 1)          # 64 @ H/4
        d6 = u.decoder6(up5)                                  # 16 @ H/2
        y = u.output_head(d6)
        # BRAN's u7/u6/u5 are 64/128/256 wide and these are half that. Duplicating each one
        # fits cond_resample's widths without adding untrained weights (§8.1).
        hs = [torch.cat([f, f], 1) for f in (torch.cat([d6, stem], 1), up5, up4)]
        return y, hs

    def forward(self, cond01):
        y, hs = self.forward_gan_bpm((2.0 * cond01 - 1.0).unsqueeze(1))
        return (y + 1.0) / 2.0, hs
