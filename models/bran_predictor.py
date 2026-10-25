"""BRAN, the paper's own initial predictor (§2.3, Fig. 3), on the CTP-only condition (bran_ext_plan.md §2).

`BranInitialPredictor` is `region_net.Unetgenerator_256` itself, built for `in_channels`
condition channels (3 CTP phases; upstream's 4 included NCCT). Its state_dict keys are upstream's,
and it returns BRAN's own `(y, hs = [u7, u6, u5])`, 64 / 128 / 256 channels at H/2, H/4, H/8,
the widths `cond_resample` was built for.

Its weights come from scripts/pretrain_bran.py: paper Eq. 1 (L1 alone) or, with `ext`, L1 plus
Zeng's extreme loss. `Network.init_weights` loads them in the train phase only, after
BaseNetwork has re-initialized every conv. In the test phase the MRDPM checkpoint carries the
fine-tuned weights, and the pretraining checkpoint is never read.

The checkpoint's `meta` must match what the config asks for (target, fold, condition channels,
masked input, the loss and the pretraining recipe), or it is refused: a predictor pretrained
one way and fine-tuned under another config's name would make the arms' comparison meaningless.
"""
import hashlib
import io
from pathlib import Path

import torch

from .loss import EXT_SIDES
from .region_net import Unetgenerator_256

NGF = 32        # define_G's: cond_resample's widths (64/128/256) and blocks 5/8/11 assume it

# The upstream README: "trained simply for 50 epochs using Adam with linear decay to reduce the
# learning rate to zero". The rest is pix2pix's defaults, which BRAN's code comes from (define_G's
# signature, the `net_G.pth` name): lr 2e-4, betas (0.5, 0.999), N(0, 0.02) init. Batch 4 is the
# paper's diffusion batch, so the BatchNorm statistics are the ones the joint stage trains with.
PRETRAIN_DEFAULTS = {"epochs": 50, "batch_size": 4, "lr": 2e-4, "betas": [0.5, 0.999],
                     "init": "normal", "init_gain": 0.02}


def pretrain_recipe(pretrain=None):
    """The pretraining recipe with PRETRAIN_DEFAULTS filled in, so two configs that mean the same
    recipe compare equal."""
    pretrain = dict(pretrain or {})
    unknown = set(pretrain) - set(PRETRAIN_DEFAULTS)
    if unknown:
        raise ValueError('unknown pretrain keys {}; known: {}'.format(sorted(unknown), sorted(PRETRAIN_DEFAULTS)))
    r = dict(PRETRAIN_DEFAULTS, **pretrain)
    return {"epochs": int(r["epochs"]), "batch_size": int(r["batch_size"]), "lr": float(r["lr"]),
            "betas": [float(b) for b in r["betas"]], "init": str(r["init"]), "init_gain": float(r["init_gain"])}


def normalize_ext(ext):
    """None (plain L1) or {"weight": w > 0, "side": one of EXT_SIDES}."""
    if not ext:
        return None
    unknown = set(ext) - {"weight", "side"}
    if unknown:
        raise ValueError('unknown ext keys {}; known: weight, side'.format(sorted(unknown)))
    weight, side = float(ext.get("weight", 0.0)), ext.get("side", "both")
    if weight < 0 or side not in EXT_SIDES:
        raise ValueError('ext needs weight >= 0 and side in {}, got {}'.format(EXT_SIDES, ext))
    return None if weight == 0 else {"weight": weight, "side": side}


def expected_meta(target, in_channels, mask_cond, ext, pretrain, expect_fold=None):
    expect = {"target": target, "in_channels": int(in_channels), "ngf": NGF, "mask_cond": bool(mask_cond),
              "ext": normalize_ext(ext), "pretrain": pretrain_recipe(pretrain)}
    if expect_fold is not None:
        expect["fold"] = int(expect_fold)
    return expect


def build_bran(in_channels):
    return Unetgenerator_256(ngf=NGF, cond_channels=in_channels)


def read_checkpoint(path, expect, allow_debug=False):
    """`(state_dict, provenance)` from scripts/pretrain_bran.py's `bran_G.pth`, refusing one whose
    meta disagrees with `expect`, or that didn't finish its recipe (a debug run) unless
    `allow_debug`. The sha256 is of the bytes that were loaded."""
    path = Path(path)
    raw = path.read_bytes()
    ckpt = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    meta = ckpt.get("meta") or {}
    bad = {k: meta.get(k) for k, v in expect.items() if meta.get(k) != v}
    if bad:
        raise ValueError("refusing BRAN checkpoint {}: its meta has {}, the config needs {}".format(
            path, bad, {k: expect[k] for k in bad}))
    finished = meta.get("epochs_done") == expect["pretrain"]["epochs"] and not meta.get("debug")
    if not finished and not allow_debug:
        raise ValueError("refusing BRAN checkpoint {}: a debug or unfinished pretraining ({} of {} epochs, "
                         "debug {})".format(path, meta.get("epochs_done"), expect["pretrain"]["epochs"],
                                            meta.get("debug")))
    provenance = {"checkpoint": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                  **{k: meta.get(k) for k in ("epochs_done", "debug", "target", "fold", "mask_cond", "ext",
                                              "split_json", "mrdpm_commit")}}
    return ckpt["generator"], provenance


class BranInitialPredictor(Unetgenerator_256):
    def __init__(self, checkpoint=None, in_channels=3, target="tmax", expect_fold=None, mask_cond=False,
                 ext=None, pretrain=None, allow_debug=False):
        super().__init__(ngf=NGF, cond_channels=in_channels)
        self.checkpoint = None if checkpoint is None else str(checkpoint)
        # Read by Network.initial: BRAN gets the condition × mask, the paper's skull-stripped input
        self.mask_cond = bool(mask_cond)
        self.allow_debug = bool(allow_debug)
        self.expect = expected_meta(target, in_channels, mask_cond, ext, pretrain, expect_fold)

    def load_pretrained(self):
        """Strict-load the pretrained weights (BatchNorm statistics included); returns the provenance.

        Network.init_weights calls it after BaseNetwork.init_weights. Not named `init_weights`:
        BaseNetwork calls a child's `init_weights(init_type, gain)`."""
        if self.checkpoint is None:
            raise ValueError("BRAN has no pretraining checkpoint: it would start from random weights")
        state, self.provenance = read_checkpoint(self.checkpoint, self.expect, self.allow_debug)
        self.load_state_dict(state, strict=True)
        return self.provenance
