# ctp-only plan: MRDPM on f13 CTP phases, with gan_bpm's generator as the initial predictor

**Status (2026-10-18): proposal. Step 0 (box prerequisites, §13) is checked; nothing else is
built.** Branch `ctp-only`. This doc collects what
was read and decided in conversation so far. Items marked **(verified)** were checked against
code or data; items marked **(proposal)** wait for Carlos's agreement.

---

## 1. Goal and first run

Build a version of MRDPM (Cai et al., MICCAI 2024, `1246_paper.pdf`) that conditions on
**3 CTP phases only, with no NCCT**. It should train on the public + UCLA mCTP data with as
few model edits as possible and **without changing the label range MRDPM expects**.

**First run (proposal):**

| | value |
|---|---|
| target | Tmax |
| data | f13 (`msu_f13_30Sept2026`), ISLES'24 + UCLA EVT + UCLA negative, UCLA v2 labels |
| root (box) | `/raid/colivares/mctp_dataset_f13_isles24_ucla_v2` |
| fold | 0 of gan_bpm's 5-fold split `splits/f13_isles24_ucla_k5.json` |
| initial predictor | gan_bpm λ_adv = 0 Tmax generator, **fold 0**, 12 s ceiling: `/radraid2/colivares/matrix/gan_bpm_tmax_f13uclav2_t12_adv0_k5/fold_0/checkpoints/best.pt` |
| predictor training | fine-tuned jointly, λ = 0.1, as in the paper (§2.1, Eq. 4) |
| resolution | 512 × 512 |
| where | the GPU box (MRDPM needs CUDA, and UCLA data is only on the desktop and the box) |

Why fold 0 and not the all-data model: see §8.5.

---

## 2. What the paper and the code do (verified)

- **Two-stage model.** An initial predictor `Iθ` (BRAN) maps the condition `y` to a coarse map
  `x_init`. The diffusion U-Net learns the residual `x_gt − x_init`, only inside a brain mask.
  The output is `x̂_0 + x_init`.
- **The predictor is pre-trained** (paper §2.1, Eq. 1) with an L1 loss before diffusion
  training, *"to ensure the next diffusion model can be steadily trained"*. It is then
  fine-tuned jointly, with `L = L_diff + λ·L_init` and λ = 0.1 (Eq. 4).
  - No weights or pretraining script were released.
  - The code loads `./tmax_net_G.pth` if it exists (`core/base_network.py`). Otherwise it
    silently starts BRAN from random weights.
  - The README says the pretraining was 50 epochs of Adam with the learning rate decayed
    linearly to 0.
- **Gradients into the predictor.** The predictor gets gradients from two places:
  - directly from `0.1·L1(y_0, y_initial)`;
  - from `L_diff`, through `y_initial` → the residual → the noisy U-Net input.

  Only the feature path into the asymmetry module is `.detach()`ed (Fig. 2's "gradient stop").
- **Multi-scale Asymmetry Prior (MAP).** BRAN's decoder features at 1/2, 1/4 and 1/8
  resolution, each joined with a mirrored copy of itself (`torch.flip(cond, [2])`), are fused
  into the U-Net before input blocks 5, 8 and 11 (`models/guided_diffusion_modules/unet.py`,
  `cond_resample`).
- **Masks.** Noise is added only inside the brain mask. The loss is restricted to it
  (Eq. 3). During sampling, `y_0` is pasted back outside it. The paper derives the mask
  *"using morphological operations from mCTA images"*.
- **The paper's data:** 256 × 256 at 0.625 mm, skull-stripped and centered on the brain.
  CBF/CBV/Tmax were "normalized to" [0, 200] / [0, 20] / [0, 30].

---

## 3. The data: f13 (verified)

**The protocol** (`mctp_ready.py` `FIXED_OFFSET_PROTOCOLS["msu_f13_30Sept2026"]`, contract
§12):

```
t1 = aif_ttp_s        # arterial
t2 = t1 + 15 s        # equilibrium
t3 = t2 + 4 s         # venous (t1 + 19 s)
```

- There is no VOF dependency and no random draw.
- `max_clip_s = 0`: a case whose venous phase lands past the last frame is **skipped**, never
  clamped.
- The case gate is `aif_timing_gate`.
- Phases are evaluated by PCHIP on a 1 Hz grid.

**On disk**, per slice (contract §5, §12, §13):

- `sliceNNN_input.npy`: `(3, 512, 512)` float32, **raw HU**, phases on axis 0.
- `sliceNNN_label.npy`: `(512, 512)` float32, **raw seconds** for Tmax.
- Orientation is **RAS**, with the slice taken as `vol[:, :, z]`. So array axis 0 is R/L and
  axis 1 is A/P.
- The builder does not normalize, mask or split anything.

**Label quirks that matter here:**

- **ISLES'24 Tmax** marks background as **−30 s**. Inside gan_bpm's brain envelope, values
  below −0.5 make up 7.5 % of voxels (`measure_label_zeros.py` on the v2 root).
- **UCLA v2** writes **−1** wherever RAPID painted no color. That is about 10 % of the brain,
  for EVT and negative CTs alike.
- **UCLA Tmax stops at 12 s** and comes in ~1 s steps. gan_bpm therefore clips every dataset
  at **12 s** (D-97).

**The combined root.** `scripts/combine_mctp_roots.py` built it on the box, 2026-10-14, for both
labels:

| | ISLES'24 | UCLA EVT | UCLA negative |
|---|---|---|---|
| cases | 100 | 267 | 76 |
| slices | 1,841 | 1,612 | 471 |

The six all-uncolored EVT slices were dropped. `case_id` and `patient_id` are 1:1 in all three
datasets, but splits still group by `patient_id`. ISLES 2018 is not in this root: it is the
external test set (gan_bpm `final_models_plan.md` Part F).

---

## 4. What MRDPM expects (verified, `data/dataset.py:NumpyDataset_mask`)

`__getitem__` returns this dict, and `models/model.py:Palette.set_input` reads it:

| key | shape | value |
|---|---|---|
| `gt_image` | `(1, H, W)` | label clamped to **[0, 1]**, then `× mask` (0 outside the brain) |
| `cond_image` | `(C, H, W)` | condition clamped to **[0, 1]**, then `× mask` |
| `mask` | `(1, H, W)` | binary brain mask |
| `mask_image` | `(1, H, W)` | `gt·(1 − mask) + mask`, for visualisation only |
| `path` | str | output file stem. It must end in `.npy` for `VisualWriter.save_images` to take the `np.save` branch |

The model side ties these together:

- `unet.in_channel = C + 1`, because the U-Net input is `cat(cond, y_noisy)`.
- The diffusion runs on `gt − y_initial`. `clip_denoised` clamps the predicted residual to
  [−1, 1].

---

## 5. Our data in MRDPM's format, with the range unchanged

| MRDPM key | from our data | relation to gan_bpm (verified in `mctp/transforms.py`, `mctp/label_scale.py`) |
|---|---|---|
| `gt_image` | `clip(Tmax, 0, 12) / 12 × mask` | gan_bpm's label is `y = clip(Tmax, 0, 12)/6 − 1` (12 s runs), so `(y + 1)/2` equals this **exactly** |
| `cond_image` | `(clip(HU, 10, 325) − 10) / 315`, per phase, `× mask` | gan_bpm's input is `norm_to_interval(HU, 10, 325)` = the same value `× 2 − 1`, so the generator's input is exactly `2·cond − 1` |
| `mask` | gan_bpm's brain envelope, `mctp.brain_mask.brain_mask_from_input(raw HU)` | computed from the input with morphology, which is what the paper describes |
| `loss_mask` (**new key**, §7.2) | envelope ∧ (raw label > −0.5) | equals gan_bpm's `support` channel for Tmax (D-96) |

**What changes in meaning:** 1.0 stands for **12 s**, not the paper's 30 s. This is forced:

- UCLA's labels stop at 12 s;
- the generator's tanh output only spans [0, 12] s.

Both conversions are affine and exact in both directions. Nothing in MRDPM's range changes.

**rCBF works the same way (later):** gan_bpm's bounds are (0, 2), so
`rcbf01 = clip(rCBF, 0, 2)/2 = (y + 1)/2`.

---

## 6. Orientation and the asymmetry flip

- MRDPM flips **dim 2** of a `(B, C, H, W)` tensor, which is H, the array rows.
- The paper calls it a *horizontal* flip, meant to mirror one hemisphere onto the other.
- In our RAS storage, array axis 0 is left/right, and it becomes tensor dim 2 once the
  channel axis is added. **So MRDPM's flip is a left/right mirror on our data, with no
  change.**
- Evidence:
  - the builder's RAS reorientation (verified);
  - a quick symmetry check on two slices that pointed the same way, but only weakly
    (head-mask overlap with its mirror: 0.80 vs 0.75 and 0.74 vs 0.70).
- **To do before training:** one PNG check (§12).

---

## 7. Where our data does not fit MRDPM's format exactly

### 7.1 MRDPM masks the input; gan_bpm's generator never saw masked input

- `NumpyDataset_mask` multiplies the condition by the mask, which is roughly the paper's skull
  stripping.
- gan_bpm normalizes the input **unmasked** (CLAUDE.md, "The brain mask"). Its generator has
  always seen the skull: above 325 HU → +1, air → −1.
- Under MRDPM's masking, everything outside the envelope becomes −1 at the generator's input.
  That is a distribution shift for the pretrained weights.

**Proposal: measure before deciding (Step 1, §13).**

- Run the fold-0 generator on fold 0's held-out slices twice: once with the native unmasked
  input, once with the input `× envelope`.
- Compare the brain MAE over `support`, in seconds.
- If masking costs little, use MRDPM's masked condition everywhere. Fine-tuning (λ = 0.1)
  absorbs the rest.
- If it costs a lot, the fallbacks are:
  - (a) Give the predictor alone the unmasked condition. That needs a second dataset key and
    an extra argument through `Network.forward` / `restoration` and their callers in
    `Palette`, which is more invasive.
  - (b) Fine-tune the gan_bpm generator on masked inputs first.

The decision threshold is open (§15).

### 7.2 MRDPM cannot mark "brain voxel without a label"

These voxels are the UCLA −1 fill and ISLES'24's −30 s background inside the envelope. MRDPM
has one mask for three jobs: where noise goes, where the loss counts, and what is pasted
during sampling.

| option | format | problem |
|---|---|---|
| (a) treat them as Tmax 0 | exact | teaches 7.5–10 % of the brain false zeros, the defect gan_bpm fixed in D-96 |
| (b) remove them from `mask` | exact | the mask then comes from the label. MRDPM zeroes the condition outside the mask, so at test time the model is shown where the holes are. A paired comparison with gan_bpm, which never sees that, would not be clean |
| **(c) keep `mask` = envelope, add `loss_mask`** | range and mask unchanged | about 3 lines in `models/network.py` and `models/model.py` |

**Proposal: (c).**

- `Palette.set_input` reads `data.get('loss_mask', mask)`.
- `Network.forward` uses `loss_mask` in `L_diff` and `L_init`.
- Noise, the U-Net input and the sampling paste keep using `mask`.
- Without the key, the behaviour is today's, bit for bit.

---

## 8. The initial predictor: gan_bpm's generator behind BRAN's interface

### 8.1 Interface match (verified by reading both architectures)

At a 512 input, BRAN (`models/region_net.py:Unetgenerator_256`, `ngf = 32`) returns
`(y, hs)` with `hs = [u7, u6, u5]`. Each is a decoder output joined with the encoder skip at
that level.

| resolution | BRAN feature | width | gan_bpm feature (`models/generator.py`) | width |
|---|---|---|---|---|
| H/2 = 256 | `u7 = cat(up7, d1)` | 64 | `cat(decoder6 out, temporal_stem out)` | 16 + 16 = **32** |
| H/4 = 128 | `u6 = cat(up6, d2)` | 128 | `upsample5 = cat(decoder5 out, skip1)` | 32 + 32 = **64** |
| H/8 = 64 | `u5 = cat(up5, d3)` | 256 | `upsample4 = cat(decoder4 out, skip2)` | 64 + 64 = **128** |

gan_bpm's features have the same structure at exactly **half** the width. The H/2 pairing
with the stem output is our construction: gan_bpm itself never joins its stem output.

**Matching the widths:** each feature is duplicated along channels, `torch.cat([f, f], 1)`,
giving 64 / 128 / 256.

- It adds no parameters, so nothing is left untrained.
- The pretrained weights load unchanged.
- `unet.py` does not change at all.
- It is safe for `cond_resample`:
  - Its first layer is `GroupNorm(32)` → SiLU → 1×1 conv. With `[f, f]` ordering, every
    group of 4 channels holds 4 distinct channels.
  - A 1×1 conv on duplicated channels equals a conv on the originals with summed weights.
- Duplicating does **not** add information. The asymmetry module sees 32 / 64 / 128
  distinct channels instead of BRAN's 64 / 128 / 256.
  - Fallback if that proves too thin: a gan_bpm decoder built at BRAN's widths. That needs a
    new pretraining run (gan_bpm's best Tmax λ_adv = 0 epochs were 3–7, so that run is short).

Why the adapters can't be learned layers *before* the U-Net: the U-Net detaches `cond[i]`. A
layer inside the predictor that only feeds those features would never get a gradient.

### 8.2 The wrapper (proposal): `models/gan_bpm_predictor.py`

```python
class GanBpmInitialPredictor(nn.Module):
    """gan_bpm's Generator behind BRAN's (y, hs) interface, in MRDPM's [0, 1] space.

    (N, 3, H, W) condition in [0, 1] -> (y01 (N, 1, H, W) in [0, 1],
                                          [ (N, 64, H/2), (N, 128, H/4), (N, 256, H/8) ])
    """
    def __init__(self, gan_bpm_root, checkpoint=None, n_phases=3,
                 expect_target="tmax", expect_tmax_ceiling=12.0): ...

    def forward(self, cond01):
        x = (2.0 * cond01 - 1.0).unsqueeze(1)            # (N, 1, 3, H, W) in [-1, 1]
        g, u = self.G, self.G.unet
        stem = g.temporal_stem(x)                        # 16 @ H/2
        s1 = u.encoder1(stem); s2 = u.encoder2(s1); s3 = u.encoder3(s2)
        s4 = u.encoder4(s3);   s5 = u.encoder5(s4);  t = u.encoder6(s5)
        up1 = torch.cat([u.decoder1(t), s5], 1)
        up2 = torch.cat([u.decoder2(up1), s4], 1)
        up3 = torch.cat([u.decoder3(up2), s3], 1)
        up4 = torch.cat([u.decoder4(up3), s2], 1)        # 128 @ H/8
        up5 = torch.cat([u.decoder5(up4), s1], 1)        # 64 @ H/4
        d6 = u.decoder6(up5)                             # 16 @ H/2
        y = u.output_head(d6)                            # [-1, 1]
        f2 = torch.cat([d6, stem], 1)                    # 32 @ H/2
        hs = [torch.cat([f, f], 1) for f in (f2, up5, up4)]
        return (y + 1.0) / 2.0, hs
```

The forward repeats `GeneratorUNet.forward`'s exact sequence so that it can return the
intermediate tensors. A check enforces that `y` is bitwise equal to `Generator.forward` (§12).

### 8.3 Loading and guards (proposal)

- **Import by file path.** Load gan_bpm's `models/generator.py` with
  `importlib.util.spec_from_file_location("gan_bpm_generator", ...)`. MRDPM has its own
  `models` package, so `import models.generator` would collide. `generator.py` only imports
  `torch`.
- **What to load.** `torch.load(..., weights_only=False)` (gan_bpm's own `load_checkpoint`
  does the same), then `ckpt["generator"]`, strict.
- **Refuse** a checkpoint whose `ckpt["config"]` disagrees with any of:
  - `target == "tmax"`
  - `tmax_ceiling == 12`
  - `n_phases == 3`
  - `label_transform == "linear"`

  A 40 s checkpoint (e.g. the local `gan_bpm_tmax_f13_k5`, `tmax_ceiling: None`, i.e. 40 s)
  must fail loudly. Under it, `(y + 1)/2` would mean Tmax/40, not Tmax/12.
- **Record** the checkpoint path, its sha256 and `ckpt["epoch"]` in the experiment's log.

### 8.4 Initialization order: a real trap (verified)

- `models/__init__.py:define_network` calls `net.init_weights()` in the train phase, after
  construction.
- `BaseNetwork.init_weights` runs `self.apply(init_func)` over **every** conv, the predictor
  included. Weights loaded in the wrapper's `__init__` would be overwritten with Kaiming noise.
- **Fix (proposal):**
  - `Network.init_weights` calls `super().init_weights()`, then
    `self.initial_net.load_pretrained()`.
  - The wrapper must not define its own `init_weights`, because `BaseNetwork` calls children's
    `init_weights(init_type, gain)`.
- **`./tmax_net_G.pth`.** BaseNetwork's load of this file must be skipped when the gan_bpm
  predictor is selected. Otherwise a stray file would be strict-loaded into the wrong module.
- **Test phase:** `init_weights` isn't called. The predictor's fine-tuned weights come from the
  MRDPM checkpoint (`resume_state`), which is correct.

### 8.5 Which checkpoint: leakage (proposal)

- **The all-data model has trained on every case** in the combined root
  (`..._t12_adv0_all/all/checkpoints/{epoch_5.pt, last.pt}`). As MRDPM's predictor, any
  cross-validated MRDPM score would leak. It is only valid for an ISLES 2018–only evaluation.
- **Fold models are clean.** Use gan_bpm `fold_k/checkpoints/best.pt` as MRDPM fold k's
  predictor, on the same cases.
  - MRDPM reads that fold's own `fold_k/split.json` (`train_case_ids`, `val_case_ids`). The
    cases then match the checkpoint by construction.
- **This also gives the paired comparison** that matters: on fold 0's held-out cases, does
  MRDPM's diffusion on the residual beat the regressor alone?
- **Caveat:** gan_bpm picked `best.pt` on its fold-0 validation fold, so that fold has
  influenced the predictor's epoch choice. MRDPM itself never selects on it (§9).

---

## 9. Splits, monitoring and test (proposal)

**Naming:** MRDPM "test" = gan_bpm fold 0's **held-out fold** (`val_case_ids`). MRDPM
"monitor" = a small validation subset carved from the **training** cases.

- **Train:** fold 0's `train_case_ids`, minus the monitor cases.
- **Monitor:**
  - a few training cases (by patient, deterministic seed), capped at about 32 slices;
  - used only to watch training, never to choose an epoch;
  - the predictor saw these cases in pretraining, so the numbers are slightly optimistic for
    that part.
- **Test:** fold 0's `val_case_ids`. Use the **last** epoch (the paper trains a fixed
  200-epoch schedule), not one chosen by looking.
- **Why not MRDPM's own split:** `validation_split` takes random **slices** out of the
  training set (`data/__init__.py:define_dataset`), so one patient's slices land on both sides.
  - Proposal: an optional `which_val_dataset` block next to `which_dataset`. When present,
    `define_dataset` builds the validation set from it and skips the slice split. Without it,
    behaviour is unchanged. This is about 6 lines, and debug mode must respect it.
- **QC, as gan_bpm does:**
  - drop slices with `label_fg_frac == 0` (all no-data);
  - gan_bpm also drops slices whose `support` is empty, at collate time. For MRDPM such a
    slice would just add zero loss, so filtering is optional.

---

## 10. Code changes, file by file (proposal)

| file | change | default behaviour |
|---|---|---|
| `data/dataset.py` | new class `MCTPDataset_mask` (§10.1). `NumpyDataset_mask` untouched | n/a, new class |
| `data/__init__.py` | optional `which_val_dataset` (§9) | unchanged without it |
| `models/gan_bpm_predictor.py` | new: the wrapper (§8.2–8.3) | n/a |
| `models/network.py` | `Network(initial_predictor=...)` picks BRAN (default) or the wrapper; `init_weights` reloads the pretrained predictor (§8.4); `forward` takes `loss_mask` (§7.2) | BRAN, today's loss |
| `models/model.py` | `set_input` reads `loss_mask` (falls back to `mask`) and passes it to `netG(...)` | unchanged |
| `core/base_network.py` | skip the `tmax_net_G.pth` load unless the predictor is BRAN | unchanged for BRAN |
| `config/ctp_only_f13_tmax_fold0.json` | new config (§11) | n/a |

**Untouched:** the diffusion maths, `unet.py`, the asymmetry module, the training loop, EMA,
checkpointing and `run.py`.

### 10.1 `MCTPDataset_mask`

It reads the f13 root through gan_bpm's `mctp` package. `mctp` only imports numpy, pandas,
scipy and torch, never `models`, so it is safe to append gan_bpm's root to `sys.path`
(appended, so MRDPM's `models` still wins).

- **`__init__` arguments:**
  - `mctp_root`, `label_type="tmax"`, `split_json`
  - `part` ∈ {`train`, `monitor`, `test`}, plus the monitor-carve settings
  - `tmax_ceiling=12.0`
  - `mask_cache_root` (reuse gan_bpm's `MaskCache`; the envelope costs about 134 ms per slice
    without it)
- **`__getitem__`:**
  1. Load the raw input `(3, 512, 512)` HU and the raw label `(512, 512)` s.
  2. `envelope = brain_mask_from_input(raw)` (cached).
  3. `cond = clip((raw − 10)/315, 0, 1) × envelope`
  4. `gt = clip(label, 0, 12)/12 × envelope`
  5. `loss_mask = envelope ∧ (label > −0.5)`
  6. Return MRDPM's dict (§4) plus `loss_mask`. `path = f"{case_id}_slice{idx:03d}.npy"`.
- **Masked-condition switch.** One flag turns off the `× envelope` on the condition, in case
  §7.1's fallback (a) is needed.

---

## 11. Config sketch (proposal)

Only what differs from `config/npy_1000linear_-6_-2_b4_nc64.json`:

```jsonc
{
    "name": "ctp_only_f13_tmax_fold0",
    "datasets": {
        "train": {
            "which_dataset": {
                "name": ["data.dataset", "MCTPDataset_mask"],
                "args": {
                    "mctp_root": "/raid/colivares/mctp_dataset_f13_isles24_ucla_v2",
                    "label_type": "tmax",
                    "split_json": "/radraid2/colivares/matrix/gan_bpm_tmax_f13uclav2_t12_adv0_k5/fold_0/split.json",
                    "part": "train",
                    "tmax_ceiling": 12.0
                }
            },
            "which_val_dataset": { /* same, "part": "monitor" */ },
            "dataloader": { "args": { "batch_size": 1 /* measure at 512 */ } }
        },
        "test": { "which_dataset": { /* same, "part": "test" */ } }
    },
    "model": {
        "which_networks": [{
            "name": ["models.network", "Network"],
            "args": {
                "initial_predictor": {
                    "type": "gan_bpm",
                    "gan_bpm_root": "/raid/colivares/github_repos/gan_bpm",
                    "checkpoint": "/radraid2/colivares/matrix/gan_bpm_tmax_f13uclav2_t12_adv0_k5/fold_0/checkpoints/best.pt"
                },
                "unet": { "in_channel": 4, "image_size": 512 /* rest unchanged */ }
            }
        }]
    }
}
```

The learning rate (5e-5), the noise schedule (1000 linear steps, 1e-6 → 0.01), EMA and the
hardcoded LR schedule (constant to epoch 100, then linear to 0 at 200) stay as the paper and
code have them.

---

## 12. Checks (proposal)

Every check gets a negative control, following gan_bpm's practice.

**Data:**

- The cases in each part equal `split.json`'s. No case or patient is in two parts.
  - Negative control: a monitor case that is also in train is refused.
- `2·cond − 1` equals gan_bpm's `PairedNormalize` image on real slices, inside the envelope
  (within float32 tolerance).
- `gt` equals `(gan_bpm normed_label + 1)/2` inside the envelope wherever the label is valid.
- `loss_mask` equals gan_bpm's `support` channel exactly.
- `path` ends in `.npy`.

**Predictor:**

- The wrapper's `y` is bitwise equal to `Generator.forward` on the same input.
- Feature shapes are `[(64, 256, 256), (128, 128, 128), (256, 64, 64)]` at 512.
- After `define_network`, the predictor's weights equal the checkpoint's.
  - Negative control: without the reload in `init_weights`, they differ.
- Refused checkpoints: 40 s ceiling, `n_phases != 3`, wrong target.

**Loss:**

- With `loss_mask`, changing `gt` at a hole voxel doesn't change the loss.
  - Negative control: with `loss_mask = mask`, it does.
- Without `loss_mask`, the loss is bit-identical to today's.

**Orientation:**

- One PNG: a slice, its `flip(dim=2)`, and their overlay. Confirm by eye that the flip swaps
  the hemispheres.

**Where they run:** the data and predictor checks on the Mac (`torch` env, CPU) where they
don't need UCLA data. The rest runs on the box. `Network.set_new_noise_schedule` defaults to
`cuda`, so the full-network checks need the box or a `device` argument.

---

## 13. Order of work (proposal)

0. **Box prerequisites: checked 2026-10-18** (`condor/probe_box.sh`, run on the box with
   `docker run` in `pytorch-env-v2`).
   - Fold-0 checkpoint (verified): `target` tmax, `tmax_ceiling` 12, `n_phases` 3,
     `label_transform` linear, `lambda_adv` 0, `fold` 0, `data_root` = the v2 root. Epoch 5,
     sha256 `2428e88482ce0994229a086ac4bb4a56b7b37ada5494156ae3e2ea0eba3a4731`. The
     generator's module names match §8.2.
   - `fold_0/split.json` (verified): 353 train / 90 val cases (= 100 + 267 + 76), no
     overlap. The data root already has a `mask_cache/`.
   - Condor jobs run in `pytorch-env-v2` (torch 2.5.1, CUDA 12.4, Python 3.11), **not** the
     box's conda env (torch 2.13). The image lacks `einops`, `dropblock`, `torchviz`,
     `pydot_ng`, `cv2` and `tensorboardX`/`tensorboard`.
     - Only `dropblock` is actually called (BRAN's `UNetUp` dropout). The rest are
       import-time only: `einops` is imported but unused in `region_net.py`, `torchviz` and
       `pydot_ng` are unused in `network.py`, and `cv2` comes in through `data/dataset.py`'s
       unused `.util.mask` import. Tensorboard is optional (the logger only warns).
     - Open: how to add them (a new image tag, or a `pip --target` dir on `/raid`).
1. **The masked-input check (§7.1).** This decides the input path. Script:
   `scripts/masked_input_check.py`. Its native arm goes through gan_bpm's own
   `build_dataloaders` and must reproduce the checkpoint's stored `val_L_pix_brain` (control
   tested on synthetic data: |diff| 3e-8; a re-initialized generator fails it).
2. **Build** the dataset class, the wrapper, the `network.py` / `model.py` changes, the
   validation hook and the config, with the checks in §12.
3. **Smoke on the box:**
   - `python run.py -p train -d` with the new config: a few iterations, one monitor pass on
     2 slices;
   - then `-p test` on 2 slices with `resume_state` set.
4. **The first real run:** fold 0, Tmax.
5. **Score against gan_bpm fold 0** on the held-out fold, in seconds, over `support`.
   - Proposal: write MRDPM's `Out_` maps as gan_bpm's external-maps `.npz`
     (`pred_norm = 2·out − 1`), so gan_bpm's evaluation code scores both models the same way.
   - Not designed yet.

---

## 14. Compute (to measure, not estimated here)

- **512 vs 256.** The generator was trained at 512, so the diffusion U-Net runs at 512. That
  is about 4× the paper's per-step cost. Attention at `ds = 16` becomes 32 × 32 tokens.
  - The batch size has to be measured. The paper used 4 at 256 on a 40 GB A100.
- **Sampling.** Every monitor or test slice costs 1,000 U-Net evaluations (`restoration`).
  - DDIM (`restoration_ddim`) exists but isn't wired in, and it would crash: it calls
    `denoise_fn` without the asymmetry features.
  - Measure seconds per step on the box, then size the monitor subset and `val_epoch`.
- **Disk.** Each saved slice includes `Process_`, the sampling trajectory:
  `(sample_num + 1)` × 512² float32, about 5 MB with `sample_num = 4`.

---

## 15. Open (Carlos)

- **§7.1 threshold:** how much masked-input degradation is acceptable before fallback (a) or
  (b)?
- **§7.2:** confirm option (c), `loss_mask`.
- **Fine-tune or freeze** the predictor. The proposal is the paper's fine-tune; freezing is a
  one-line switch.
- **Epochs:** the paper's 200, with the hardcoded schedule, versus what fits on the box.
- **EMA at test:** the code validates and tests with `netG`, not the EMA copy. The paper says
  EMA "was employed" without saying where.
- **Monitor subset size** and `val_epoch`, once seconds per step are known.
- **After Tmax:** rCBF (same mapping, §5). A BRAN baseline from scratch on 3 channels
  (`Encoderv1x(in_channels=3)`) is an optional ablation.
