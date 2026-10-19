# ctp-only plan: MRDPM on f13 CTP phases, with gan_bpm's generator as the initial predictor

**Status (2026-10-19): steps 0–2 (§13) are done: the input path (§7.1) and the loss mask (§7.2)
are decided, and the build passes every check, synthetic and on the real data (§12). The
orientation is right (§6), but the scans aren't aligned, so the asymmetry flip is off (§6). Step 3,
the GPU smoke run, passed (§13, §14). Step 4, fold 0 on the H100 box, is built
(`condor/ctp_only_train_h100.sub`, with the health log of §16): a debug submission first.** Decisions, dated, are listed
in `ctp_only_decisions.md`. Branch `ctp-only`. This doc collects what
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
- **Box check (2026-10-19, §12 cpu part):** over 792 held-out slices, the envelope's median
  dice with its mirror, re-centred, is 0.967 across axis 0 (MRDPM's flip) and 0.945 across
  axis 1. Axis 0 wins, but by little: a head outline is nearly symmetric both ways.
- **`orientation.png`, by eye (2026-10-19): the axis is right.** On the slice with the most
  brain (`dYoSKe7gFe_slice001`), the head's long, front-to-back axis runs along the columns
  and the frontal sinus sits on the midline, so `flip(dim=2)` swaps the two hemispheres.
- **But the mirror is approximate.** The head is tilted by roughly 10° and sits off the frame's
  centre, so the mirrored brain lands a few millimetres off: the overlay's edges don't meet,
  and a bright midline calcification shows as a red and a green dot side by side.
- **The paper aligned its images; we don't (verified, paper §3, correcting an earlier line
  here).** Its preprocessing included *"adjusting images (cropping, rotation, translation) for
  brain centralization"*, so its plain array flip mirrored roughly aligned hemispheres. The
  released code has no alignment step only because it ships no preprocessing.
- **Where the flip is in our model:** not in BRAN. It is in the diffusion U-Net's asymmetry
  module, `cond_resample.forward` (`models/guided_diffusion_modules/unet.py:332`), which mirrors
  the initial predictor's feature maps `hs` (H/2, H/4, H/8) before fusing them. With BRAN
  replaced, `hs` are gan_bpm's decoder features, so the U-Net mirrors those.
- gan_bpm trains with no augmentation at all (`mctp/transforms.py:build_transforms`), so its
  generator has only seen scans as stored.
- **Decision (Carlos, 2026-10-19): no flip.** `unet.mirror_hs: false` keeps the module's
  feature path but pairs each feature map with itself, not its mirror image
  (`ctp_only_decisions.md` #10). Report the model as MRDPM without the asymmetry prior.

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

**Result (verified 2026-10-19, `scripts/masked_input_check.py` on the box, fold 0, 90 cases,
793 slices).** Masking costs about a fifth of the error, and the cost is about the same in every
dataset:

| | cases | slices | native | masked | Δ |
|---|---|---|---|---|---|
| all | 90 | 793 | 1.433 s | 1.732 s | +0.300 s (+20.9 %) |
| isles24 | 20 | 370 | 1.133 s | 1.363 s | +0.230 s (+20.3 %) |
| ucla_evt | 54 | 325 | 1.788 s | 2.144 s | +0.356 s (+19.9 %) |
| ucla_neg | 16 | 98 | 1.292 s | 1.641 s | +0.349 s (+27.0 %) |

The native arm's pooled L1 (0.23878) agrees with the checkpoint's `val_L_pix_brain` (0.2388).

**Decision (Carlos, 2026-10-19): the generator gets the whole head; only the diffusion
U-Net's copy of the condition is masked.**

- The dataset returns the condition **unmasked**, as gan_bpm's `PairedNormalize` does (it
  windows the whole frame and masks only the label, `mctp/transforms.py:302–319`).
- `Network` multiplies the condition by `mask` (= envelope) where it enters the diffusion
  U-Net: the two `denoise_fn` calls in `forward`, and the `y_cond` that `restoration` hands
  to `p_sample`. The predictor gets the unmasked condition.
- The predictor's output is already cut to `mask`: `y_initial = y_initial*mask`
  (`models/network.py:98, 182`).
- Inputs are only ever masked with the **envelope**, never `support`. `support` comes from
  the label, so masking an input with it would show the model where the label's holes are
  at test time.
- This is fallback (a), but done inside `Network` with the mask it already has: no second
  dataset key and no `Palette` change.
- Each half sees what it was built for: gan_bpm the whole head, the diffusion U-Net the
  paper's skull-stripped condition.

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

**Decision (Carlos, 2026-10-19): (c).** The training loss and every metric count only
`support`. Whatever the model puts in a hole is ignored: a no-data hole has no right answer
to learn or score. Both models are scored on the same voxels, defined by the label alone, so
the paired comparison with gan_bpm stays fair. Two things to report or check:

- **The holes are not random tissue.** About half of UCLA's sit in the parenchyma band, which
  fits RAPID failing in tissue (gan_bpm `final_models_plan.md`, "UCLA v1 cross-check").
  Scores over `support` say nothing about those voxels, so report the excluded fraction with
  each score. Nobody has measured whether UCLA's holes overlap the lesion (ISLES 2018's
  blanks: 2.4 % in the DWI lesion).
- **Diffusion only.** gan_bpm's output at a hole affects nothing else. MRDPM's sampler feeds
  each step's whole image, holes included, back in as the next input. The network can't see
  where the holes are, so at sampling it treats them like similar tissue. In training,
  though, the noisy input at a hole carries the clipped marker (a Tmax-0 target), which it
  never sees at sampling. Probably small. Check after the first run: compare MRDPM's and
  gan_bpm's errors on `support` voxels next to holes and far from them.

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

### 8.2 The wrapper (built): `models/gan_bpm_predictor.py`

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

### 8.3 Loading and guards (built)

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

## 9. Splits, monitoring and test (built)

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

## 10. Code changes, file by file (built 2026-10-19)

| file | change | without the new keys |
|---|---|---|
| `data/dataset.py` | new class `MCTPDataset_mask` (§10.1). `NumpyDataset_mask` untouched | n/a, new class |
| `data/__init__.py` | optional `which_val_dataset` (§9). It refuses a validation set that shares a patient with the training set. In debug mode it cuts both, to `debug_split` and `debug_val_split` | unchanged |
| `models/gan_bpm_predictor.py` | new: the wrapper (§8.2–8.3), plus an optional `expect_fold` guard | n/a |
| `models/network.py` | `initial_predictor` picks BRAN (absent) or the wrapper. `init_weights` reloads a pretrained predictor (§8.4). `mask_unet_cond` masks the condition at the U-Net input, in `forward` and `restoration` (§7.1). `forward` takes `loss_mask` (§7.2) | BRAN, with today's U-Net input and loss, bit for bit (checked against `main`'s `forward`) |
| `models/model.py` | `set_input` reads `loss_mask` (`None` when absent); `train_step` passes it on | unchanged |
| `models/__init__.py` | the train log records the predictor's checkpoint, sha256 and epoch | unchanged for BRAN |
| `core/base_network.py` | skips `./tmax_net_G.pth` when the predictor loads its own checkpoint | unchanged for BRAN |
| `core/logger.py` | `LogTracker.reset` no longer writes through `.values` (below) | same values |
| `models/guided_diffusion_modules/unet.py` | `UNet(mirror_hs=...)` → `cond_resample(mirror=...)`: when off, the asymmetry module pairs the predictor's features with themselves instead of their flip (§6). Same layers and shapes | the flip, as upstream |
| `config/ctp_only_f13_tmax_fold0.json` | new config (§11) | n/a |
| `models/health.py` | new: the training-health log (§16) | n/a, used only with `health_log` |
| `models/model.py`, again | `health_log` hooks in `train_step` / `val_step`; a resumed run continues the LR schedule (§16) | unchanged, except that a resumed run's LR is now right |
| `models/network.py`, again | the `loss_mask` branch keeps its two terms in `self.loss_terms` | unchanged |

**Untouched:** the diffusion maths, the training loop, EMA, checkpointing and `run.py`.

**An upstream bug, fixed in `core/logger.py`.** Under pandas ≥ 3, `self._data[col].values` is a
read-only view, so `LogTracker.reset` raised and `Palette` could not be built. The box image
has pandas 3.0.5 (probe, §13 step 0), so this would have stopped the first GPU job. The CPU run
of `run.py` in `scripts/ctp_only_checks.py` found it.

### 10.1 `MCTPDataset_mask`

It builds gan_bpm's own `MCTPDataset` with `build_transforms`, the same `PairedNormalize` that
gan_bpm trained and validated with, and converts each item to MRDPM's dict. So the input
window, the envelope and its cache, and the no-data rule are gan_bpm's code, not copies of it.
gan_bpm's root is appended to `sys.path`, after MRDPM's, so MRDPM's `models` still wins.
`mctp` never imports `models`.

- **Arguments:** `mctp_root`, `gan_bpm_root`, `split_json`, `part` ∈ {`train`, `monitor`,
  `test`}, `label_type="tmax"`, `tmax_ceiling=12.0`, `monitor_cases_per_dataset=1`,
  `monitor_max_slices=32`, `monitor_seed=0`, `mask_cache=True`. Also gan_bpm's row QC,
  `min_label_fg_frac` and `qc_drop_slice_idx`, with gan_bpm's defaults.
- **Refuses** a `split.json` whose train and val share a patient, or that lists a case the
  index lacks.
- **Monitor:** `monitor_cases_per_dataset` whole patients per dataset, drawn from
  `train_case_ids` with a seeded permutation, then capped at `monitor_max_slices` evenly
  spaced slices. Train is `train_case_ids` minus every monitor patient's cases.
- **Item:**
  - `cond_image = (gan_bpm input + 1)/2` = `clip((HU − 10)/315, 0, 1)`, **not** masked
  - `gt_image = (gan_bpm label + 1)/2 × envelope` = `clip(Tmax, 0, 12)/12` inside the envelope
  - `mask` = the envelope, `loss_mask` = `support`, `path = f"{case_id}_slice{idx:03d}.npy"`
- **DataLoader workers must fork** (Linux's default). A spawned worker, as on macOS, cannot
  import `mctp`.

---

## 11. Config (built): `config/ctp_only_f13_tmax_fold0.json`

What differs from `config/npy_1000linear_-6_-2_b4_nc64.json`:

- **Datasets:** `MCTPDataset_mask` for train, monitor (`which_val_dataset`) and test, all on
  the v2 root and fold 0's `split.json`. The monitor is 1 patient per dataset, at most 32
  slices.
- **Network:** `initial_predictor` = gan_bpm's fold-0 `best.pt` with `expect_fold: 0`;
  `mask_unet_cond: true`; `unet.in_channel` 4, `image_size` 512 and `mirror_hs: false` (§6).
- **The recipe (decided 2026-10-19, `ctp_only_decisions.md` #11–#12):** train batch 4 (the
  paper's; needs the H100), monitor 16 slices at batch 16 every 10 epochs, test at batch 16,
  `health_log` on (§16). The debug cut is 4 training slices, one whole batch, because the train
  loader drops an incomplete batch.
- **Output:** `path.base_dir` = `/radraid2/colivares/mrdpm_ctp_only/experiments`, not the repo.
- **Batch:** train 1 (to measure, §14). Monitor and test 4: sampling only, no gradients.
- **`save_checkpoint_epoch` 10, not 1:** each save is 1.1 GB (§14).
- **`debug` block:** `n_epoch` 0 (the loop runs while `epoch <= n_epoch`, so that is one
  epoch), 2 training slices, 2 monitor slices, a checkpoint and a monitor pass at epoch 1.

Unchanged from the paper and the code: lr 5e-5, the noise schedule (1000 linear steps,
1e-6 → 0.01), EMA, the hardcoded LR schedule (constant to epoch 100, then linear to 0 at 200),
200 epochs, a monitor pass every 10.

---

## 12. Checks

Two scripts. Where a check could pass without testing anything, a negative control follows it.

**`scripts/ctp_only_checks.py`: synthetic data, CPU. 74/74 passed on the laptop (2026-10-19).**
In the box image it passed every section up to the `run.py` run, which crashed: the box's
docker shows containers its GPUs, so `core/util.py:set_device` put the model on `cuda:0` next to
the script's CPU noise schedule. The script now hides the GPUs before importing torch; rerun
pending.
It builds a fake f13 root at 256 × 256 (three datasets, a skull, no-data holes, real zeros,
labels above 12 s, one patient with two cases) and fake gan_bpm checkpoints.

- **Data:**
  - parts against `split.json`, whole patients, the monitor cap;
  - negative control: a split with a patient in train and val is refused;
  - values against gan_bpm's pipeline, and against §5's formulas computed from the raw files:
    `cond` exact, `gt` to 4e-8 (the 40 s scale fails);
  - `loss_mask` = `support` and `mask` = envelope, exactly; `.npy` paths, unique.
- **`which_val_dataset`:** each set from its own block, the debug cuts, the upstream slice
  split unchanged without it. Negative control: shared patients are refused.
- **Predictor:**
  - strict weights and their sha256;
  - `y` bitwise equal to `Generator.forward`;
  - `hs` equal to G's own intermediate tensors, doubled;
  - BRAN's shapes at 512;
  - refuses a 40 s ceiling, 4 phases, rCBF, `sqrt` and the wrong fold.
- **Initialization order:** the predictor's weights equal the checkpoint's after
  `define_network` (negative control: BaseNetwork's init alone overwrites them). A stray
  `tmax_net_G.pth` is skipped, and the log records the checkpoint.
- **Network:**
  - the predictor reads the whole condition and the U-Net the condition × mask, in `forward`
    and in every `restoration` step;
  - negative controls: the two differ on the batch; masking the predictor's input changes
    `y_initial`; with the flag off the U-Net reads the whole condition.
  - With `loss_mask`, `gt` in the holes doesn't change the loss (negative control: with
    `mask` as the loss mask, it does). This uses a pointwise stand-in for the U-Net. The real
    U-Net mixes neighbours, so the hole values do reach the loss through its input: §7.2's
    second caveat.
  - Without `loss_mask` and `mask_unet_cond`, the loss is bit-identical to `main`'s `forward`.
  - Gradients reach the predictor and the U-Net.
- **Config:** the parts, the shared arguments, the same monitor carve for train and monitor,
  the checkpoint in the split's fold directory, `expect_fold`, `in_channel`, 512,
  `mask_unet_cond`, `log_iter`.
- **`run.py` end to end on the CPU** (5 noise steps, 256 × 256):
  - `train -d` writes checkpoint 1 and a monitor pass (`GT_`/`Out_`/`Process_`/`Mask_` for
    2 slices, finite, 0 outside the mask). The log names the checkpoint, and the predictor's
    weights moved.
  - `test -d` with `resume_state` writes `Out_` for 2 held-out slices and `ddim_test.txt`.
  - The MRDPM checkpoint covers every weight (`strict=False` skips only the six noise-schedule
    buffers), so the predictor tested is the fine-tuned one.

**`scripts/ctp_only_box_checks.py`: the real data and checkpoint, on the box, in two parts.
Both parts pass (2026-10-19): cpu 13/13, gpu 6/6.** Tested on the laptop against a 512 fixture (the gpu part with CUDA stubbed to the
CPU): everything passes except the two checks that cannot there (the sha256, and orientation on
a perfectly round head).

- **`--part cpu`** (`docker run` on the box, data only):
  1. The real checkpoint loads, with §13's sha256, and was trained on this root.
  2. The parts against `split.json`; the monitor is 1 patient per dataset.
  3. The test part against gan_bpm's own validation loader, built from the checkpoint's
     config: the same slices in the same order, the same values.
  4. Orientation (§6): the envelope's mirror dice across each axis, as stored and re-centred,
     plus `orientation.png` (a slice, its `flip(dim=2)`, the overlay) to check by eye.
  - **Result:** train 350 cases / 3,096 slices, monitor 3 cases / 32 slices (one per
    dataset), test 90 / 793. The test part equals gan_bpm's validation set exactly (max diff 0,
    masks exact). Orientation: §6.
- **`--part gpu`** (Condor, `condor/ctp_only_gpu_checks.sub`):
  1. The checkpoint, as above.
  2. Control: the predictor alone, through MRDPM's dataset and wrapper, reproduces
     `val_L_pix_brain` = 0.2388 (1.433 s). **Result:** 0.238780 against the stored 0.238780,
     over the 793 held-out slices.
  3. The Network from the real config at 512: the predictor's weights after init, one forward,
     its time and peak memory at batch 1.

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
     - Resolved: a new image tag, `registry.cvib.ucla.edu/colivares:mrdpm-env-v1`
       (`Dockerfile`), pushed 2026-10-18.
1. **The masked-input check (§7.1).** This decides the input path. Script:
   `scripts/masked_input_check.py`. Its native arm goes through gan_bpm's own
   `build_dataloaders` and must reproduce the checkpoint's stored `val_L_pix_brain` (control
   tested on synthetic data: |diff| 3e-8; a re-initialized generator fails it).
   **Done 2026-10-19:** native 1.433 s, masked 1.732 s (+21 %); the native arm reproduces
   the stored 0.2388. Decision in §7.1.
2. **Build** the dataset class, the wrapper, the `network.py` / `model.py` changes, the
   validation hook and the config, with the checks in §12. **Built 2026-10-19** (§10, §11).
   The synthetic checks pass on the laptop; the real-data checks are next.
3. **Smoke on the box's GPU** (a Condor job). **Built 2026-10-19:** `condor/ctp_only_smoke.sub`.
   - `python run.py -p train -d` with the new config: one epoch on 2 slices, one monitor pass
     on 2 slices;
   - then `-p test -d` on 2 slices with `resume_state` set (the job writes a copy of the
     config with it, since `run.py` has no flag);
   - then `scripts/ctp_only_gpu_probe.py`: peak memory and seconds per step for a training
     step at batch 1, 2, 4, 8 and a sampling step at 1, 4, 8, 16, with what each costs per
     epoch, monitor pass and test pass (§14).
   - The job refuses to start when its container sees more than one GPU: `run.py` sets
     `CUDA_VISIBLE_DEVICES` to the config's `gpu_ids` ("0"), which is only this job's card
     when it is the only one.
   - **Done 2026-10-19, passed:** `run.py` train → monitor pass → checkpoint 1 → test from it,
     on a Quadro RTX 8000. Monitor and test each sampled 2 slices in about 5½ min
     (1,000 steps at 3.07 it/s, batch 2). Probe results in §14.
   - **Next (2026-10-19): the same probe on the H100 NVL box** (REDLRADADM35838, 95 GB per
     card), `condor/ctp_only_probe_remote.sub`. That machine has none of 14958's paths, so
     the job ships a bundle: a `git archive` of this repo, 32 training slices, the fold-0
     checkpoint and gan_bpm's `generator.py` (`ctp_only_gpu_probe.py --make-bundle`). A real
     run there would also need the data on that machine.
4. **The first real run:** fold 0, Tmax, on the H100 box. **Built 2026-10-19:**
   `condor/ctp_only_train_h100.sub` (arguments: fold, `debug` or `full`, an optional resume
   checkpoint).
   - The job copies into its workspace, from `/radraid2/colivares/mrdpm_ctp_only/stage/`,
     only what it reads (`scripts/ctp_only_stage_data.py`): the Tmax index, the mask cache, and
     the input and Tmax-label files of the fold's training cases. The whole staged root is
     31 GB (every fold's held-out cases, rCBF labels and more), against a shared 41 GB free on
     the box's local disk. Also gan_bpm's code and the fold's `split.json` and `best.pt`. The
     cluster's rule: never run on data in `/radraid*`.
   - It writes a copy of the config pointing at those copies and at the fold, and trains there.
   - Every 10 minutes it moves finished checkpoints to
     `/radraid2/colivares/mrdpm_ctp_only/runs/<experiment>/` and copies the rest (health files,
     monitor maps, `train.log`, run.py's output). The newest checkpoint set stays until a newer
     one exists, since it may still be being written; at the end everything goes.
   - Condor brings back only a summary file, so the workspace's data never lands in `condor/`.
   - Submit `debug` first (1 epoch on 4 slices, a monitor pass on 2), then `full`.
5. **Score against gan_bpm fold 0** on the held-out fold, in seconds, over `support`.
   - Proposal: write MRDPM's `Out_` maps as gan_bpm's external-maps `.npz`
     (`pred_norm = 2·out − 1`), so gan_bpm's evaluation code scores both models the same way.
   - Not designed yet.

---

## 14. Compute (to measure, not estimated here)

- **Measured (verified 2026-10-19, `scripts/ctp_only_gpu_probe.py`, Quadro RTX 8000, 47.3 GiB):**

  | step | batch | s/step | peak allocated |
  |---|---|---|---|
  | train (forward, backward, Adam, EMA) | 1 | 0.559 | 12.75 GiB |
  | | 2 | 1.069 | 24.29 GiB |
  | | 4 | out of memory | |
  | sample (one of 1,000) | 1 | 0.174 | 3.60 GiB |
  | | 4 | 0.619 | 10.79 GiB |
  | | 8 | 1.237 | 20.38 GiB |
  | | 16 | 2.095 | 39.55 GiB |

  - Training: 0.48 h per epoch (3,096 slices) at batch 1, 0.46 h at batch 2. The card is
    already busy at batch 1, so batch 2 buys no speed.
  - Sampling: batching saves only about a quarter per slice (0.174 → 0.131 s per slice-step),
    so one slice costs 2.2–2.9 min.
  - As configured (200 epochs, 32 monitor slices at batch 4 every 10 epochs, test at batch 4):
    training 96 h, monitoring 20 × 1.38 h = 28 h, test 34 h (29 h at batch 16). About 6.5 days
    for one fold on one card.
  - During the smoke run, `nvidia-smi` showed up to 40.6 GB in use on the card, against
    PyTorch's 12.75 GiB peak at batch 1. `nvidia-smi` counts PyTorch's cache and anything else
    on the card. Not explained yet; a training job should get a card to itself.
- **The H100 NVL box (verified 2026-10-19, `condor/ctp_only_probe_remote.sub`, REDLRADADM35838,
  93.1 GiB usable; TF32 on in convolutions by PyTorch's default, off in matmuls):**

  | step | batch | s/step | peak allocated | vs the RTX 8000 |
  |---|---|---|---|---|
  | train | 1 | 0.133 | 12.72 GiB | 4.2× faster |
  | | 2 | 0.222 | 24.11 GiB | 4.8× |
  | | 4 | 0.400 | 46.88 GiB | (didn't fit) |
  | | 6 | 0.593 | 69.66 GiB | |
  | | 8 | out of memory | | |
  | sample | 1 | 0.042 | 3.72 GiB | 4.1× |
  | | 16 | 0.436 | 39.68 GiB | 4.8× |
  | | 32 | 0.878 | 78.04 GiB | |

  - 200 epochs: 22.9 h at batch 1, 17.2 h at batch 4, 17.0 h at batch 6.
  - A test pass (793 slices): 6.1 h at batch 16 or 32. The configured monitoring (32 slices every
    10 epochs): 20 × 0.26 h = 5.2 h.
  - So one fold is about a day on one H100, against about 6.5 days on an RTX 8000.
  - **What a job on the H100 box can reach (verified 2026-10-19, `condor/probe_h100_storage.sub`).**
    The box takes no logins, only Condor jobs.
    - `/radraid` and `/radraid2` are NFS shares, the same ones 14958 sees, and a job can write
      back to `/radraid*/colivares`. (gan_bpm's Condor notes call `/radraid2` per-machine; it
      isn't.)
    - Its `/raid` is a different, empty, root-owned directory on the local disk.
    - The job's workspace is on the local disk too: 98 GB, 41 GB free at the time, shared with
      every other job on the box.
    - The image has `cp` and `tar` but no `rsync`.
  - **The cluster's rule** (Carlos's knowledge doc): don't run on data in `/radraid*`. Put the
    data on the share once, copy it into the job's workspace at the start of every job, and move
    the results back at the end.
  - **Plan:** a one-time copy of the dataset and gan_bpm's code from 14958's `/raid` to
    `/radraid2/colivares/mrdpm_ctp_only/stage/`. Each H100 job copies them into its workspace.
    Workspace disk is the tight resource: the dataset plus checkpoints (1.1 GB each) must fit,
    so not every checkpoint can stay there until the end.
- **512 vs 256.** The generator was trained at 512, so the diffusion U-Net runs at 512. That
  is about 4× the paper's per-step cost.
  - **Attention (verified, correcting an earlier line here).** `attn_res: [16]` never
    matches: with four levels the U-Net's `ds` only reaches 8. The only attention layer is the
    middle block's, at H/8. That is 64 × 64 = 4,096 tokens at 512, against the paper's
    1,024 at 256: 16× the attention memory.
  - The batch size has to be measured. The paper used 4 at 256 on a 40 GB A100.
  - One forward of one 512 slice, without gradients (verified, gpu check, Quadro RTX 8000,
    47.3 GiB): 0.193 s warm, 2.68 GiB peak allocated. On the laptop's CPU it takes 15 s.
  - That is roughly one sampling step, so one slice sampled at batch 1 costs about 1,000 ×
    0.19 s ≈ 3 min. Batching should lower the per-slice cost; step 3 measures by how much.
- **Size (verified).** The U-Net has 68.5 M parameters and the predictor 1.9 M. One save is
  the Network (282 MB), its EMA copy (282 MB) and the optimizer state (564 MB): 1.1 GB. Saving
  every 10 epochs, a 200-epoch run writes about 22 GB.
- **Sampling.** Every monitor or test slice costs 1,000 U-Net evaluations (`restoration`).
  - DDIM (`restoration_ddim`) exists but isn't wired in, and it would crash: it calls
    `denoise_fn` without the asymmetry features.
  - Measure seconds per step on the box, then size the monitor subset and `val_epoch`.
- **Disk.** Each saved slice includes `Process_`, the sampling trajectory:
  `(sample_num + 1)` × 512² float32, about 5 MB with `sample_num = 4`.

---

## 15. Open (Carlos)

- ~~§7.1 threshold~~ Decided 2026-10-19: masking costs +0.30 s, so the predictor gets the
  whole head and only the U-Net's condition is masked (§7.1).
- ~~§7.2~~ Decided 2026-10-19: (c), `loss_mask` = `support` for the loss and every metric.
- ~~**Fine-tune or freeze**~~ Decided 2026-10-19: fine-tune jointly, as the paper (decisions #11).
- ~~**Epochs**~~ Decided 2026-10-19: the paper's 200 with its schedule (decisions #11).
- **EMA at test:** the code validates and tests with `netG`, not the EMA copy. The paper says
  EMA "was employed" without saying where.
- ~~**Monitor subset size** and `val_epoch`~~ Decided 2026-10-19: 16 slices every 10 epochs
  (decisions #12).
- **Validity (§17):** repeat the masking check and the orientation look on training cases, so
  fold 0's held-out cases informed no decision. Freeze the evaluation before any held-out fold
  is scored (decisions #15).
- ~~**Alignment for the asymmetry prior**~~ Decided 2026-10-19: no flip (`mirror_hs: false`,
  §6). The options below are what it would take to turn it back on.
  The paper rotated and centred its images; ours are tilted and off-centre, so the flip mirrors
  misaligned hemispheres. In the paper's
  ablation (Table 2) the asymmetry module is worth MAE 0.049 → 0.047. Options:
  - (a) flip anyway, and report it as a deviation from the paper's preprocessing;
  - (b) rotate and centre each case in the data loader, as the paper did. gan_bpm's generator
    then sees inputs it was never trained on (re-check it as in §7.1), and the comparison with
    gan_bpm needs one common space;
  - (c) keep every input as stored and replace the array flip in `cond_resample` with a
    reflection across each case's estimated midline (a per-sample `grid_sample`). One model
    edit; gan_bpm's input and the scoring space are unchanged.
  - (b) and (c) both need a midline estimate per case.
- **After Tmax:** rCBF (same mapping, §5). A BRAN baseline from scratch on 3 channels
  (`Encoderv1x(in_channels=3)`) is an optional ablation.

---

## 16. Training health log (built 2026-10-19)

MRDPM itself logs one combined loss every `log_iter` samples and, at each monitor pass, a `mae`
over the whole image (background included, so it says little). With `health_log` in Palette's
config, `models/health.py` adds, under `{experiment}/health/`:

- **`train.csv`, per epoch:** the loss and its two terms (the diffusion noise MSE and the
  predictor's L1 against the label, both over `loss_mask`); the learning rate and the
  schedule's position (`lr_epoch`); the gradient norm, mean and max over the epoch's steps;
  non-finite steps; seconds; slices per second; peak GPU memory.
- **`monitor.csv`, per monitor pass:** the pooled error over `loss_mask` in seconds, for
  MRDPM's output and for the predictor alone on the same slices. That shows whether diffusion
  improves on the starting map and whether fine-tuning degrades the predictor. Also the
  fraction of the envelope that `loss_mask` covers.
- **`monitor_eNNN.png`:** label, predictor, MRDPM and both error maps for 4 slices.

It only reads: `scripts/ctp_only_checks.py` checks that the loss and the gradients are the same
with it, and that the loss equals its two logged terms exactly.

`scripts/plot_health.py <experiment dir>` turns the CSVs into `health/curves.png`, at any point
during the run.

**Also fixed: a resumed run restarted the LR schedule.** Palette's `LambdaLR` isn't in
`self.schedulers`, so it was never saved. A run resumed at epoch E restarted it at 0: constant
LR until epoch E + 100, then the decay. `Palette.__init__` now sets it to the resumed epoch.
Checked end to end: a run resumed from checkpoint 1 trains epoch 2 with the schedule at
epoch 1 (`lr_epoch` in `train.csv`).

## 17. Validity (discussed 2026-10-19)

- **Cross-validation:** gan_bpm's 5 folds, split by patient; MRDPM fold k uses gan_bpm's fold-k
  generator, which trained only on fold k's training cases. Every case is tested once, by models
  that never trained on it. The cross-validation result needs all 5 folds.
- **ISLES:** ISLES'24 is inside the cross-validation data. ISLES 2018 is gan_bpm's external
  set: no model trains on it, it is scored once after every model exists, and nothing changes
  because of it.
- **Clean by construction:** patient-level splits; the held-out fold chooses nothing for MRDPM
  (fixed schedule, last epoch, monitor from training patients); fixed scaling constants; both
  models scored on the same voxels.
- **Two weak spots:**
  1. gan_bpm chose each fold's `best.pt` on that fold's held-out cases, the ones MRDPM is tested
     on. It favours gan_bpm's own scores equally, so the paired difference stays fair; absolute
     numbers are slightly optimistic for both. Proposal: disclose.
  2. The masking check (§7.1) scored gan_bpm's generator on fold 0's held-out cases, and the
     no-flip decision (§6) came from one held-out input image (no labels, no predictions).
     Proposal: repeat both on training cases.
- **The rule from here:** freeze the evaluation (metrics, one number per patient, the paired
  comparison) before any held-out fold is scored; change nothing because of test results.

