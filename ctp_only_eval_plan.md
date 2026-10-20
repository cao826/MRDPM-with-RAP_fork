# ctp-only evaluation plan: MRDPM against gan_bpm

**Status (2026-10-20): drafted from Carlos's requirements (decisions #21). It is frozen once
Carlos settles the open items in §9. No held-out fold and no ISLES 2018 scan is scored before
then, and nothing changes because of what they show (decisions #15).**

Background and the model itself: `ctp_only_plan.md`. Decisions, dated: `ctp_only_decisions.md`.
The evaluation reuses gan_bpm's, so both models are scored by the same code on the same voxels:
gan_bpm's `evaluation/` framework (CV) and its `zeng_isles18_eval` notebook (ISLES 2018).

---

## 1. What is compared

- **MRDPM (ctp-only) against gan_bpm's λ_adv = 0 generator**, which is also MRDPM's initial
  predictor. So the comparison answers one question: does the diffusion stage improve on its
  own starting map? Paired per patient. Both targets: Tmax (12 s ceiling) and rCBF.
- **Two evaluations, as gan_bpm's:**
  - **Internal, 5-fold CV** on ISLES'24 + UCLA EVT + UCLA negative (443 cases, gan_bpm's
    `splits/f13_isles24_ucla_k5.json`). MRDPM fold k against gan_bpm fold k's `best.pt`, on
    fold k's held-out patients. Every case is scored once, by models that never trained on it.
  - **External, ISLES 2018:** all 94 TRAINING scans (63 patients), as Zeng et al. used them. No
    model trains on it. It is scored once, after every model exists.
- **Context rows, not the comparison:** gan_bpm's LSGAN and WGAN-GP; the CV's level-0 ladder
  (training median, rCBF 1.0, oracle slice median, per-voxel linear), as in gan_bpm's v2 spec;
  on ISLES 2018, the reference (the shipped label) as the ceiling; Zeng et al.'s published
  numbers.

## 2. The headline: core Dice with DWI on ISLES 2018

As gan_bpm's Section 2f (`zeng_isles18_eval/PLAN.md`, regions decided by Carlos 2026-10-18):

- **Core** = region ∩ predicted rCBF < 0.3 (CBF below 30 % of normal). **Lesion** = the DWI
  `OT` mask on the build's 512² grid.
- **The hole rule:** voxels RAPID blanked (rCBF = 0 and Tmax = 0, dilated by 1 px) are
  unlabelled.
- **Regions:** `parenchyma` (headline: envelope ∩ phase-1 HU in [15, 100), from the input),
  `valid` (envelope minus the hole), `envelope`.
- **Reported:** per-scan mean, then per patient; each model's Dice as a fraction of the
  reference's in the same region; precision, recall, core volume against DWI volume, Dice by
  lesion size.

**The numbers it sits next to** (gan_bpm stage 2, 2026-10-18, per-scan mean on `parenchyma`):

| source | Dice |
|---|---|
| reference (the label): the ceiling for a model trained on it | 0.238 |
| gan_bpm λ_adv = 0, `epoch_<M>` / `last` | 0.036 / 0.099 |
| gan_bpm WGAN-GP `last` (the best gan_bpm model) | 0.109 |
| Zeng et al.: model / their reference | 0.239 / 0.294 |

**About "0.29".** The label reached 0.292 in the old notebook's `zeros_csf` region, which
removes CSF using pre-contrast frames that the f13 build doesn't carry. In the current regions
the label's own Dice is 0.238 (`parenchyma`) and 0.255 (`valid`). That is the ceiling MRDPM is
measured against here (open item §9.4).

**Why this is where MRDPM could win.** gan_bpm calls almost no core: it squeezes predictions
toward normal (low-rCBF tissue read about 0.27 too high, spread ratio about 0.79), so infarct
at true rCBF 0.1–0.2 comes out near 0.4, above the cutoff. MAE hides this. A diffusion model
samples realistic values instead of the average, so it may recover the low tail even where its
MAE is worse (the Tmax monitor: MRDPM 1.41 s against the predictor's 1.26 s, plan §16).
Whether it does is what this evaluation measures.

## 3. Metrics, both evaluations

gan_bpm's `evaluation/metrics.py`, unchanged except one addition (PSNR full frame):

| metric | brain: `support` = envelope ∧ valid label | full frame: prediction set to the label background outside the envelope |
|---|---|---|
| MAE | `MAE_brain` | `MAE_full_zeroed` |
| NRMSE | `NRMSE_brain` | `NRMSE_full_zeroed` |
| PSNR | `PSNR_brain` | `PSNR_full_zeroed` (**to add**: gan_bpm has no such metric yet) |
| SSIM | `SSIM_brain` | `SSIM_full_zeroed` |

- **Units:** MAE in s (Tmax) or rCBF; NRMSE as a fraction of the label range (12 s; rCBF 2);
  PSNR = −20 log10 NRMSE; SSIM in normalized space, 7 × 7 windows.
- **Full frame is for comparing with papers only.** The background is trivially right, so
  these numbers are optimistic. They are always reported next to the brain numbers, never
  instead of them. The background value is set from the input's envelope, so no label is read
  to build it.
- **Not comparable number for number with the MRDPM paper**, whose ranges were Tmax 30 s and
  CBF 200; NRMSE and PSNR depend on the range.
- **Aggregation:** per slice and patient-pooled; one number per patient; the cohort mean over
  patients with a 95 % bootstrap CI (2000 resamples, seed 0); the paired difference
  MRDPM − gan_bpm per patient with its CI. As gan_bpm.
- **Also from gan_bpm's framework**, because MAE hides the squeeze: bias by tissue band (rCBF
  0.3 / 1.2, Tmax 4 / 6 s), spread ratio, W1, sharpness ratio; volumes and Dice of rCBF < 0.3
  and Tmax > 6 s against the label's own regions; Bland–Altman and ICC.

## 4. MRDPM's maps

- **Checkpoint:** epoch 200, the last (fixed schedule, decisions #8). Never chosen by score.
- **Weights:** EMA (proposal, §9.1).
- **One sample per slice** (proposal, §9.2). Averaging several samples lowers MAE but pulls the
  map toward the middle, the squeeze that hurts core Dice.
- **Sampler:** the full 1000-step sampler MRDPM trains for (`restoration`). No DDIM.
- **Seed:** recorded. The maps are saved, so scoring never resamples.
- **Format:** gan_bpm's stage-1 maps, one per case, so both evaluations read the same files:
  `maps/<case_id>.npz` with `pred_norm` (S, H, W) = 2·`Out_` − 1 over the whole frame,
  `pred_phys` (NaN outside the envelope), `envelope`, `ext_gate`, `slice_idx`, `case_id`,
  `patient_id`, and `manifest.json` written last. Every slice of the index, no QC drop.

## 5. The CV evaluation

- **Maps:** MRDPM's test phase on fold k's held-out cases (88–90 cases, about 790 slices) on
  the H100: about 6 h per fold per target, 10 runs, about 60 GPU-hours.
- **Scoring:** gan_bpm's runner, with a new `ExternalMapsPredictor` that reads MRDPM's maps for
  the fold being scored. It refuses maps from another fold and a missing slice. MRDPM is listed
  in the v2 CV specs (`evaluation/specs/{rcbf,tmax}_f13uclav2.json`, gan_bpm
  `final_models_plan.md` Part F) beside λ_adv = 0.
- **To verify:** those v2 specs aren't in the local gan_bpm checkout. Whether gan_bpm's v2 CV
  evaluation has been built or run.
- **Disclosed:** gan_bpm chose each `best.pt` on that fold's held-out cases (decisions #2). It
  favours gan_bpm's own scores, so MRDPM's paired difference is, if anything, understated.

## 6. ISLES 2018

- **The models:** gan_bpm scores its all-data runs on ISLES 2018. The matched MRDPM is an
  **all-data MRDPM per target** (proposal, §9.3): trained on all 443 cases with the same
  200-epoch recipe, its initial predictor gan_bpm's all-data λ_adv = 0 `epoch_<M>`. About 23 h
  each (3,924 slices per epoch against a fold's 3,096).
- **Why `epoch_<M>`, not `last`:** `epoch_<M>` is the all-data counterpart of the folds'
  `best.pt` (M = the median of the folds' best epochs). gan_bpm's `last` scored better on the
  ISLES 2018 DWI Dice, and choosing it for that reason would be choosing on the external test
  set.
- **Inputs:** `/raid/colivares/mctp_dataset_msu_f13_isles2018_all94` (94 scans, 502 slices),
  staged to the H100's `/home`. Maps: about 4 h per target.
- **Scoring:** gan_bpm's stage-2 notebook scores MRDPM's maps as one more model, with its
  guards (no leakage, the same protocol) and the hole rule.
- **Alternative, not proposed:** the 5 fold models on ISLES 2018. No new training, but 5× the
  sampling, and gan_bpm would need its fold models' ISLES 2018 maps as well.

## 7. To build

Each with checks and a negative control (the repos' rule).

1. **Testing with the EMA weights:** a switch in `Palette.test`. The EMA copy also needs its
   noise schedule set before sampling.
2. **An H100 test job** (`target`, fold, checkpoint, weights), syncing its maps to `/radraid2`
   as the training job does.
3. **`Out_` → stage-1 maps**, a script. It refuses a missing slice; a round trip of `GT_`
   through the same path must equal the dataset's label.
4. **gan_bpm:** `ExternalMapsPredictor`, `PSNR_full_zeroed`, MRDPM in the v2 CV specs.
5. **All-data MRDPM:** a split with every case in training, a config per target, gan_bpm's
   all-data generators staged on the H100, the ISLES 2018 root staged there.
6. **ISLES 2018 maps** with the test job, pointed at the ISLES 2018 root.

## 8. Rules

- Nothing changes because of a held-out or ISLES 2018 result (decisions #15).
- Any test-time choice (none is planned beyond §9) is made on the monitor patients, which are
  training patients, never on the held-out fold or ISLES 2018.
- Every number comes with its CI. Comparisons are paired per patient.
- gan_bpm's `best.pt` caveat (§5) and the no-asymmetry-prior deviation (decisions #10) are
  stated with the results.

## 9. Open (Carlos)

1. **EMA or raw weights** at test. Proposal: EMA. Standard for diffusion models and what the
   paper says it "employed"; its released code tests with the raw weights. Our LR goes to 0
   over epochs 100–200, so by epoch 200 the two should be close.
2. **One sample per slice.** Proposal: yes. Optional: on the monitor patients, sample twice and
   report how much MAE and the core volume move between samples.
3. **An all-data MRDPM for ISLES 2018,** with gan_bpm's all-data λ_adv = 0 `epoch_<M>` as the
   predictor. Proposal: yes. Costs 2 trainings of about 23 h.
4. **A CSF-excluding region closer to the old 0.292.** It needs pre-contrast data the build
   doesn't carry, so it is a gan_bpm change for every model, not an MRDPM one.
5. **Which gan_bpm rows** besides λ_adv = 0 appear in the tables (LSGAN, WGAN-GP, the ladder).
