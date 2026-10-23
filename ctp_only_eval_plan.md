# ctp-only evaluation plan: MRDPM against gan_bpm

**Status (2026-10-20): frozen.** Carlos set the requirements (decisions #21) and settled the open
items the same day (decisions #22, §9). What is scored, on which voxels and how, is fixed. Build
details (§7) can still be settled as long as they don't change that. No held-out fold and no
ISLES 2018 scan is scored until the build's checks pass, and nothing changes because of what
they show (decisions #15).

Background and the model itself: `ctp_only_plan.md`. Decisions, dated: `ctp_only_decisions.md`.

**No edits to gan_bpm** (Carlos, 2026-10-20). Its evaluation code is imported read-only by
scoring scripts in this repo (§5, §6), which score gan_bpm's models with the same code. Their
gan_bpm numbers must reproduce gan_bpm's own published ones, which is the check that the reuse
is faithful.

---

## 1. What is compared

- **MRDPM (ctp-only) against gan_bpm**, paired per patient, both targets: Tmax (12 s ceiling)
  and rCBF. The main comparison is with gan_bpm's λ_adv = 0 generator, MRDPM's own initial
  predictor: does the diffusion stage improve on its starting map?
- **Two evaluations, as gan_bpm's:**
  - **Internal, 5-fold CV** on ISLES'24 + UCLA EVT + UCLA negative (443 cases, gan_bpm's
    `splits/f13_isles24_ucla_k5.json`). MRDPM fold k against gan_bpm fold k on fold k's
    held-out patients. Every case is scored once, by models that never trained on it.
  - **External, ISLES 2018:** all 94 TRAINING scans (63 patients), as Zeng et al. used them. No
    model trains on it. Two comparisons (§6):
    - **all-data:** MRDPM's all-data models against gan_bpm's, which stage 2 of gan_bpm's ISLES
      2018 evaluation already scored;
    - **fold models:** MRDPM fold k against gan_bpm fold k, each of the 5 on all 94 scans
      (decisions #24).
- **Every applicable gan_bpm row** (Carlos: their performance is about the same):
  - CV: λ_adv = 0, LSGAN and WGAN-GP, each fold's `best.pt`;
  - ISLES 2018, all-data: the 6 all-data models at both snapshots (`epoch_<M>` and `last`), as
    stage 2; fold models: λ_adv = 0, LSGAN and WGAN-GP, each fold's `best.pt`.
  - On ISLES 2018, the reference (the shipped label) as the ceiling, and Zeng et al.'s published
    numbers. The CV's level-0 ladder (training median etc.) is a baseline, not a model; it is
    left out.

## 2. The headline: core Dice with DWI on ISLES 2018

As gan_bpm's Section 2f (`zeng_isles18_eval/PLAN.md`, regions decided by Carlos 2026-10-18):

- **Core** = region ∩ predicted rCBF < 0.3 (CBF below 30 % of normal). **Lesion** = the DWI
  `OT` mask on the build's 512² grid.
- **The hole rule:** voxels RAPID blanked (rCBF = 0 and Tmax = 0, dilated by 1 px) are
  unlabelled.
- **Regions:** `parenchyma` (headline: envelope ∩ phase-1 HU in [15, 100), from the input),
  `valid` (envelope minus the hole), `envelope`. No CSF region (§9.4).
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
the label's own Dice is 0.238 (`parenchyma`) and 0.255 (`valid`): that is the ceiling here.

**Why this is where MRDPM could win.** gan_bpm calls almost no core: it squeezes predictions
toward normal (low-rCBF tissue read about 0.27 too high, spread ratio about 0.79), so infarct
at true rCBF 0.1–0.2 comes out near 0.4, above the cutoff. MAE hides this. A diffusion model
samples realistic values instead of the average, so it may recover the low tail even where its
MAE is worse (the Tmax monitor: MRDPM 1.41 s against the predictor's 1.26 s, plan §16).
Whether it does is what this evaluation measures.

## 3. Metrics, both evaluations

gan_bpm's `evaluation/metrics.py`, plus one metric defined on this side (PSNR full frame):

| metric | brain: `support` = envelope ∧ valid label | full frame: prediction set to the label background outside the envelope |
|---|---|---|
| MAE | `MAE_brain` | `MAE_full_zeroed` |
| NRMSE | `NRMSE_brain` | `NRMSE_full_zeroed` |
| PSNR | `PSNR_brain` | `PSNR_full_zeroed`: −20 log10 of `NRMSE_full_zeroed` per slice, a `Metric` defined in the scoring script (gan_bpm has none) |
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
  MRDPM − gan_bpm per patient with its CI. As gan_bpm (`evaluation/aggregate.py`).
- **Also from gan_bpm's framework**, because MAE hides the squeeze: bias by tissue band (rCBF
  0.3 / 1.2, Tmax 4 / 6 s), spread ratio, W1, sharpness ratio; volumes and Dice of rCBF < 0.3
  and Tmax > 6 s against the label's own regions; Bland–Altman and ICC.

## 4. MRDPM's maps

- **Checkpoint:** epoch 200, the last (fixed schedule, decisions #8). Never chosen by score.
- **Weights: EMA** (decided, §9.1).
- **One sample per slice, as the paper's code does** (decided, §9.2). No averaging: it would
  lower MAE but pull the map toward the middle, the squeeze that hurts core Dice.
- **Sampler:** the full 1000-step sampler MRDPM trains for (`restoration`). No DDIM.
- **Seed:** recorded. The maps are saved, so scoring never resamples.
- **Format:** gan_bpm's stage-1 maps, one per case, so both evaluations read the same files:
  `maps/<case_id>.npz` with `pred_norm` (S, H, W) = 2·`Out_` − 1 over the whole frame,
  clipped to [−1, 1] (decisions #25),
  `pred_phys` (NaN outside the envelope), `envelope`, `ext_gate`, `slice_idx`, `case_id`,
  `patient_id`, and `manifest.json` written last. Every slice of the index, no QC drop.

## 5. The CV evaluation

- **Maps:** MRDPM's test phase on fold k's held-out cases (88–90 cases, about 790 slices), on
  the H100: about 6 h per fold per target, 10 runs, about 60 GPU-hours.
- **Scoring:** a script in this repo. Per fold, it loads each held-out case through gan_bpm's
  `CaseLoader`, takes gan_bpm's predictions from its fold checkpoints through gan_bpm's
  `CheckpointPredictor` and MRDPM's from its maps, and scores all of them with gan_bpm's
  `score_case` and aggregates with gan_bpm's `aggregate`. One script, the same voxels, paired by
  construction.
- **Check:** its gan_bpm λ_adv = 0 numbers reproduce gan_bpm's own: each fold's stored
  `val_L_pix_brain`, and gan_bpm's v2 CV evaluation where it exists (`evaluation/specs/
  {rcbf,tmax}_f13uclav2.json` aren't in the laptop's checkout; whether it has been run is to
  find out).
- **Disclosed:** gan_bpm chose each `best.pt` on that fold's held-out cases (decisions #2). It
  favours gan_bpm's own scores, so MRDPM's paired difference is, if anything, understated.

## 6. ISLES 2018

Two comparisons on the same 94 scans, both scored by one script.

### 6.1 The fold models (decisions #24)

- **What:** each of MRDPM's 5 fold models, at epoch 200 with the EMA weights, on all 94 scans,
  against gan_bpm's model of the same fold (λ_adv = 0, its own predictor; LSGAN and WGAN-GP as
  context), each fold's `best.pt`. No leakage: no fold trained on ISLES 2018.
- **gan_bpm's side:** its stage 1 saved maps for the all-data models only. The fold models'
  predictions are computed here, through gan_bpm's `CaseLoader` and `CheckpointPredictor`,
  read-only (the generator runs at about 60 slices/s), and saved as stage-1 maps before
  scoring (decisions #25).
- **Reported:** every metric per fold, and the mean over the 5 folds with the spread between
  them. The paired difference MRDPM − gan_bpm is taken per patient within each fold, averaged
  over the folds per patient, then bootstrapped over patients. The 5 maps are never averaged
  into one: that would pull them toward the middle, as averaging samples would (§4).
- **Cost:** about 4 h of H100 sampling per fold per target, so about 40 GPU-hours.

### 6.2 The all-data models

- **The models:** gan_bpm scores its all-data runs on ISLES 2018, so MRDPM has an **all-data
  model per target** (decided, §9.3; built, decisions #23): `config/ctp_only_f13_{tmax,rcbf}_all.json`,
  the fold-0 config trained on all 443 cases (gan_bpm's all-data `split.json`), no monitor, its
  initial predictor gan_bpm's all-data λ_adv = 0 at `epoch_<M>`. About 23 h each.
- **Why `epoch_<M>`, not `last`:** `epoch_<M>` is the all-data counterpart of the folds'
  `best.pt` (M = the median of the folds' best epochs). gan_bpm's `last` scored better on the
  ISLES 2018 DWI Dice, and choosing it for that reason would be choosing on the external test
  set.
- **Never on the CV data:** these models trained on every CV case.
- **Inputs:** `/raid/colivares/mctp_dataset_msu_f13_isles2018_all94` (94 scans, 502 slices),
  staged to the H100's `/home`. Maps: about 4 h per target.
- **Scoring:** a script in this repo, using gan_bpm's `zeng_isles18_eval/helpers.py` and
  `evaluation/` read-only. It scores MRDPM's maps and gan_bpm's saved stage-1 maps
  (`/radraid2/colivares/external_maps/isles2018_all94/`) with the same code, with the hole rule
  and the guards (no leakage, the same protocol).
- **Check:** its gan_bpm numbers reproduce stage 2's (the reference 0.238 / 0.255 / 0.163;
  λ_adv = 0 0.036 and 0.099 on `parenchyma`).

## 7. To build

Each with checks and a negative control (the repo's rule). Nothing in gan_bpm changes.

1. **Testing with the EMA weights: built 2026-10-20.** `test_ema` in Palette's config (on in all
   four configs, test phase only, so training is unchanged); the EMA copy gets its own noise
   schedule. (EMA averages parameters only; the model has no BatchNorm, so the copy is
   complete.) Synthetic check: with a fixed seed, two netG tests give the same maps and the EMA
   test other maps.
2. **An H100 test job: built 2026-10-20,** `condor/ctp_only_test_h100.sub` (`run` = a training
   run folder, which gives the target and fold; `testset` = `cv` or `isles18`; `mode`; `epoch`,
   200). It copies the checkpoint to `/home`, samples there, and copies everything but the
   `Process_` trajectories to `/radraid2/colivares/mrdpm_ctp_only/runs/test_…`.
3. **`Out_` → stage-1 maps: built 2026-10-23,** `scripts/ctp_only_isles18_maps.py mrdpm`
   (decisions #25: clipped to [−1, 1]). It refuses a missing slice, a `Mask_` that isn't the
   scan's envelope, a `GT_` that isn't the scan's label (every slice, |Δ| ≤ 1e-5), and a test
   whose EMA checkpoint isn't the training run's (`source_run.json`, written by the test job).
   It also writes `GT_` as a map once per target, for the scoring check.
4. **The CV scoring script** (§5) [not built] and **the ISLES 2018 scoring script** (§6):
   the fold models **built 2026-10-23**, `scripts/ctp_only_isles18_score.py`, with gan_bpm's
   fold maps from `ctp_only_isles18_maps.py gan-bpm` (`condor/ctp_only_isles18_ganbpm_maps.sub`,
   14958) and stage 2's reproduction check first; the all-data comparison (§6.2) not wired yet.
5. **ISLES 2018 on the H100: built 2026-10-23.** `ctp_only_isles18_maps.py split` writes
   `/radraid2/colivares/mrdpm_ctp_only/isles2018/split.json` (94 scans as `val_case_ids`);
   `scripts/ctp_only_stage_data.py` copies the root to the `/radraid2` stage on 14958, and
   `condor/ctp_only_stage_isles18.sub` to the H100's `/home`. `scripts/ctp_only_isles18_dag.py`
   writes the DAG of tests, one per trained fold.
6. **The all-data MRDPM runs: built** (decisions #23).

## 8. Rules

- Nothing changes because of a held-out or ISLES 2018 result (decisions #15).
- No test-time choice is left: the checkpoint, the weights and the sampling are fixed above.
- Every number comes with its CI. Comparisons are paired per patient.
- gan_bpm's `best.pt` caveat (§5) and the no-asymmetry-prior deviation (decisions #10) are
  stated with the results.

## 9. Settled (Carlos, 2026-10-20; decisions #22)

1. **EMA weights at test: yes.**
2. **One sample per slice, as the paper's code does.** No averaging, no repeat-sample check.
3. **All-data MRDPM runs for ISLES 2018: yes**, with gan_bpm's all-data λ_adv = 0
   `epoch_<M>` as the predictor. **The 5 fold models are also scored on ISLES 2018**, against
   gan_bpm's fold models (§6.1, decisions #24), besides the CV.
4. **No CSF region, and no gan_bpm edits for now.**
5. **Every applicable gan_bpm row** (§1).
