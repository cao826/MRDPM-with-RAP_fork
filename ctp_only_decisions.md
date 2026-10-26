# ctp-only decisions log

Every design decision on the `ctp-only` branch, with its date. Newest last. Add an entry when
a decision is made; don't rewrite old ones. If one is reversed, add a new entry that says so
and mark the old one *superseded*.

The reasoning, measurements and code details live in `ctp_only_plan.md` (the § numbers below).

**Status:**
- **Decided:** Carlos decided it.
- **Adopted:** Claude proposed it, it is built and in use, and Carlos hasn't overruled it. It
  can still change.

---

### 1. 2026-10-18 · What we are building

- **Decision:** MRDPM conditioned on the **3 CTP phases of the f13 protocol, with no NCCT**.
  Tmax first. The initial predictor is **gan_bpm's λ_adv = 0 Tmax generator** (12 s ceiling),
  pretrained, in place of the paper's BRAN.
- **Why:** reuse a predictor that is already trained on this data, and change MRDPM as little
  as possible.
- **Status:** decided (Carlos). Plan §1.

### 2. 2026-10-18 · Fold models, not the all-data model

- **Decision:** MRDPM fold k uses gan_bpm's `fold_k/checkpoints/best.pt` and that fold's own
  `split.json`. The first run is fold 0.
- **Why:** the all-data generator has trained on every case, so any cross-validated MRDPM score
  would leak. With fold models, both models are scored on the same held-out cases.
- **Caveat:** gan_bpm chose `best.pt` on its fold-0 validation cases.
- **Status:** adopted. Plan §8.5.

### 3. 2026-10-18 · Value ranges

- **Decision:** `gt = clip(Tmax, 0, 12) / 12`, so 1.0 means **12 s**, not the paper's 30 s. The
  condition is gan_bpm's window, `(clip(HU, 10, 325) − 10) / 315`.
- **Why:** UCLA's labels stop at 12 s and the generator's output spans only [0, 12] s. Both
  scalings are exact affine maps of gan_bpm's, so the generator's input and output carry over
  unchanged.
- **Status:** adopted (forced by the data). Plan §5.

### 4. 2026-10-18 · 512 × 512, not 256

- **Decision:** train and sample at 512, the generator's resolution.
- **Cost:** about 4× the paper's compute per step, and 16× its attention memory.
- **Status:** adopted. Plan §1, §14.

### 5. 2026-10-18 · Where things run

- **Decision:**
  - The laptop only writes code. Nothing runs there.
  - GPU work runs as Condor jobs on 14958, in a new image `mrdpm-env-v1` (gan_bpm's image
    plus MRDPM's missing packages).
  - CPU-only work is a `docker run` pasted on the box.
  - Code goes through the GitHub fork: push `ctp-only`, then pull on the box.
- **Status:** decided (Carlos). The rule against running on the laptop was added 2026-10-19.

### 6. 2026-10-19 · The generator sees the whole head; only the U-Net's condition is masked

- **Decision:** the dataset returns the CTP condition unmasked. `Network` multiplies it by the
  brain envelope only where it enters the diffusion U-Net (`mask_unet_cond: true`).
- **Why:** gan_bpm's generator was trained on the whole head. On fold 0's 793 held-out slices,
  masking its input worsened its Tmax error from 1.433 s to 1.732 s (+21 %).
- **Rule:** inputs are only ever masked with the envelope, never with `support`. `support` comes
  from the label, so it would show the model the label's holes at test time.
- **Status:** decided (Carlos). Plan §7.1.

### 7. 2026-10-19 · Loss and metrics count only `support`

- **Decision:** `mask` stays the brain envelope (noise, U-Net input, sampling paste). A new
  `loss_mask` (gan_bpm's `support`: envelope ∧ valid label) restricts the training loss and every
  metric. Whatever the model puts in a no-data hole is ignored.
- **Why:** a hole has no right answer to learn or to score. Both models are scored on the same
  voxels, defined by the label alone.
- **Report:** the fraction of the brain each score excludes. About half of UCLA's holes sit in
  the parenchyma.
- **Status:** decided (Carlos). Plan §7.2.

### 8. 2026-10-19 · Splits: by patient, from gan_bpm's fold

- **Decision:**
  - The **test** set is fold 0's held-out cases (`val_case_ids`).
  - The **monitor** set is 1 patient per dataset, taken from the training cases and capped at
    32 slices. It is used only to watch training, never to pick an epoch.
  - Test uses the **last** epoch.
  - A `which_val_dataset` config block replaces MRDPM's random slice split.
- **Why:** MRDPM's own split draws random slices, so one patient's slices land on both sides.
- **Status:** adopted. Plan §9.

### 9. 2026-10-19 · Save a checkpoint every 10 epochs

- **Why:** each save is 1.1 GB (Network, its EMA copy, optimizer state). Every 10 epochs, a
  200-epoch run writes about 22 GB.
- **Status:** adopted. Plan §11, §14.

### 10. 2026-10-19 · No asymmetry flip (`unet.mirror_hs: false`)

- **Decision:** the U-Net's asymmetry module still adds the initial predictor's feature maps
  into the U-Net at three scales, but pairs each with itself instead of its mirror image.
  Every layer keeps its shape, so checkpoints load either way.
- **Why:**
  - The paper rotated and centred its images before training (its §3: "cropping, rotation,
    translation for brain centralization"). There, a plain array flip mirrors one hemisphere
    onto the other.
  - Our scans are stored as acquired. The flip's axis is right (left/right, checked by eye on
    `orientation.png`), but heads are tilted (roughly 10° by eye on the slice checked) and off-centre,
    so each spot would be paired with a point a few millimetres off the true opposite side.
- **Cost:** no asymmetry prior, the "RAP" in the paper's title. Report this model as **MRDPM
  without RAP**. In the paper's ablation (Table 2) the whole module, feature path included, is
  worth MAE 0.049 → 0.047. The mirror's own share is unknown.
- **Alternatives:**
  - (a) flip the scans as they are;
  - (b) rotate and centre each case in the data loader. gan_bpm's generator never saw such
    input: it trained with no augmentation;
  - (c) reflect across each case's estimated midline instead of flipping the array.

  (b) and (c) need a midline estimate per case. Either would make the flip worth turning back
  on.
- **Code:** `models/guided_diffusion_modules/unet.py` (`cond_resample(mirror=...)`,
  `UNet(mirror_hs=...)`, default on, so `main` and upstream configs are unchanged), and
  `config/ctp_only_f13_tmax_fold0.json`.
- **Status:** decided (Carlos). Plan §6, §15.

### 11. 2026-10-19 · The training recipe for fold 0, on the H100 box

- **Decision:**
  - Batch 4, learning rate 5e-5, 200 epochs with the paper's schedule (constant to epoch 100,
    then linear to 0 at 200), EMA 0.9999.
  - The predictor is fine-tuned jointly, λ = 0.1, as in the paper.
  - Train on the H100 NVL box (REDLRADADM35838): about 4× the RTX 8000, about 20 h per fold.
- **Why batch 4, not 6:**
  - Batch 4 is the paper's.
  - Batch 6 is no faster per epoch (17.0 h vs 17.2 h): the card is already fully busy.
  - It would cut the number of updates by a third at the same learning rate.
  - The memory left over can't be used by another job, since the job holds the whole card
    either way.
- **Status:** decided (Carlos). Plan §11, §14.

### 12. 2026-10-19 · Monitor: 16 slices every 10 epochs

- **Decision:** 16 slices (one batch on the H100) from the 3 monitor patients, sampled every
  10 epochs: 20 points on the curve, about 2.4 h of the run.
- **Why:** it watches training, it never selects an epoch. The same slices are sampled every
  time, so the trend is steady. Its real limit is the 3 patients, not the slice count.
- **Alternatives:** 8 slices every 20 epochs (Carlos found it underpowered); 32 every 10
  (about 5 h).
- **Status:** decided (Carlos). Plan §11.

### 13. 2026-10-19 · Data reach the H100 box through `/radraid2`

- **Decision:** a one-time copy of the dataset and gan_bpm's code from 14958's `/raid` to
  `/radraid2/colivares/mrdpm_ctp_only/stage/` (`rsync -aL`, because the combined root is
  symlinks; 31 GB). Each job copies into its workspace only the files it reads (its side of
  the fold, Tmax only), trains there, and moves its results back to
  `/radraid2/colivares/mrdpm_ctp_only/runs/`.
- **Why:** the H100 box takes no logins. The cluster's rule (Carlos's knowledge doc) is to never
  run on data in `/radraid*`, but to copy in at the start and move results back at the end.
- **Status:** decided (Carlos). Plan §13 step 4, §14.

### 14. 2026-10-19 · Training-health log, and the LR schedule on resume

- **Decision:** `models/health.py`, switched on by `health_log` in the config, writes per-epoch
  and per-monitor-pass CSVs and a PNG per monitor pass. `scripts/plot_health.py` plots them.
  A resumed run now continues the LR schedule instead of restarting it.
- **Why:** MRDPM alone logs one combined loss and a whole-image MAE, which says little.
- **Status:** decided (Carlos). Plan §16.

### 15. 2026-10-19 · Freeze the evaluation before scoring any held-out fold

- **Decision:** training can start now. No held-out fold is scored until the evaluation plan
  (metrics, one number per patient, the paired comparison with gan_bpm) is written down. Nothing
  changes because of what a held-out fold shows.
- **Status:** decided (Carlos). Plan §17.

### 16. 2026-10-19 · No fixed seed (`seed: -1`, upstream's)

- **Decision:** keep upstream's `seed: -1`. Runs aren't bit-reproducible: the data order and the
  noise differ from run to run.
- **Why:** a fixed seed also switches cuDNN to deterministic mode (`core/util.py:set_seed`),
  which is slower. With `-1`, cuDNN benchmarks its convolution methods on the first step: a
  first-epoch memory peak of 90 GiB on the H100, the card's own either way.
- **Status:** decided (Carlos). Plan §13 step 4.

### 17. 2026-10-19 · On the H100 box, data and run folders live on its own `/home`

- **Decision:** a one-time Condor job (`condor/ctp_only_stage_home.sub`) copies the Tmax files of
  all 443 cases (about 17 GB), gan_bpm's code and the 5 fold generators from the `/radraid2`
  stage to `/home/colivares/mrdpm_ctp_only/stage` on the H100 box. Training jobs read them there
  and write their run folders there, asking Condor for 2 GB of workspace instead of 24. Results
  still go to `/radraid2/colivares/mrdpm_ctp_only/runs/` every 10 minutes, since 14958 can't see
  that disk. This supersedes #13's per-job copy into the workspace.
- **Why:**
  - Condor gives jobs only about 55 GB of workspace on that box, shared by its 8 GPUs (job
    workspace is on the 98 GB system disk). One of our jobs held 24 GB of it.
  - `/home` there is the machine's own 84 TB disk (7.1 TB free), visible and writable from jobs.
  - The cluster's rule is against running on the network shares; `/home` is local to the box.
    Carlos confirmed that jobs may use it.
- **Status:** decided (Carlos). Plan §13 step 4.

### 18. 2026-10-20 · rCBF: the Tmax recipe, with gan_bpm's rCBF generators

- **Decision:** a second model, f13's 3 CTP phases → rCBF. The initial predictor of fold k is
  `gan_bpm_rcbf_f13uclav2_adv0_k5/fold_k/checkpoints/best.pt` (λ_adv = 0), with that fold's
  `split.json`. Everything else is the Tmax recipe (#3–#17): `gt = clip(rCBF, 0, 2)/2`, the same
  input path, masks, splits, batch, schedule, monitor and checkpointing.
  `config/ctp_only_f13_rcbf_fold0.json` differs from Tmax's only in the label's keys, which
  `scripts/ctp_only_checks.py` enforces.
- **Verified on the box (2026-10-20):** all 5 rCBF fold generators are linear, 3-phase,
  λ_adv = 0, on the v2 root; every fold's `split.json` equals the Tmax run's; the `/radraid2`
  stage holds all of rCBF's files, on the same slices and with the same inputs as Tmax's.
- **Why:** reuse a predictor already trained on this data and these folds, as #1 did for Tmax.
  With the same folds, the two targets' scores are on the same patients.
- **Caveats:** as #2, gan_bpm chose each `best.pt` on the held-out fold. About 12 % of UCLA's
  positive rCBF voxels are above 2 and clip, for both models (gan_bpm's ceiling).
- **Code:** `models/gan_bpm_predictor.py` (`target`), the config.
- **Status:** decided (Carlos asked for rCBF, 2026-10-20); the recipe being Tmax's is adopted.
  Plan §18.

### 19. 2026-10-20 · rCBF trains with a zero residual at no-data voxels (`zero_hole_residual`)

- **Decision:** in training, the residual is `(y_0 − y_initial) × loss_mask`, not `× mask`, so
  the U-Net's noisy input at a no-data voxel is noise alone. On for rCBF; Tmax's config doesn't
  set it, so the Tmax runs continue unchanged.
- **Why:** a hole's label clips to 0. For rCBF that's dead tissue, so the residual there is about
  −0.5 wherever the tissue is normal: a fake core in the U-Net's input on about 10 % of every
  UCLA brain, which the convolutions spread to voxels the loss counts. Sampling never shows the
  model such input. For Tmax, 0 s is normal tissue and the residual at a hole is small.
- **Alternatives:** (a) leave it, as Tmax does; (b) also switch it on for Tmax, which would make
  the Tmax folds already trained differ from the later ones.
- **Code:** `models/network.py` (`zero_hole_residual`), the rCBF config.
- **Status:** proposed by Claude; **decided (Carlos, 2026-10-20): (a), on for rCBF, off for
  Tmax.** The effect isn't measured either way. After rCBF's first fold, compare MRDPM's error on
  valid voxels next to holes with its error far from them (plan §7.2's check). Plan §18.

### 20. 2026-10-20 · rCBF runs beside Tmax on the H100 box

- **Decision:** the Condor jobs take a `target` macro, passed as `MRDPM_TARGET` (default tmax,
  arguments unchanged). rCBF is staged into the same `/home` root, with its own generators
  (`gan_run_rcbf/`) and marker (`STAGED_OK_rcbf`), reusing Tmax's copy of gan_bpm's code and
  the mask cache without touching them. `condor/ctp_only_rcbf_folds.dag` runs folds 0–4, one at
  a time.
- **Why:** the Tmax folds' DAG is running, and its later nodes read the checkout as it is when
  they start: a new positional argument, or a staging job that replaced gan_bpm's code or Tmax's
  generators, could break them. One at a time keeps the account at 2 of its 3 GPUs.
- **Alternatives:** 2 folds at a time (about 2.5 days instead of 4), if nothing else of Carlos's
  holds a GPU.
- **Correction (2026-10-20):** "2 of its 3" left out Tmax fold 0, submitted on its own. While it
  runs, Tmax holds 2 GPUs and this DAG the third. 2 at a time fits only after it ends.
- **Status:** adopted. Plan §18.

### 21. 2026-10-20 · How MRDPM is evaluated: as gan_bpm, CV and ISLES 2018

- **Decision:** evaluate MRDPM the way gan_bpm is evaluated, against gan_bpm, both in the
  5-fold CV and on ISLES 2018 (all 94 scans). Metrics: MAE, NRMSE, PSNR and SSIM over the brain
  mask, and over the full frame with the prediction set to background outside the mask (the
  way other papers report; optimistic, so always beside the brain numbers). **The headline is
  the Dice between the rCBF < 0.3 core and the DWI lesion on ISLES 2018**, against the label's
  own Dice as the ceiling.
- **Why:** the comparison that matters is with gan_bpm, so it uses gan_bpm's scoring code and
  conventions; the papers' full-frame numbers have to be reproducible too.
- **Plan:** `ctp_only_eval_plan.md`; its §9 lists what is still open (EMA, one sample, an
  all-data MRDPM for ISLES 2018, a CSF region, context rows).
- **Status:** decided (Carlos). Freezes when §9 is settled; nothing is scored before (#15).

### 22. 2026-10-20 · The evaluation's open items, settled

- **Decision (Carlos):**
  - MRDPM is tested with its **EMA weights**.
  - **One sample per slice, as the paper's code does**: no averaging, no repeat-sample check.
  - **All-data MRDPM runs** for ISLES 2018; the 5-fold models are compared with gan_bpm's in
    the CV.
  - **No edits to gan_bpm** for now, so no CSF region: scoring scripts in this repo import
    gan_bpm's evaluation code read-only, score gan_bpm's models with it too, and must reproduce
    gan_bpm's published numbers.
  - **Every applicable gan_bpm row** (λ_adv = 0, LSGAN, WGAN-GP; on ISLES 2018 both snapshots),
    since their performance is about the same.
- **Why:** EMA is standard for diffusion models and the paper says it "employed" it; averaging
  samples would re-create the squeeze toward normal that hurts the core Dice.
- **Status:** decided (Carlos). `ctp_only_eval_plan.md` is frozen (§9).

### 23. 2026-10-20 · The all-data MRDPM runs

- **Decision:** `config/ctp_only_f13_{tmax,rcbf}_all.json`: each target's fold-0 config,
  trained on all 443 cases (gan_bpm's all-data `all/split.json`), **no monitor** (every case
  trains, as gan_bpm's all-data runs have no validation), ISLES 2018 as the test set, and gan_bpm's
  all-data λ_adv = 0 generator at `epoch_<M>` as the predictor (Tmax `epoch_5`, rCBF `epoch_9`,
  from gan_bpm's records; the staging job fails if either is missing).
- **Why `epoch_<M>`:** the all-data counterpart of the folds' `best.pt`. gan_bpm's `last` did
  better on ISLES 2018, so choosing it would be choosing on the external test set.
- **Running it:** `condor/ctp_only_stage_alldata.sub` stages the two generators (adds files
  only); the training job takes `fold = all`; `condor/ctp_only_alldata.dag` runs both, one at a
  time. With both fold DAGs running, it fits under the 3-GPU cap only after Tmax fold 0 ends.
- **Alternatives:** keep a monitor (3 patients fewer in training); `last` as the predictor.
- **Status:** adopted. `ctp_only_eval_plan.md` §6.

### 24. 2026-10-20 · The fold models are scored on ISLES 2018 too

- **Decision (Carlos):** besides the CV and the all-data models, each of MRDPM's 5 fold models
  is scored on all 94 ISLES 2018 scans, against gan_bpm's model of the same fold.
- **How, without editing gan_bpm:** gan_bpm never saved fold-model maps for ISLES 2018, so the
  scoring script computes them through gan_bpm's own `CaseLoader` and `CheckpointPredictor`,
  read-only. Each fold is reported, and the mean over folds; the paired difference is per
  patient within a fold, averaged over folds, bootstrapped over patients. The fold maps are
  never averaged into one map.
- **Cost:** about 40 H100-hours of sampling (5 folds × 2 targets × about 4 h).
- **Status:** decided (Carlos). `ctp_only_eval_plan.md` §6.1.

### 25. 2026-10-23 · How the ISLES 2018 fold-model evaluation is built

Settled before any ISLES 2018 map exists, so no result shaped it (#15).

- **MRDPM's map is clipped to the label range:** `pred_norm = clip(2·Out_ − 1, −1, 1)`, not
  the plain `2·Out_ − 1` of eval plan §4. A diffusion sample can step past the range the
  label was clipped to (rCBF below 0 or above 2, Tmax above 12 s). gan_bpm's maps can't
  (Tanh), and gan_bpm's own predictor contract (`predict_case`) refuses values outside
  [−1, 1]. Clipping changes no core or lesion region (rCBF < 0.3 and Tmax > 6 s lie inside
  the range), and it can only bring a value closer to a truth that lives in the range. The
  manifest counts the clipped voxels per model. *Alternative:* no clipping, which breaks
  gan_bpm's contract and can only add error.
- **gan_bpm's fold maps are written to disk, not computed inside the scoring script** (#24
  said "computes them"): `ctp_only_isles18_maps.py gan-bpm`, through the same `CaseLoader` and
  `CheckpointPredictor`, in stage 1's map format. Every model, MRDPM's, gan_bpm's folds and
  the all-data ones, then goes through stage 2's `score_target` and `score_dwi` along one
  code path. The job first reruns stage 1's all-data λ_adv = 0 `last` through the same
  loader and writer and must match stage 1's stored maps.
- **ISLES 2018 uses stage 2's metric set:** levels 1–3 after the hole rule and the DWI Dice,
  without the full-frame (paper-only) metrics, which stage 2 leaves out because they ignore
  the hole rule. Eval plan §3 listed them for both evaluations; on ISLES 2018 they're
  [not built]. The headline doesn't depend on them.
- **Where each map came from is recorded:** the test job writes `source_run.json` (training
  folder, epoch, sha256 of the EMA checkpoint); the converter refuses a test whose checkpoint
  no longer matches.
- **Checks,** in the scoring script, before any table: stage 2's DWI numbers reproduced
  (reference 0.238 / 0.255 / 0.163, λ_adv = 0 0.036 / 0.099, WGAN-GP `last` 0.109, ±0.0005);
  MRDPM's saved label as a map scores MAE 0 and the reference's Dice on every scan, and the
  same one slice off doesn't; the fold-averaged comparison over one fold equals gan_bpm's
  `comparison_rows`, and differs from another fold's.
- **Status:** adopted (Carlos may veto the clipping before the first scoring).
  `scripts/ctp_only_isles18_{maps,score,dag}.py`, `condor/ctp_only_stage_isles18.sub`,
  `condor/ctp_only_isles18_ganbpm_maps.sub`.

### 26. 2026-10-25 · Runs that Condor restarted: recovered, and a restart now resumes

- **What happened:** Condor evicted three training jobs and ran each again from the top, under
  the same job id and with no `resume`: Tmax fold 3 (job 1821650.0) and rCBF fold 4
  (1821651.0) on 10-21, and rCBF fold 3 (1822551.0) three times on 10-23/24. Each attempt
  trained from epoch 0 into a new run folder beside the old one on the H100's `/home`, and the
  job's copy step copied only the first folder by name, which was the dead attempt's. The last
  attempts reached epoch 200 and exited 0, so DAGMan counted the folds done while `/radraid2`
  showed them stopped at 180, 180 and 130. Found from the jobs' `.log` (2 or 4 starts) and
  `.out` (two folders in the progress lines).
- **Decision:**
  - **These three fold models are the restarted runs:** `train_ctp_only_f13_tmax_fold3_261021_183315`,
    `train_ctp_only_f13_rcbf_fold4_261021_183440`, `train_ctp_only_f13_rcbf_fold3_261024_032243`.
    Each is a whole, uninterrupted run of the recipe from epoch 0, like folds 1 and 4 of Tmax.
    `condor/ctp_only_recover_runs.sub` copies finished run folders from the box to `/radraid2`.
    It only adds folders, and it renames a copy only after its files and sizes match.
  - **A restarted training job resumes** from its earlier attempts' newest checkpoint whose
    three files load (on `/home` or already on `/radraid2`) when it is newer than its
    `resume`, and the job copies every run folder it holds, not only the first. The test job
    copies its newest attempt's folder (a test has nothing to resume).
- **Why:** the models trained correctly; only the copy failed. The rule that picks them is the
  fixed one (checkpoint 200), with no score involved. Each restart from scratch cost about
  20 GPU-hours.
- **Alternatives:** resume the dead attempts from 180, 180 and 130 by hand (about 2, 2 and 7 h),
  which throws away three finished models.
- **Status:** adopted. `condor/ctp_only_recover_runs.{sub,sh}`, `condor/ctp_only_train_h100.sh`,
  `condor/ctp_only_test_h100.sh`. Running jobs keep the code they were submitted with:
  rCBF fold 2 (1822702.0) still has the old copy step.

### 27. 2026-10-25 · Why the core is missed: diagnosed on the training data, not on ISLES 2018

- **Decision (Carlos):** after the first ISLES 2018 scores (rCBF folds 0 and 1: DWI Dice
  0.083 on `parenchyma`, against the reference's 0.238), looking at maps to find out why
  MRDPM misses the core is done on the CV folds' held-out cases, plus any training slices
  or cases the model struggles on. Not on ISLES 2018, for now.
- **Why:** ISLES 2018 is the held-out test; changing the model because of what its maps show
  would make the next ISLES number optimistic (#15).
- **The figures (bookmarked, not built):** for each chosen slice, four images: the input as
  read from file ("raw"), the input as the model gets it (after gan_bpm's `mctp` pipeline:
  normalization, masking), MRDPM's generated rCBF and the ground-truth rCBF.
- **Status:** decided (Carlos), not built.

### 28. 2026-10-25 · The "right place or wrong level" check, on the training data; the data-audit leads closed

- **Decision (Carlos):** run the check that tells apart the two reasons the core can be
  missed, on the training data (#27), against the label's own rCBF < 0.3 core (no DWI there):
  - **Matched cores:** per scan, a map's k lowest-rCBF voxels in the brain (`support`), with
    k the size of the label's rCBF < 0.3 core there, and their Dice with the label's core.
    Scans whose label has no core are left out and counted.
  - **Cutoff sweep:** the Dice of map < t with the label's core (the label stays at 0.3),
    for t = 0.2, 0.3, … 1.0.
  - **Floor:** k voxels drawn at random in the same brain: the Dice of a map that carries
    no information.
  - Every map on the same scans: MRDPM's and its own predictor's (gan_bpm λ_adv = 0, same
    fold).
- **How to read it:** a matched Dice well above the Dice at 0.3 means right place, wrong
  level: the scale is off, and a cutoff or a rescaling chosen on the CV data could recover it
  without retraining. A matched Dice near the random floor means wrong place: the model
  doesn't locate the core, and no cutoff helps.
- **Why the label's core and not DWI:** on the training data the question is whether the
  model reproduces its label, and it leaves ISLES 2018 untouched. gan_bpm's
  `zeng_isles18_eval` 2g asks the same on ISLES 2018 against DWI.
- **The data-audit leads, closed (Carlos):** ISLES 2018 comes motion-corrected as
  distributed (the build has no step for it because none is needed). The phases stay f13.
  ISLES 2018 scans whose automatic timing failed use the raters' manual AIF. Still
  unexamined: Zeng's spatial and temporal smoothing (kernels 7 and 3) before the phases are
  taken.
- **Status:** decided (Carlos), not built. It needs MRDPM's maps on those cases; the CV
  tests (`ctp_only_test_h100.sub`, `testset = cv`, about 6 h per fold) make them, and the CV
  evaluation (eval plan §5) needs them anyway.

### 29. 2026-10-25 · The CV folds become the development set; the core check, built

- **Decision (Carlos):** the diagnosis of #27 and #28 uses the CV folds' held-out cases. So
  the CV is a development set from here on: what it shows may change the model, and its
  numbers are reported as such. ISLES 2018 stays the one untouched test (#15 still holds for
  it).
- **Alternatives:** training cases in-sample (no test set touched, but a new test option and
  extra sampling), or the monitor patients (only a handful).
- **Built:**
  - `scripts/ctp_only_core_check.py` (the check of #28, on 14958 through
    `condor/ctp_only_core_check.sub`, one GPU for gan_bpm's predictions). Maps: MRDPM's
    `Out_`, MRDPM's own fine-tuned predictor (`Init_`), gan_bpm λ_adv = 0 `best.pt`. Its
    controls stop it before any model number: the label as a map scores 1, the label one slice
    off scores below 0.9 on average, and k random voxels land on the analytic floor. It also
    refuses a test that holds a training case, a `Mask_` that isn't the envelope and a `GT_`
    that isn't the label.
  - **The test phase also saves `Init_`,** the initial predictor's map alone
    (`Out_ = Init_ +` the sampled residual inside the mask), so the check can tell what the
    diffusion stage adds from what the predictor's fine-tuning did. It is a second forward
    pass of the predictor after sampling; the predictor has no randomness in eval mode, so no
    sample changes. Training and the monitor are untouched.
    `scripts/ctp_only_checks.py` gained four checks for it.
  - `scripts/ctp_only_isles18_dag.py --testset cv` writes the DAG of CV tests.
  - `--gan-bpm-only` (Carlos's question: gan_bpm can answer faster): gan_bpm's fold models
    alone (λ_adv = 0, with `--recipe` LSGAN and WGAN-GP) on every held-out case, with no MRDPM
    test. Minutes on a GPU against about 6 h of sampling per MRDPM fold, and it already says
    whether MRDPM's starting map knows where the core is.
- **Status:** decided (Carlos); built 2026-10-25, not yet run.

### 30. 2026-10-25 · The paper's own predictor, BRAN, as an arm (branch `bran-ext`)

- **Decision:** a second kind of initial predictor beside gan_bpm's generator: **BRAN**, the
  paper's (`models/region_net.py`), at 3 input channels and otherwise as the code builds it (ngf
  32, region-aware convs, BatchNorm, DropBlock). It is pretrained alone, then trained into MRDPM
  with today's recipe:
  - **Pretraining:** L1 over `support` (paper Eq. 1), 50 epochs, Adam, LR linear to 0 (upstream
    README). **lr 2e-4, betas (0.5, 0.999), N(0, 0.02) init and batch 4 are our choices**: the
    README and the paper give none of them. The first three are pix2pix's defaults, which BRAN's
    code comes from; batch 4 is the joint stage's. The model is the last epoch. It trains on the
    fold's training cases minus the monitor patients, as the MRDPM run does.
  - **Input:** BRAN reads the condition × envelope (`mask_cond`), the paper's skull-stripped
    input, as the U-Net already does. This differs from #6, where gan_bpm's generator reads the
    whole head because that is how it was trained.
  - **Flip:** still off (#10). The paper's ablation gives the whole asymmetry module 0.002 MAE
    and BRAN 0.001, against 0.014 for the mask, and reports no core number. Nothing there argues
    for registering our scans first.
- **Why:** Carlos asked for the run closest to the paper's setup (`model_improvement_plan.md`
  item 1). gan_bpm's generator is half BRAN's width, with InstanceNorm and no region-aware convs,
  and its `best.pt` was chosen on the held-out fold (#2).
- **Guard:** `Network` refuses a BRAN checkpoint whose `meta` disagrees with the config (target,
  fold, channels, masked input, loss, recipe), or a debug or unfinished pretraining unless
  `allow_debug` (only the debug training job sets it).
- **Code:** `models/bran_predictor.py`, `models/region_net.py` (`cond_channels`),
  `models/network.py` (`type: "bran"`, `Network.initial`), `scripts/pretrain_bran.py`,
  `condor/ctp_only_bran_pretrain_h100.{sub,sh}`. Plan: `bran_ext_plan.md` §2.
- **Status:** proposed by Claude and built 2026-10-25, at Carlos's request; checks not run, nothing
  trained. Carlos to confirm the recipe choices and the masked input.

### 31. 2026-10-25 · Zeng's extreme loss inside MRDPM: in the predictor, and as a voxel weight on L_diff

- **Decision:** two optional terms, both off unless a config sets them:
  - `initial_predictor.ext = {weight w, side}`: `w · mean(support · W · (pred − gt)²)` with
    `W = (2·gt − 1)²` (Zeng's `W_B = B ⊙ B` on MRDPM's [0, 1] label), added to BRAN's pretraining
    loss and to the joint stage's L_init. **w = λ_ext / 50** is gan_bpm's balance exactly (checked
    against gan_bpm's `losses.py`), so Zeng's λ_ext = 50 is w = 1, the arms' value.
  - `diffusion_ext = {weight w, side}`: `w · mean(support · W(gt) · (ε − ε̂)²)` added to L_diff.
    Since `(ε̂ − ε)² = SNR_t · (x̂₀ − x₀)²`, this is the extreme loss on each step's estimate of
    the final map. It tilts the samples toward extreme values. w = 1 (each voxel's weight
    1 + W) has no reference value.
- **Why:** Carlos asked whether L_ext could stand in for the asymmetry prior, and where it
  belongs. The predictor term is plug and play. L_diff compares noise, so the term needs the
  translation above.
- **Code:** `models/loss.py` (`ext_weight`, `ext_term`), `models/network.py`,
  `scripts/pretrain_bran.py`; logged as `loss_init_ext` / `loss_diff_ext` in `health/train.csv`.
  Plan: `bran_ext_plan.md` §3.
- **Status:** proposed by Claude and built 2026-10-25; not run.

### 32. 2026-10-25 · The EMA copy averages BatchNorm's statistics too

- **Decision:** `EMA.update_model_average` also averages the floating-point buffers that the two
  models share, the same way it averages parameters, and copies integer ones (BatchNorm's step
  count).
- **Why:** the test samples with the EMA copy (#22). BRAN's BatchNorm statistics are buffers, so
  without this the EMA copy would test 200 epochs of fine-tuned weights under the pretraining's
  statistics. Upstream tests with netG and never met it. gan_bpm's generator and the U-Net have
  no such buffers, so today's runs are unchanged.
- **Alternatives:** test BRAN arms with netG (another rule than #22's); copy, not average, the
  statistics.
- **Code:** `models/model.py`. Plan: `bran_ext_plan.md` §2.4.
- **Status:** adopted (built 2026-10-25).

### 33. 2026-10-25 · One branch, five arms on rCBF fold 0, and how they're judged

- **Decision:** one branch, `bran-ext`. Every extreme-loss arm runs on BRAN, and plain BRAN is
  their control. The arms (`config/ctp_only_f13_rcbf_<arm>_fold0.json`) are `bran_l1`,
  `bran_ext`, `bran_ext_diffext` and `bran_l1_diffext` (a 2 × 2 of the extreme loss in the
  predictor and in L_diff), plus `bran_extlow` as a predictor alone (#34). Each config differs
  from `ctp_only_f13_rcbf_fold0.json` only in its name, `initial_predictor` and `diffusion_ext`.
  `condor/ctp_only_bran_ext.dag` pretrains, checks the BRANs alone, trains, CV-tests (today's
  model too) and scores.
- **Judged (pre-declared):** on fold 0's held-out cases against the label's rCBF < 0.3 core.
  `bran_l1` against today's model is reported as paired differences, with no pick. Among the
  arms, the rule of gan_bpm `ext_gan_plan.md` §1.4 with `bran_l1` as the control: the highest
  Dice at 0.3 whose paired difference has a 95 % CI above 0, with brain MAE at most 1.15 × the
  control's. ISLES 2018 is untouched (#15, #29).
- **Code:** the configs, `condor/ctp_only_bran_ext.dag`, `condor/ctp_only_{train,test}_h100.*`
  (`variant`), `scripts/ctp_only_core_check.py` (`--variant`, `--bran`). Plan:
  `bran_ext_plan.md` §4–§5.
- **Status:** proposed by Claude and built 2026-10-25; Carlos to approve before submitting (about
  120 GPU-hours).

### 34. 2026-10-25 · A low-tail extreme loss, as a predictor alone first

- **Decision:** `side: "low"` keeps W below the label's middle only (rCBF < 1), and the
  `bran_extlow` arm pretrains BRAN with it. It is scored alone against `bran_l1` and `bran_ext`
  before any diffusion run. Every pretraining logs `ext_low_share`, the share of L_ext that comes
  from the low half.
- **Why:** Zeng's W is 1 at rCBF 2 as at rCBF 0, and the error is squared. Vessel voxels, with
  their larger errors, may carry most of L_ext. That would explain why it barely moved gan_bpm's
  core (`model_improvement_plan.md` item 7). This is a hypothesis; the logged share measures it.
- **Next, not built:** a soft-Dice term on rCBF < 0.3 (`bran_ext_plan.md` §3.5), if the weights
  move only the level.
- **Status:** proposed by Claude and built 2026-10-25; not run.

### 35. 2026-10-25 · BRAN's pretraining runs off the H100 box

- **Decision (Carlos):** the pretrainings run on another box, staged the same way the H100 is:
  the data are copied once to that box's `/home/colivares/mrdpm_ctp_only/stage`, and the job
  works there and copies its BRAN to `/radraid2`. The default is the L40S box REDLRADADM35839;
  REDLRADADM35840 (L40S) and REDLRADADM14959 (Quadro RTX 8000) are a `gpu_host` away. The
  MRDPM trainings and the tests stay on the H100.
- **Why:** BRAN's pretraining is small (about 30 M weights; my estimate is a few GB at batch 4,
  not measured), and the H100 cards are what the 20 h trainings need: batch 4 doesn't fit on a
  48 GB card (plan §14).
- **Code:** `condor/ctp_only_bran_pretrain.{sub,sh}` (renamed from `ctp_only_bran_pretrain_h100`,
  which #30 names; default `gpu_host` REDLRADADM35839), `condor/ctp_only_stage_home.{sub,sh}`
  (`gpu_host`; rCBF on a box with no Tmax stage copies gan_bpm's code itself; it checks first
  that the job sees `/home/colivares` and `/radraid2`), `scripts/pretrain_bran.py` (logs peak GPU
  memory), the DAG's `PRE_*` nodes. Plan: `bran_ext_plan.md` §2.5.
- **Not verified:** that docker jobs on those boxes see `/home/colivares` and `/radraid2`. The
  staging job is the first test.
- **Status:** decided (Carlos); built 2026-10-25, not run.

### 36. 2026-10-25 · The pretraining box is REDLRADADM35840, not 35839

- **What happened:** the rCBF staging on REDLRADADM35839 (#35's default) stopped at once: no
  `/home/colivares`. `condor/probe_storage.sub` and `probe_storage_host.sub` (one job per box, in
  our docker image and on the box itself) found that 35839 has none in docker or out, so no mount
  setting would help. 35840 (the other L40S box) has one, writable from a job. 14959 (Quadro) has
  none, but a writable local `/raid/colivares` and a 3.2 TB workspace. Every box sees `/radraid*`.
- **Decision:** the pretraining's default `gpu_host`, the DAG's `PRE_*` nodes and the staging
  command name REDLRADADM35840. Whether docker jobs there see its `/home` is the staging job's
  first check.
- **Alternatives:** on 35839, copy the data into each job's workspace (about 17 GB of its 53 GB)
  and keep the resume point on `/radraid2`; or 14959's local `/raid` or its workspace.
- **Status:** adopted (built 2026-10-25); the docker probe of 35840 and its staging pending.

### 37. 2026-10-25 · BRAN pretrains on 35840 or 14959, each staged on its own disk

- **Decision (Carlos):** stage the rCBF data on two boxes: the L40S box REDLRADADM35840 in
  `/home/colivares/mrdpm_ctp_only/stage`, and the Quadro box REDLRADADM14959 in its local
  `/raid/colivares/mrdpm_ctp_only/stage` (it has no `/home/colivares`, #36). The staging and
  pretraining scripts pick the folder from the box they land on (Condor's machine ad, else the
  hostname): `/raid` on 14959, `/home` elsewhere, so the H100's jobs are unchanged. A
  pretraining runs on whichever of the two has a free GPU (`box_a`, `box_b`; name one twice to
  pin it).
- **Why:** two boxes give the three pretrainings more free GPUs to land on, with no H100 card.
- **Cost:** a job Condor evicts and reruns on the other box starts its BRAN over: the resume file
  is on the first box's disk.
- **Code:** `condor/ctp_only_stage_home.{sub,sh}`, `condor/ctp_only_bran_pretrain.{sub,sh}`, the
  DAG's `PRE_*` nodes (no host now). Plan: `bran_ext_plan.md` §2.5.
- **Status:** decided (Carlos); built 2026-10-25. Whether docker jobs see 35840's `/home` and
  14959's `/raid` is each staging job's first check.

### 38. 2026-10-25 · A pretraining never runs on the CPU; 14959 on hold

- **What happened:** the first debug pretraining (job 1822820.0, `bran_l1`) landed on
  REDLRADADM14959 with a GPU slot, but its container had no GPU (`nvidia-smi -L` failed), and
  `pretrain_bran.py`'s `--device auto` fell back to the CPU without a word. A full run would have
  taken days instead of failing. Everything else worked: the staged data on 14959's `/raid`, the
  4 held-out maps, the copy to `/radraid2/colivares/mrdpm_ctp_only/bran/debug_rcbf_l1/fold_0`.
- **Decision:** the pretraining job checks that torch sees a GPU and fails at once if not,
  printing what Condor assigned and what the container got; it passes `--device cuda`, which
  `pretrain_bran.py` refuses without a GPU. `condor/probe_gpu.sub` asks each box (14959, 35840,
  and 14958 for comparison) whether a GPU reaches a docker job. Until 14959 passes, the DAG pins
  its `PRE_*` nodes to 35840.
- **Status:** adopted (built 2026-10-25). **The GPU probe, 2026-10-25: torch sees 1 GPU in docker
  on 14958 and 35840, 0 on 14959.** 14959's Condor doesn't hand GPUs to docker jobs (an admin
  fix). The pretraining job's default is now 35840 alone, and the DAG names no box.

### 39. 2026-10-26 · gan_bpm scores its own runs; `--run-dir` stays here for one comparison

- **What changed:** gan_bpm's λ_ext pilots and fine-tunes were to be scored by
  `scripts/ctp_only_core_check.py --gan-bpm-only --run-dir` (#29). Carlos asked why a gan_bpm
  experiment goes through this repo, and the scoring moved to gan_bpm's branch
  `ext_gan_25Oct2026` as `scripts/core_check.py` (gan_bpm's D-100).
- **Why:** this script reads gan_bpm through the first checkout (`new_eval_14Oct2026`), whose code
  refuses step 2's relativistic arm, and its pick applies gan_bpm's §1.4 alone: it would have
  judged step 2's fine-tunes by step 1's rule.
- **What stays:** `--run-dir` here, unchanged, for one job: gan_bpm's scorer must reproduce it on
  the three step-1 pilots scan for scan (`--compare-with`) before it judges any fine-tune. After
  that, nothing in gan_bpm's experiments is scored here; MRDPM's own core check (#29) and
  `--bran` are unaffected.
- **Status:** adopted.

---

## Open

The current setting is in brackets.

- **The evaluation build** (`ctp_only_eval_plan.md` §7): EMA testing, the test job, ISLES 2018
  on the H100, the map converter and the ISLES 2018 fold-model scoring [built, not yet run on
  the box]; the CV scoring script and the all-data ISLES 2018 comparison [not built].
- **Full-frame metrics on ISLES 2018** (#25) [not built].
- **Validity checks on training cases** (plan §17): repeat the masking check and the
  orientation look on training cases [not done].
- **gan_bpm's `best.pt` was chosen on the held-out fold** (plan §17) [proposal: disclose].
- **Holes, measured** (#19, plan §7.2): after the first fold of each target, MRDPM's error on
  valid voxels next to no-data holes against its error far from them [not done].
- **The evaluation plan** covers rCBF too (#15): rCBF < 0.3 Dice and volume beside the MAE.
- **An optional ablation:** BRAN from scratch on 3 channels.
- **Map figures on the CV held-out cases and hard training slices** (#27): raw input, model
  input, generated and true rCBF [bookmarked, not built].
- **Why the core is missed** (2026-10-25 discussion): the "right place or wrong level" check
  on the CV held-out cases (#28, #29) [built; gan_bpm's side run 2026-10-25, MRDPM's needs the
  CV tests]. It also scores gan_bpm's λ_ext pilots (`--run-dir`, at `last.pt`) and applies their
  pre-declared pick rule (gan_bpm `ext_gan_plan.md` §1.4, D-98); gan_bpm's 2g, the
  same on ISLES 2018 against DWI (in gan_bpm's uncommitted working tree) [not run]; Zeng's
  10–80 HU window as a reported region; several samples per slice instead of one (would
  revise #22); Zeng's spatial and temporal smoothing [not started]. How much of gan_bpm's
  gradient L_ext carries at the trained fold models (`scripts/ctp_only_ext_share.py`; gan_bpm
  measured it at initialization only) [measured 2026-10-25, rCBF λ_adv = 0: L_ext supplies
  6–20 % of the step at `best.pt` (epochs 7–16 of 40), 30 % at initialization; L_pix 80–94 %;
  cos(pix, ext) 0.27 / 0.80 / 0.79 / −0.08 / 0.35 for folds 0–4. The core check on the CV
  held-out cases, gan_bpm only: Dice at 0.3 is 0.42, matched 0.48, floor 0.06, volume 0.59 × the
  label's: mostly a place problem. `/radraid2/colivares/mrdpm_ctp_only/{ext_share,cv_core_check}/`].
- **The BRAN and extreme-loss arms** (#30–#38, branch `bran-ext`, `bran_ext_plan.md`): the box
  checks (`scripts/bran_ext_checks.py`, `scripts/ctp_only_checks.py`), a debug pretraining and
  training of `bran_l1`, then `condor/ctp_only_bran_ext.dag` [built, not run]; the recipe
  choices of #30 and the masked input [Carlos to confirm]; the soft-Dice term [not built].
