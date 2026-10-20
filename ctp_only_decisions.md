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

---

## Open

The current setting is in brackets.

- **The evaluation plan** (#15, #21): drafted in `ctp_only_eval_plan.md`; its §9 to settle,
  then freeze.
- **Validity checks on training cases** (plan §17): repeat the masking check and the
  orientation look on training cases [not done].
- **gan_bpm's `best.pt` was chosen on the held-out fold** (plan §17) [proposal: disclose].
- **EMA at test** [the code validates and tests with `netG`, not the EMA copy; proposal:
  EMA, `ctp_only_eval_plan.md` §9.1].
- **Holes, measured** (#19, plan §7.2): after the first fold of each target, MRDPM's error on
  valid voxels next to no-data holes against its error far from them [not done].
- **The evaluation plan** covers rCBF too (#15): rCBF < 0.3 Dice and volume beside the MAE.
- **An optional ablation:** BRAN from scratch on 3 channels.
