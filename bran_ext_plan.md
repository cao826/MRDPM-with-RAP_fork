# bran-ext plan: the paper's own predictor (BRAN), and Zeng's extreme loss inside MRDPM

Branch **`bran-ext`**, from `ctp-only` at `836ff8b`. Started 2026-10-25 from Carlos's
`model_improvement_plan.md` (items 1 and 7). **Status: built, checks not yet run on the box, nothing
trained.** Decisions #30–#34 in `ctp_only_decisions.md`. Everything new is behind config keys
that are off in the existing configs, so the gan_bpm-predictor runs and their tests behave as
before. `scripts/ctp_only_checks.py` must still pass to confirm it.

---

## 1. The questions, answered short

| question | answer |
|---|---|
| What is the original predictor? | **BRAN** (paper §2.3, `models/region_net.py:Unetgenerator_256`): a pix2pix U-Net, 8 levels deep, with a region-aware conv before each encoder level, BatchNorm, and DropBlock in 3 decoder levels. Channel progression in §2.1. |
| What loss? | Pretraining: **L1 alone** (paper Eq. 1). Then, inside diffusion training, it is fine-tuned with `L = L_diff + 0.1 · L1` (Eq. 4). |
| How many epochs? | Pretraining: **50**, Adam, learning rate falling linearly to 0 (upstream README; the paper doesn't say). Diffusion stage: 200 (the recipe we already run). |
| What else is missing? | The README and paper don't give the pretraining's learning rate, batch size, Adam betas or weight init. We use pix2pix's defaults, which BRAN's code comes from (§2.2). The paper also skull-strips the input, which today's runs don't do for the predictor. |
| Steps? | (1) Pretrain BRAN alone. (2) Train MRDPM with it: the diffusion U-Net learns, and BRAN is fine-tuned in the same run (there is no separate fine-tuning stage). (3) Sample the held-out fold and score it. |
| Can L_ext plug into the predictor? | **Yes, directly**: one more term in its loss, both in pretraining and in the joint L_init. |
| Into the diffusion step? | **Not as written.** L_diff compares noise, not maps. The weighted analogue is principled and about 10 lines (§3.3). |
| Predictor, diffusion or both? | Unknown, so the experiment is a 2 × 2 on fold 0 (§5). |
| One branch or two? | **One.** Every L_ext arm needs the BRAN code, and BRAN with plain L1 is the control for all of them (§4). |
| A better idea than L_ext? | Probably. Zeng's weight is 1 at both ends of the range, so for rCBF it weights vessels (rCBF ≥ 2) as much as the core. With a squared error, the vessels' larger errors may dominate it. We test a **low-tail-only** weight cheaply (§3.5). The next step, not built: a soft-Dice term on rCBF < 0.3 itself. |

**What the paper's ablation says about registration** (Table 2, Tmax MAE): adding the mask takes it
from 0.065 to 0.051, residual diffusion to 0.050, BRAN to 0.049, and the asymmetry module (MAP,
whose mirror needs registration) to 0.047. So BRAN is worth 0.001 and MAP 0.002, against 0.014
for the mask. The mirror's own share inside MAP isn't reported, and the paper reports no core or
Dice number at all. **Nothing in the paper suggests that registering our data for the flip would
fix the core**, so this branch leaves the flip off (decision #10 stands).

---

## 2. The paper's predictor, as close as our data allows

### 2.1 BRAN as the code builds it (verified, `models/region_net.py`, `models/DRconv1.py`)

`define_G` always builds `Unetgenerator_256(ngf=32)`. Every encoder level:

1. A **region-aware 1 × 1 conv** (`DRConv2d_v8_inception`, paper Fig. 3, Eq. 10–13):
   - a 1 × 1 conv sorts each pixel into one of 4 regions (argmax);
   - each region gets its own 1 × 1 filter, generated per image from the image's average
     (two filter sets, summed);
   - so it mixes channels pixel by pixel, differently per region, with no spatial extent
     (`kernel_size=1` everywhere).
2. Its output is concatenated with the level's input.
3. Then a 4 × 4 stride-2 conv, BatchNorm (not at levels 1 and 8), LeakyReLU 0.2 (level 8: ReLU).

Each decoder level is a 4 × 4 stride-2 transposed conv, BatchNorm, DropBlock 0.5 (levels 2–4),
ReLU, then a concat with the encoder's skip. The last one ends in Tanh.

Widths at our 512 × 512, with 3 input channels (upstream: 4):

| encoder | in | region conv + concat | out | size |
|---|---|---|---|---|
| d1 | 3 | 6 | 32 | 256 |
| d2 | 32 | 64 | 64 | 128 |
| d3 | 64 | 128 | 128 | 64 |
| d4 | 128 | 256 | 256 | 32 |
| d5 – d8 | 256 | 512 | 256 | 16, 8, 4, 2 |

| decoder | in | out | + skip = | size | |
|---|---|---|---|---|---|
| up1 | 256 (d8) | 256 | 512 | 4 | |
| up2 – up4 | 512 | 256, DropBlock | 512 | 8, 16, 32 | |
| up5 | 512 | 128 | 256 | 64 = H/8 | → `hs[2]`, 256 channels |
| up6 | 256 | 64 | 128 | 128 = H/4 | → `hs[1]`, 128 channels |
| up7 | 128 | 32 | 64 | 256 = H/2 | → `hs[0]`, 64 channels |
| up8 | 64 | 1, Tanh | | 512 | the map |

`hs` are exactly the widths `cond_resample` was built for, so nothing is duplicated as it is for
gan_bpm's generator (plan §8.1).

**Against gan_bpm's generator** (`gan_bpm/models/generator.py`): a 3D temporal stem (the 3 phases
as depth, 64 → 128 channels, collapsed to one), then a U-Net with 6 encoder levels of 32, 64,
128, 128, 128, 128 channels and nearest-upsample + 3 × 3 conv decoders. Its widths stop at 128,
not 256. It uses InstanceNorm (no running statistics), has no dropout and no region-aware convs,
and feeds the U-Net half-width features (32 / 64 / 128), duplicated to fit.

### 2.2 What we keep, and what we can't

| | paper / upstream code | this branch | why |
|---|---|---|---|
| condition | NCCT + 3 mCTA phases (4 channels) | 3 CTP phases (f13) | no NCCT (ctp-only goal) |
| predictor | BRAN, ngf 32 | **BRAN, ngf 32, 3 input channels** | `Unetgenerator_256(cond_channels=3)`; upstream's default stays 4 |
| its input | skull-stripped (§3), the same masked condition as the U-Net | **condition × envelope** (`mask_cond`) | the paper's; today's runs give gan_bpm's generator the whole head (#6), because it was trained that way |
| pretraining loss | L1 (Eq. 1) | L1 over `support` | our labels have no-data holes (#7) |
| pretraining recipe | 50 epochs, Adam, LR linear to 0 (README) | the same; **lr 2e-4, betas (0.5, 0.999), N(0, 0.02) init** | not given; pix2pix's defaults. BRAN's `define_G` has pix2pix's signature, the weights file is `tmax_net_G.pth` (pix2pix's `net_G` naming), and "50 epochs with linear decay" is pix2pix's LR policy |
| pretraining batch | not given | **4** | the paper's diffusion batch, so BatchNorm's statistics are the ones the joint stage trains with |
| model chosen | not said | **the last epoch** | nothing is selected on any held-out data (gan_bpm's `best.pt` was, #2) |
| joint stage | L_diff + 0.1 L_init, Adam 5e-5, 200 epochs, EMA 0.9999, batch 4 | unchanged | |
| resolution | 256 × 256 at 0.625 mm, cropped and centred | 512 × 512 as acquired | our data and evaluation; BRAN works at any multiple of 256 |
| asymmetry flip (MAP) | on | **off** (`mirror_hs: false`) | scans not aligned (#10, §1) |
| labels | CBF [0, 200] absolute | rCBF [0, 2] → [0, 1] | our label (#18) |
| testing | netG | EMA weights (#22) | **BatchNorm made this need a fix (§2.4)** |

### 2.3 The steps

1. **Pretrain BRAN** on fold k's training cases (`scripts/pretrain_bran.py`, job
   `condor/ctp_only_bran_pretrain.sub`). It reads everything from the arm's MRDPM config,
   so it sees exactly the data, split and monitor carve the MRDPM run will. At the end it maps
   fold k's held-out cases (Init_/GT_/Mask_), so BRAN alone can be scored before any diffusion.
   It writes `/radraid2/colivares/mrdpm_ctp_only/bran/<name>/fold_<k>/`: `bran_G.pth` (weights
   plus a `meta` record), `pretrain_log.csv`, `heldout/`.
2. **Train MRDPM** with it (`condor/ctp_only_train_h100.sub -append 'variant = <arm>'`). This
   is today's recipe, with BRAN in place of gan_bpm's generator. `Network.init_weights` loads
   BRAN after BaseNetwork's re-initialization (the trap of plan §8.4), and refuses a checkpoint
   whose `meta` disagrees with the config: target, fold, channels, masked input, loss or recipe.
3. **Sample fold k's held-out cases** (`condor/ctp_only_test_h100.sub`; the run's name carries
   the arm), then **score** with `scripts/ctp_only_core_check.py --variant <arm> --bran <dir>`.

### 2.4 EMA and BatchNorm (a fix this needed)

The EMA copy, which the test samples with (#22), averaged only *parameters*. BatchNorm's running
mean and variance are *buffers*, so the EMA copy kept the pretraining's statistics under 200
epochs of fine-tuned weights. Upstream never noticed because it tests with `netG`. gan_bpm's
generator and the U-Net have no such buffers. `EMA.update_model_average` now averages
floating-point buffers the two models share, the same way it averages weights (timm's
`ModelEmaV2` does this), and copies the step counts. For today's runs it changes nothing.

### 2.5 Where the pretraining runs: off the H100 box (decisions #35)

The H100s are for the 20 h MRDPM trainings and the tests. BRAN's pretraining runs on another
box: whichever of the L40S box **REDLRADADM35840** (8 × 46 GB) and the Quadro box
**REDLRADADM14959** (10 × 48 GB) has a free GPU (decisions #36–#37). It fits on any of them. The
model is about 30 M weights, against the U-Net's 68.5 M, and my estimate is a few GB at batch 4.
That's an estimate, not a measurement: the log's `gpu_peak_gib` column gives the real number
from the first debug run. The L40S is the faster card.

It works as on the H100: the data are copied once from `/radraid2/colivares/mrdpm_ctp_only/stage/`
to each box's own disk (`ctp_only_stage_home.sub` with `gpu_host`, rCBF only, about 17 GB):
`/home/colivares/mrdpm_ctp_only/stage` on 35840, the local `/raid/colivares/mrdpm_ctp_only/stage`
on 14959, which has no `/home/colivares`. The staging and pretraining scripts pick the folder
from the box they land on. The job works on that box's disk, and the finished BRAN goes to
`/radraid2/colivares/mrdpm_ctp_only/bran/`, where the H100 trainings read it. On a box with no
Tmax stage, the rCBF staging copies gan_bpm's code itself (the H100's still reuses Tmax's).

**Storage, probed 2026-10-25** (`condor/probe_storage.sub`, `probe_storage_host.sub`): 35839 has
no `/home/colivares` at all, in docker or out; 35840 has one, writable; 14959 has none, but a
writable local `/raid/colivares` and a 3.2 TB workspace. Every box sees `/radraid*`. Whether
docker jobs see 35840's `/home` and 14959's `/raid` is checked by each staging job first, which
fails at once if not. A pretraining Condor evicts and reruns on the other box starts over.

**14959's GPU, 2026-10-25 (decisions #38):** the first debug pretraining landed on 14959, got a
GPU slot, saw no GPU in its container, and ran on the CPU. The job now refuses to run without a
GPU, `condor/probe_gpu.sub` tests each box, and the DAG pins its pretrainings to 35840 until
14959 passes. The account's
3-GPU cap is cluster-wide, so this frees H100 cards but adds no GPUs.

---

## 3. The extreme loss

### 3.1 What it is

Zeng et al., supplement Eq. 4: `L_ext = E[W_B · (B − G(A))²]`, with `W_B = B ⊙ B` and B the label
in gan_bpm's [−1, 1]. It is weighted 50, against L1's 100. It is "to remind the model to pay
attention on infarct region and high blood flow region". On MRDPM's [0, 1] label, `B = 2·gt − 1`,
so `W = (2·gt − 1)²`:

- **rCBF:** W = 1 at rCBF 0 (dead tissue) and at rCBF 2 (vessels, hyperaemia); 0 at rCBF 1
  (normal); 0.49 at rCBF 0.3 (the core's edge).
- **Tmax:** W = 1 at 0 s (normal tissue!) and at 12 s; 0 at 6 s, the lesion threshold. So for
  Tmax it weights normal tissue most.

### 3.2 In the predictor: plug and play (built)

`initial_predictor.ext = {"weight": w, "side": "both"}` adds `w · mean(support · W · (pred − gt)²)`
to BRAN's pretraining loss and to the joint stage's L_init. **`w = λ_ext / 50`** reproduces
gan_bpm's balance exactly: 100·L_pix + λ_ext·L_ext on [−1, 1] equals 200 × (L1 + (λ_ext/50)·L_ext)
on [0, 1] (`scripts/bran_ext_checks.py` checks this against gan_bpm's own `losses.py`). So Zeng's
λ_ext = 50 is **w = 1**, the arms' value. gan_bpm's running sweep (200, 500) would be w = 4, 10.
It works with gan_bpm's predictor too (L_init only, since that one isn't pretrained here).

### 3.3 In the diffusion loss: not as written; the analogue (built)

L_diff is `‖ε − ε̂‖²`, the error in predicting the added *noise*. Noise has no extremes, so
Zeng's term has nothing to act on. But for one noise level t,

    x̂₀ − x₀ = −√((1 − γ̄_t) / γ̄_t) · (ε̂ − ε),   so   (ε̂ − ε)² = SNR_t · (x̂₀ − x₀)²

and x̂₀ + y_initial is that step's estimate of the final map, while x₀ + y_initial is the label.
So weighting each voxel's `(ε − ε̂)²` by `W(gt)` *is* the extreme loss on each step's estimate
of the final map, at the usual ε-prediction weighting. `diffusion_ext = {"weight": w}` adds
`w · mean(support · W(gt) · (ε − ε̂)²)`: each voxel's weight becomes 1 + w·W.

What it does, plainly: the denoiser can't see gt, so for a voxel that might be core or might not,
it learns a compromise tilted toward the heavier-weighted answer. **Its samples come out pushed
toward the extremes**: more core called, and more very high values. There is no paper value for
w. The arms use 1, which doubles the weight of the most extreme voxels.

### 3.4 Predictor, diffusion, or both?

- **The squeeze** (a predicted core 0.59 × the label's; spread 0.81) comes from L1. An L1-trained
  map is the conditional median, which calls core only where the model is more than half sure.
- **In MRDPM the diffusion stage was supposed to add the spread back**: it samples the residual
  instead of averaging it. On ISLES 2018 it didn't (MRDPM's bias and spread are gan_bpm's).
- **L_ext in the predictor** moves `y_initial`, the centre the residual is built around.
- **L_ext in L_diff** acts on the stage that is supposed to produce the tails.

We don't know which matters, so we run both and both together (§5). One caution from the core
check: gan_bpm's main shortfall is **place**: only about half its k lowest voxels land in the
core. A loss weight mostly moves **level**. If only the Dice at the cutoff and the volume move,
and the matched Dice doesn't, that is what happened.

### 3.5 A better idea than Zeng's weight?

1. **The weight is two-sided, and the vessels may drown the core** (a hypothesis, measured here).
   Vessel voxels (rCBF ≥ 2, about 12 % of UCLA's positive voxels clip at 2) get W = 1, and their
   errors are larger than the core's. Squared, a vessel voxel can contribute several times what a
   core voxel does. That would explain why L_ext barely moved gan_bpm's core
   (`model_improvement_plan.md` item 7). Two things in this branch test it:
   - every pretraining logs `ext_low_share`: the fraction of L_ext that comes from the low half
     (gt < rCBF 1), on train and monitor;
   - the **`bran_extlow` arm** keeps only the low half (`side: "low"`, W = (1 − 2gt)² below the
     middle, 0 above). It is scored as a predictor alone first, which is cheap.

   For Tmax the side to keep would be `"high"`.
2. **Optimize the reported overlap itself** (not built): a soft-Dice term between
   `σ((0.15 − pred)/τ)` and `[gt < 0.15]` (rCBF < 0.3). Dice normalizes by the core's size, so
   small cores count, and it rewards place as well as level. Build it if §5 shows a weight moving
   only the level.

---

## 4. One branch

`bran-ext` holds both pieces. The extreme-loss arms run on BRAN, and BRAN with plain L1 is their
control, so two branches would duplicate the BRAN code or need a merge before anything runs. All
new behaviour is behind config keys (`initial_predictor.type: "bran"`, `ext`, `mask_cond`,
`diffusion_ext`) that the existing configs don't set. It can merge into `ctp-only` whenever.

On the box: the jobs ship the 14958 checkout at submission (`transfer_input_files`), so switching
`/raid/colivares/github_repos/MRDPM-with-RAP_fork` to `bran-ext` changes nothing for jobs already
running. The scoring still imports gan_bpm from its first checkout, which stays as it is.

---

## 5. The experiments (rCBF, fold 0, the development set of #29)

### 5.1 The arms

| arm (config `ctp_only_f13_rcbf_<arm>_fold0.json`) | BRAN pretrained with | L_init | L_diff | answers |
|---|---|---|---|---|
| *today's model* (`ctp_only_f13_rcbf_fold0`) | (gan_bpm λ_adv = 0, `best.pt`) | L1 | plain | the control |
| `bran_l1` | L1 | L1 | plain | Q1: the paper's predictor |
| `bran_ext` | L1 + L_ext | L1 + L_ext | plain | L_ext in the predictor |
| `bran_ext_diffext` | L1 + L_ext (bran_ext's BRAN) | L1 + L_ext | weighted | both |
| `bran_l1_diffext` | L1 (bran_l1's BRAN) | L1 | weighted | L_ext in the diffusion only |
| `bran_extlow` | L1 + low-half L_ext | (not trained) | | §3.5, as a predictor alone |

### 5.2 Order and cost: `condor/ctp_only_bran_ext.dag`

1. `PRE_L1`, `PRE_EXT`, `PRE_LOW`: three pretrainings on 35840 or 14959 (§2.5), a few hours
   each (to measure).
2. `CHECK_PRE` (14958, minutes): BRAN alone against gan_bpm's `best.pt` on fold 0's held-out
   cases (`--gan-bpm-only --bran …`).
3. `TRAIN_*`: four MRDPM trainings, about 20 h each.
4. `TEST_*`: CV tests of the four arms and of today's model (`TEST_BASE`), about 6 h each.
   Delete every line naming `TEST_BASE` if today's fold-0 CV test already exists.
5. `CHECK_ALL` (14958): everything on the same slices (`--variant … --bran …`).

About 120 GPU-hours. With 3 GPUs, roughly 2 days of wall-clock time. The account holds 3 GPUs
across the cluster, and gan_bpm's sweep uses some, so lower `MAXJOBS` accordingly.

### 5.3 How they're judged (pre-declared, before any arm is trained)

On fold 0's held-out cases, each scan a unit, against the label's rCBF < 0.3 core
(`scripts/ctp_only_core_check.py`):

- **Q1, the paper's predictor:** `mrdpm_bran_l1` against `mrdpm` (today's) and against gan_bpm.
  Reported as paired differences with 95 % bootstrap intervals: Dice at 0.3, matched Dice,
  brain MAE. There is no pick: whichever way it goes, the comparison is the answer.
- **The extreme-loss arms:** the rule of gan_bpm `ext_gan_plan.md` §1.4, with `bran_l1` as the
  control. The pick is the arm with the highest Dice at 0.3 whose paired difference from the
  control has a 95 % CI above 0, and whose brain MAE is at most 1.15 × the control's. The
  summary's `PICK among the MRDPM arms:` line applies it. The same rule over the BRANs alone
  (control `rcbf_l1`) is `PICK among the pretrained BRANs:`.
- **Reading place against level:** the matched Dice moves with place, and the Dice at the cutoff
  and the volume ratio move with level (#28).
- Nothing here touches ISLES 2018. A winner goes to all 5 folds first.

---

## 6. Checks (run on the box, not the laptop)

```bash
cd /raid/colivares/github_repos/MRDPM-with-RAP_fork && git fetch && git switch bran-ext && git pull
docker run --rm --shm-size=8g --user "$(id -u):$(id -g)" -e NVIDIA_VISIBLE_DEVICES=void \
    -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -v /raid:/raid -v /radraid2:/radraid2 \
    -v "$PWD:/repo" -w /repo registry.cvib.ucla.edu/colivares:mrdpm-env-v1 \
    bash -c 'python scripts/bran_ext_checks.py; python scripts/ctp_only_checks.py'
```

Both summaries must pass. `scripts/bran_ext_checks.py` (synthetic, CPU) covers:

- BRAN at 3 channels, and upstream's at 4;
- the extreme loss against gan_bpm's `losses.py` at λ_ext 50, 200 and 500;
- `pretrain_bran.py` end to end, stopped and resumed, with refusals and its held-out maps;
- the checkpoint guard (9 mismatches refused);
- `Network`'s masked predictor input and the exact loss terms, with no-data holes;
- the EMA of BatchNorm statistics;
- run.py train and test with a BRAN arm (the test without the pretraining checkpoint);
- the five configs and the DAG;
- the run-name patterns, bash's included;
- the three jobs' config-rewriting Python, on fake stage folders.

`ctp_only_checks.py` covers the code that changed under today's configs.

Then the data on the pretraining box (once), and a debug pass before the DAG:

```bash
cd condor && mkdir -p log
condor_submit ctp_only_stage_home.sub -append 'target = rcbf' -append 'gpu_host = REDLRADADM35840'  # no GPU
condor_submit ctp_only_stage_home.sub -append 'target = rcbf' -append 'gpu_host = REDLRADADM14959'  # no GPU
condor_submit ctp_only_bran_pretrain.sub -append 'variant = bran_l1'            # 35840 or 14959, 1 epoch -> bran/debug_rcbf_l1/
condor_submit ctp_only_train_h100.sub -append 'target = rcbf' -append 'variant = bran_l1'   # H100, uses the debug BRAN
```

Delete `/radraid2/colivares/mrdpm_ctp_only/bran/debug_rcbf_l1` afterwards if you want to rerun the
debug.

---

## 7. Files

| file | change |
|---|---|
| `models/region_net.py` | `Unetgenerator_256(cond_channels=4)`: the condition's channel count (was always 4) |
| `models/bran_predictor.py` | new: `BranInitialPredictor`, the checkpoint guard, `PRETRAIN_DEFAULTS` |
| `models/loss.py` | `ext_weight`, `ext_term` |
| `models/network.py` | `initial_predictor.type: "bran"`; `Network.initial` (the predictor's input rule, used everywhere); `ext` and `diffusion_ext` terms in the `loss_mask` branch, logged as `loss_init_ext` / `loss_diff_ext` |
| `models/model.py` | the EMA averages shared buffers; the test's `Init_` goes through `Network.initial` |
| `models/health.py` | the monitor goes through `Network.initial`; extra loss terms get their own columns |
| `scripts/pretrain_bran.py` | new: the pretraining |
| `scripts/ctp_only_core_check.py` | `--variant`, `--bran`, `--allow-debug-bran`; arm-aware names; every map on the slices they all have; picks among arms and among BRANs |
| `scripts/bran_ext_checks.py` | new: §6 |
| `config/ctp_only_f13_rcbf_{bran_l1,bran_ext,bran_ext_diffext,bran_l1_diffext,bran_extlow}_fold0.json` | new: rCBF fold 0's config with only the name, `initial_predictor` and `diffusion_ext` changed |
| `condor/ctp_only_bran_pretrain.{sub,sh}` | new; runs off the H100, on 35840 or 14959 (`box_a`, `box_b`), files on that box's own disk |
| `condor/ctp_only_stage_home.{sub,sh}` | `gpu_host`; the folder by box (14959: local `/raid`); rCBF on a box with no Tmax stage copies gan_bpm's code itself |
| `condor/ctp_only_train_h100.{sub,sh}` | `variant` macro; a BRAN arm's predictor is copied from `/radraid2` to `/home` |
| `condor/ctp_only_test_h100.{sub,sh}` | arm-aware run names; `run` may be a name without its time (the DAG's) |
| `condor/ctp_only_bran_ext.dag` | new: §5.2 |

## 8. Open

- **Pretraining time and memory** at 512, batch 4, on the L40S: unmeasured. The debug job prints
  seconds and peak GPU memory per epoch.
- **The diffusion weight** (w = 1) has no reference value. If `bran_*_diffext` over-calls the core
  (volume ratio well above 1), try 0.5.
- **gan_bpm's λ_ext sweep** may pick 200 or 500. Then an arm at w = 4 or 10 is a config copy.
- **Folds 1–4 and ISLES 2018** for whichever arm wins: the folds first (`fold` macro). For
  ISLES 2018, an all-data BRAN and an all-data config (not built).
- **The soft-Dice term** (§3.5) if the weights move only the level.
- **Tmax arms:** not configured. For Tmax, Zeng's weight emphasizes normal tissue, so its arm
  would want `side: "high"`.
