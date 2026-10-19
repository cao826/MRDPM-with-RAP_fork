# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A fork of the anonymized code release for **MRDPM-with-RAP** (Masked Residual Diffusion Probabilistic Model with Regional Asymmetry Prior), built on the Palette image-to-image diffusion codebase. It generates a perfusion parameter map (**Tmax**) from a 4-channel conditioning stack: in our use, an **NCCT plus 3 CTP timepoints** (the code and data folders still call this stack `mCTA`).

**Goal of the `ctp-only` branch:** a variant that conditions on the **3 CTP timepoints only, with no NCCT**, trained on our public and private (UCLA) datasets, with as few model edits as possible. The upstream 4-channel path on `main` is the reference and should stay runnable.

## Commands

There is no test suite, linter, or build step. Everything runs through `run.py` plus one JSON config.

```bash
# Train (default config: config/npy_1000linear_-6_-2_b4_nc64.json)
python run.py -p train -c config/npy_1000linear_-6_-2_b4_nc64.json

# Debug run: prefixes the run name with debug_, merges the config's "debug" block over "train",
# and cuts the dataset to debug_split samples
python run.py -p train -d

# Override the batch size per GPU / select GPUs (more than one id switches to DDP via mp.spawn + NCCL)
python run.py -p train -b 4 -gpu 0,1 -P 21012

# Test: set path.resume_state in the config first (see Checkpoints). Writes .npy outputs, then PNGs via npy2img
python run.py -p test

# FID / Inception Score between two image folders (inherited from Palette)
python eval.py -s <gt_dir> -d <generated_dir>
```

Run from the repo root. `core/praser.py` backs up `config/ models/ core/ data/` and every top-level `.py`/`.sh` in the cwd into each experiment folder.

**A CUDA GPU is required.** `Network.set_new_noise_schedule` puts its buffers on `cuda` by default, and multi-GPU training uses NCCL. This will not train on a Mac.

**`requirements.txt` is incomplete.** `models/network.py` also imports `torchviz` and `pydot_ng` at module top level (they are unused, but the import fails without them), `models/region_net.py` imports `einops`, and `matplotlib`/`Pillow` are used too.

**`random_data.py` only works on Windows.** It saves to paths like `.\\dataset\\Tmax\\000.npy` and expects `dataset/{Tmax,mask,mCTA}/` to exist already. On macOS or Linux it writes files with backslashes in their names into the cwd.

## Architecture

### Config-driven construction

`core/praser.py:parse` reads the JSON config after removing everything from `//` to the end of each line (so a `//` inside a string value would also be cut). `init_obj` imports whatever `"name": ["module.path", "ClassName"]` names and calls it with `"args"` as kwargs. That is how the dataset, the model wrapper (`Palette`), the network, the losses and the metrics are all chosen. A new dataset class or network variant is normally a new class plus a config that points at it, with no change to `run.py`.

Every run creates `experiments/{phase|debug}_{name}_{YYMMDD_HHMMSS}/` containing `config.json`, `code/`, `checkpoint/`, `results/` and `tb_logger/`.

### Data contract (`data/dataset.py:NumpyDataset_mask`)

- `data_flist` is a text file with one sample id per line and no extension. Each sample is read from `{data_root}/Tmax/{id}.npy` (H×W), `{data_root}/mCTA/{id}.npy` (H×W×4), and `{data_root}/mask/{id}.npy` (H×W brain mask).
- Target and conditioning are both **clamped to [0, 1]**, and the conditioning is multiplied by the mask (`np.concatenate([mask]*4)`, so the channel count is hardcoded here). `gt_image = Tmax * mask`.
- The validation set is carved out of the **train** list (`validation_split`: an int is a count, a float is a fraction), with a split seeded from `opt['seed']`. The test phase uses the separate `test_list.txt`.

### Two-stage model (`models/network.py:Network`)

1. **Initial predictor** (`models/region_net.py:Unetgenerator_256`, created through `networks_unet256.define_G`): a pix2pix-style U-Net whose encoder interleaves DRConv region-aware dynamic convolutions (`models/DRconv1.py`). It maps cond → a coarse Tmax `y_initial` and also returns three decoder feature maps `hs = [u7, u6, u5]` (64/128/256 channels at H/2, H/4, H/8).
2. **Diffusion U-Net** (`models/guided_diffusion_modules/unet.py:UNet`): learns the **residual** `(y_0 − y_initial) * mask`. Its input is `cat(cond, y_noisy)`, so `unet.in_channel = cond_channels + 1` (currently 5).
3. **Regional Asymmetry Prior**: before input blocks 5, 8 and 11, `cond_resample` fuses the matching `hs` feature with its mirror image (`torch.flip(cond, [2])`) and adds the result to the U-Net activations. The flip is along **dim 2 (H, the rows)**, so it is only a hemispheric mirror if the left/right axis of the brain runs along the array rows. Check the orientation of any new data against this.
4. `loss = MSE(mask*noise, mask*noise_hat) + 0.1 * L1(y_0, y_initial)` (paper Eq. 4). The initial predictor shares the optimizer and gets gradients from both terms: directly from the L1 term, and from the diffusion term through `y_initial` → the residual → the noisy U-Net input. Only the `hs` feature path is `.detach()`ed (the "gradient stop" in the paper's Fig. 2).
5. Inference (`restoration`) runs the full 1000-step ancestral sampler inside the mask and returns `y_t + y_initial`.

**Hardcoded shape coupling**, which matters for any change of channel count or size:

- **Conditioning channel count** appears in three places that must agree: `unet.in_channel` in the config (cond + 1), the `Encoderv1x(in_channels=4)` default in `region_net.py` (the `input_nc` argument of `define_G` is ignored), and the mask `concatenate` in the dataset. These are the edits needed for the 3-channel CTP-only variant.
- **The `cond_resample(64,128) / (128,256,64) / (256,512,192)` widths** and the block indices 5, 8 and 11 assume `inner_channel=64`, `channel_mults=[1,2,4,8]`, `res_blocks=2` and an initial predictor with `ngf=32`. Changing any of these breaks the fusion.
- **Spatial size**: the initial predictor downsamples 8 times by stride 2, so H and W must be multiples of 256.

**Pretrained initial predictor**: the paper (§2.1, Eq. 1) pre-trains the initial predictor on its own with an L1 loss before diffusion training, and then fine-tunes it jointly. In the code, `BaseNetwork.init_weights` loads `./tmax_net_G.pth` (strict) whenever that file exists in the cwd, and only in the train phase. If the file is missing, the predictor silently starts from random weights. No weights or pretraining script are included. The upstream README says it is simply trained for 50 epochs with Adam, with the learning rate decayed linearly to zero (the paper does not give these settings).

The paper is `1246_paper.pdf` at the repo root (untracked). It describes the asymmetry prior as a *horizontal* flip, but the code flips dim 2 (rows). The flip has to match whichever axis separates the hemispheres in our data.

### Training wrapper (`models/model.py:Palette`, `core/base_model.py:BaseModel`)

- The only working `task` is `my_mission_mask`. `val_step` and `test` dispatch on that string alone. `restoration_ddim` exists but is not wired in, and it calls `denoise_fn` without the cond features, so it would crash.
- The LR schedule is hardcoded (`LambdaLR`: constant until epoch 100, then linear to 0 at epoch 200), and `which_lr_schedulers` in the config is ignored. That scheduler is not in `self.schedulers`, so it is not saved and restarts from scratch when training resumes.
- EMA weights (`netG_EMA`) are updated and checkpointed, but validation and test use `netG`, not the EMA copy.
- `self.iter` counts **samples**, not optimizer steps (`iter += batch_size`). Train logging fires only when `iter % log_iter == 0`, so a batch size that does not divide `log_iter` never logs.
- The `mae` metric is computed over the whole image, background included, on the [0, 1] scale.

### Checkpoints and outputs

- Every `save_checkpoint_epoch`, the run writes `checkpoint/{epoch}_Network.pth`, `{epoch}_Network_ema.pth` and `{epoch}.state`. To resume or test, set `path.resume_state` to the prefix without an extension, e.g. `experiments/train_mCTA2CTP_240308_131346/checkpoint/10`. Networks load with `strict=False`, so missing or extra keys are skipped silently, but a shape mismatch still raises.
- `test()` appends `resume_state` and the MAE to `./ddim_test.txt`, so it crashes if `resume_state` is null.
- Results go to `results/{phase}/{epoch}/` as `GT_`, `Out_`, `Process_` (the sampling trajectory) and `Mask_` `.npy` files. After testing, `run.py` converts everything except `Process_` to PNG.

## The ctp-only path (this branch)

`ctp_only_plan.md` is the design record: what was verified, what was decided, what is still open. What exists so far:

- **Config:** `config/ctp_only_f13_tmax_fold0.json`, f13's 3 CTP phases → Tmax on fold 0 of gan_bpm's k=5 split, with the GPU box's paths.
- **Data:** `data/dataset.py:MCTPDataset_mask` reads the f13 root through gan_bpm's own `mctp` pipeline. gan_bpm's root is appended to `sys.path` at run time, so DataLoader workers must fork (Linux, not macOS). It returns the upstream dict plus `loss_mask` (gan_bpm's `support`). `cond_image` is deliberately **not** masked. `part` (train / monitor / test) comes from a gan_bpm fold's `split.json`.
- **Predictor:** `models/gan_bpm_predictor.py` puts gan_bpm's generator behind BRAN's `(y, hs)` interface. It refuses a checkpoint whose config is not Tmax, 12 s, 3 phases, linear. `Network(initial_predictor={"type": "gan_bpm", ...})` selects it, and `Network.init_weights` reloads it after BaseNetwork's re-initialization.
- **Network switches:** `mask_unet_cond` masks only the U-Net's copy of the condition, and `forward(..., loss_mask=...)` restricts the loss. With both absent, the behaviour is upstream's bit for bit.
- **Validation set:** a `which_val_dataset` block next to `which_dataset` replaces the random slice split.
- **Checks:** `scripts/ctp_only_checks.py` (synthetic data, CPU, including `run.py` train and test end to end) and `scripts/ctp_only_box_checks.py` (the real data, on the box). Rerun the first after changing any of these files.
- **Environment:** jobs run in `registry.cvib.ucla.edu/colivares:mrdpm-env-v1` (`Dockerfile`: gan_bpm's image plus MRDPM's missing packages). It has pandas 3, under which upstream's `LogTracker.reset` raised (fixed in `core/logger.py`).

## Repo hygiene

- `README.md` and `Readme.md` are both tracked. On macOS's case-insensitive filesystem only one of them exists on disk, so git always reports `README.md` as modified. Don't commit that phantom diff.
- There is no `.gitignore`, and `__pycache__/*.pyc` files are tracked. Avoid adding more of them.

## Related repos (data side)

The datasets this branch needs to connect to live outside this repo:

- `/Users/carlosolivares/github_repos/gan_bpm`: a replication of Zeng et al. 2025 (a GAN mapping 3 CTP phases to CBF/Tmax). `mctp/dataset.py:MCTPDataset` is its loader, and `docs/mctp_dataset_contract.md` specifies the on-disk layout and tensor shapes of the public mCTP data. `docs/project_map.md` maps the multi-repo pipeline. It has its own CLAUDE.md.
- `/Users/carlosolivares/github_repos/msu_data_prep` (not a git repo): data-curation projects, including `prepare_public_datsets_15July2026`, the builder that produces the public mCTP data.

Data built for gan_bpm will not drop straight into `NumpyDataset_mask`. Check the value range (this repo clamps to [0, 1]), the in-plane size (a multiple of 256 here), the channel layout (H×W×C `.npy` here) and the orientation (see the RAP flip) before wiring it in.
