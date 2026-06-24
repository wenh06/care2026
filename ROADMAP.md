# CARE 2026 Left Atrium — Development Roadmap

> **Tasks**: (1) LA scar quantification from LGE-MRI; (2) LA cavity segmentation from LGE-MRI with cross-center domain shift; (3) Multi-structure LA segmentation from CT with semi-supervised learning.

---

## Approach Overview

Three tasks are evaluated independently; no cross-modal fusion is required.

### Tasks 1 & 2 — LGE-MRI (Two-stage coarse-to-fine V-Net)

We train **two separate 3D networks in series**.

```
Raw LGE-MRI volume (H, W, D)
        │
        ▼  resample to canonical 576×576×44 (0.625 mm in-plane)
        │
        ├─────────────────────── STAGE 1 ───────────────────────
        │  downsample to 144×144×44 → z-score norm
        │  → single-head VNet → binary LA mask (coarse)
        │  → upsample mask to 576×576×44 → LA centroid (cx, cy)
        │
        ├─────────────── MIDDLE (centroid crop) ────────────────
        │  crop 256×256×44 around (cx, cy) in canonical space
        │
        └─────────────────────── STAGE 2 ───────────────────────
           z-score norm on crop → resize to 128×128×44
           → single-head VNet → LA scar (binary)
           → upsample back to 256×256×44 → place in canonical
           → resample to original space
           → post-process: constrain scar to dilated Stage-1 LA
             (2 mm dilation to cover the atrial wall)
```

Training jitter: ±32 px random centroid offset at Stage 2 training time (simulates Stage 1 errors).

Loss (Stage 2, scar-only with spatial weighting):

```
L_total = λ₁·L_dice(scar) + λ₂·L_focal(scar) + λ₃·L_ce_weighted(scar)
```
where ``L_ce_weighted`` multiplies voxel-wise CE by a Gaussian spatial
weight map ``w(x) = 1 + w₀·exp(−d²/2σ²)`` (d = distance to nearest GT
scar voxel, w₀ = 5, σ = 2 mm).

Task 2 uses the Stage-1 LA mask directly.  Stage-2 is reserved for scar
segmentation and does not refine the LA cavity mask.

### Task 3 — CT (Semi-supervised V-Net)

100 of 150 training CTs have no labels; the model must leverage unlabelled data.

Two semi-supervised modes, switched via ``CT_TrainCfg.semi_supervised_mode``:

#### Mode 1: CPS (current default)

Cross Pseudo Supervision [Chen et al., 2021] with two parallel 3D V-Nets:

```
Labelled batch
       ├─ Model 1 forward → supervised loss (Dice + CE)
       └─ Model 2 forward → supervised loss (Dice + CE)

Unlabelled batch
       ├─ Model 1 forward → pseudo-labels → supervise Model 2
       └─ Model 2 forward → pseudo-labels → supervise Model 1
```

L_total = L_sup(M1) + L_sup(M2) + λ_cps · [L_cps(M1←M2) + L_cps(M2←M1)]

λ_cps ramps from 0 → 1 over 30 epochs.

**Drawback**: both models can converge to similar errors (confirmation bias).

#### Mode 2: Mean Teacher (backup)

Mean Teacher [Tarvainen & Valpola, NeurIPS 2017] — single student VNet
+ EMA teacher.  The teacher's predictions serve as a consistency target
for the student on unlabelled data.

```
Labelled batch  →  student → supervised loss (Dice + CE)
Unlabelled batch →  student → softmax(x_s)
                    teacher  → softmax(x_t)  (no grad)
                    L_consist = MSE(softmax(x_s), softmax(x_t))
```

Teacher updated via EMA: θ_t ← α·θ_t + (1−α)·θ_s  (α = 0.99/step).

**Why this works for LA**: the LA cavity is a large, well-defined
structure — teacher predictions are stable; student consistency acts as
a smoothness regulariser.  Validated on the LA benchmark by Wu et al.
(MICCAI 2021, Mutual Consistency Training), MisMatch (IEEE TMI 2023),
and Geometry-Aware Consistency Training (arXiv 2024).

#### Other approaches surveyed (for reference)

| Method | Venue | Key idea | LA dataset |
|--------|-------|----------|-----------|
| Mutual Consistency | MICCAI 2021 | Dual-view consistency | ✓ |
| MisMatch | IEEE TMI 2023 | Morphological perturbation consistency | ✓ |
| OMF | MICCAI 2024 | Teacher-student overlay augmentation | ✓ |
| Geometry-Aware | arXiv 2024 | Geometric consistency + boundary weighting | ✓ |
| FixMatch | NeurIPS 2020 | Weak→strong augmentation pseudo-labeling | ✗ |

Additional tricks:
- CT windowing to soft-tissue window (clip to −200…+800 HU, normalise to [0,1]).
- Resample all volumes to uniform 0.5 × 0.5 × 0.5 mm isotropic.
- Per-class CE weighting to handle extreme class imbalance (LA : PV : LAA ≈
  13 : 1 : 3).  Weights: [bg=0.2, LA=1.0, PV=6.0, LAA=2.0].
- Configurable augmentations (probabilities set in ``TrainCfg.augmentation``):
  flips, 90° rotations, gamma correction, brightness/contrast jitter,
  Gaussian noise, Gaussian blur, elastic deformation, low-resolution
  simulation.  Each method is independently toggleable via its ``prob``.

---

## Phase 1 — Data Exploration & Repository Setup ✅

- [x] Explore raw training data at `/Data1/wenh06/CARE2026-LeftAtrium/`.
- [x] Document data layout: Center A MRI (Tasks 1 & 2), Center D CT (Task 3).
- [x] Discover quirks: LA cavity mask values are `{0, ~420}` (not `{0, 1}`); CT labels only for train_1..50.
- [x] Create full repository structure: `models/`, `models/loss/`, `utils/`, `checkpoints/`, `log/`, `results/`.
- [x] Create all Python stub files: `cfg.py`, `const.py`, `data_reader.py`, `dataset.py`, `outputs.py`, `pipeline.py`, `predict.py`, `trainer.py`, `post_docker_build.py`.
- [x] Write `README.md`, `.gitignore`, `.pre-commit-config.yaml`, `Dockerfile`, `requirements*.txt`.
- [x] Add `.github/workflows/`: `check-formatting.yml`, `docker-test.yml`.
- [x] Implement `data_reader.py`:
  - `CARE2026_MRI`: Tasks 1 & 2; normalises LA mask `{0,~420}→{0,1}`; `load_la_ann`, `load_scar_ann`, `load_ann_box`, crop, resample.
  - `CARE2026_CT`: Task 3; `labeled_records` / `unlabeled_records`; multi-class `{0,1,2,3}`; crop, resample.

---

## Phase 2 — Constants, Config & Dataset ✅

### 2.1 `const.py`

Define project-wide shape, spacing, and class constants.  All shape constants are derived from training-set statistics; no hardcoding anywhere else in the pipeline.

**MRI shape constants** (all in voxels, `H × W × D`):

| Constant | Value | Derivation |
|----------|-------|------------|
| `MRI_CANONICAL_SHAPE` | `(576, 576, 44)` | Modal raw image shape across all 190 training MRIs |
| `MRI_STAGE1_SHAPE` | `(144, 144, 44)` | 4× downsampled in H, W for coarse LA localisation |
| `MRI_STAGE2_CROP_SHAPE` | `(256, 256, 44)` | Centroid crop fed to Stage-2 model; covers p95 LA extent (≈230 px) with ~13 px margin |
| `MRI_STAGE2_CACHE_SHAPE` | `(320, 320, 44)` | Cache with ±32 px jitter margin: (320−256)/2 = 32 |
| `MRI_STAGE2_CENTROID_JITTER` | `(32, 32, 0)` | Max per-axis centroid offset during Stage-2 training |
| `MRI_PATCH_SHAPE` | alias for `MRI_STAGE2_CROP_SHAPE` | Backward compatibility |

**Other constants**:

| Constant | Value | Notes |
|----------|-------|-------|
| `MRI_CENTER_A_SPACING` | `(0.625, 0.625, 2.5)` mm | Center A Siemens scanner (in-plane × z) |
| `CT_TARGET_SPACING` | `(0.5, 0.5, 0.5)` mm | Isotropic resampling target |
| `CT_PATCH_SIZE` | `128` voxels | Sliding-window patch per side |
| `CT_HU_MIN / CT_HU_MAX` | `−200 / +800` HU | Soft-tissue window |
| `TASK1_TRAIN_COUNT` | `60` | Task 1 MRI samples (scar + cavity) |
| `TASK2_TRAIN_COUNT` | `130` | Task 2 MRI samples (cavity only) |
| `CT_LABELED_COUNT` | `50` | CT samples with GT labels |
| `CT_UNLABELED_COUNT` | `100` | CT samples without labels |

### 2.2 `cfg.py`

Implement `BaseCfg`, `ModelCfg`, and per-stage/per-task training configs using `torch_ecg.cfg.CFG`.

Key settings:

| Config | Value |
|--------|-------|
| `MRI_Stage1_TrainCfg.n_epochs` | 100 |
| `MRI_Stage2_TrainCfg.n_epochs` | 150 |
| `CT_TrainCfg.n_epochs` | 200 |
| `MRI_Stage1_TrainCfg.batch_size` | 4 |
| `MRI_Stage2_TrainCfg.batch_size` | 4 |
| `CT_TrainCfg.batch_size` | 2 |
| `MRI_*.optimizer` | `"adamw"` |
| `CT_TrainCfg.optimizer` | `"sgd"` |
| `MRI_*.lr` | `3e-4` |
| `CT_TrainCfg.lr` | `1e-2` |
| `MRI_*.lr_scheduler` | `"cosine"` |
| `CT_TrainCfg.lr_scheduler` | `"poly"` |
| `ModelCfg.vnet_stage1` | VNet config, `num_classes=2` |
| `ModelCfg.vnet_stage2` | VNet config, `num_classes=2` (background + scar) |
| `ModelCfg.vnet_ct` | VNet config, `num_classes=4` |

### 2.3 `dataset.py`

Implement PyTorch Dataset classes:

- `CARE2026_MRI_Stage1_Dataset`: resample raw volume → canonical → downsample → z-score; caches `(image, la_mask)` at `MRI_STAGE1_SHAPE`; augment on-the-fly (flip, elastic, gamma).
- `CARE2026_MRI_Stage2_Dataset`: resample raw → canonical; cache `MRI_STAGE2_CACHE_SHAPE` patch centred on GT LA centroid; at `__getitem__` apply ±`MRI_STAGE2_CENTROID_JITTER` offset and sub-crop to `MRI_STAGE2_CROP_SHAPE`; returns `(image, scar_mask, has_scar)` plus LA context for crop placement.
- `CARE2026_CT_Dataset`: CT windowing; resample to isotropic 0.5 mm; random patch; returns `(image, mask, is_labeled)`.
- `collate_fn_mri_stage1`, `collate_fn_mri`, `collate_fn_ct`.

---

## Phase 3 — Model Implementation ✅

### 3.1 `models/layers.py` — shared 3D building blocks

`ConvNormAct`, `ResBlock3D`, `DownBlock3D`, `UpBlock3D`, `NestedUpBlock3D`.

### 3.2 `models/vnet.py` — V-Net for MRI & CT

- `_SegEncoder3D`: shared encoder (stem + DownBlock3D stack).
- `VNet`: shared encoder → single decoder. Used for MRI Stage 1, MRI Stage 2 scar segmentation, and CT segmentation.

### 3.3 `models/nested_vnet.py` — Nested V-Net (UNet++)

`NestedVNet`: UNet++-style single-output variant.
- Dense skip connections across encoder/decoder levels.
- **Deep supervision**: returns one logit tensor per decoder level (coarse → fine); loss averaged across levels when enabled.

### 3.4 `models/loss/`

| File | Contents |
|------|----------|
| `dice_loss.py` | `SoftDiceLoss`, `DiceCELoss`, `TverskyLoss`, `FocalTverskyLoss` |
| `boundary_loss.py` | `BoundaryLoss` (on-the-fly scipy distance transform) |
| `__init__.py` | `Stage1MRILoss`, `ScarLoss`, `CTLoss` |

### 3.5 `models/__init__.py` — model wrappers

`CARE2026_MRI_Stage1_Model`: single-head VNet for LA localisation and Task 2 LA output.

`CARE2026_MRI_Stage2_Model`: single-head VNet for scar segmentation only.  It does not predict LA; scar post-processing is constrained by the Stage-1 LA mask.

`CARE2026_CT_Model`: wraps CPS or Mean Teacher VNet(s); loss includes supervised DiceCE plus consistency loss.

---

## Insights & Lessons Learned from MBAS2024

> Summarised from the MBAS2024 post-challenge analysis. These findings directly inform our design and experimental choices.

### What works

| Finding | Implication for CARE2026 |
|---------|--------------------------|
| **Architecture > hyperparameter tuning** — baseline DSC was statistically indistinguishable across augmentation strategies, loss functions, and ensemble variants | Focus engineering effort on architecture; avoid endless hyperparameter search |
| **CNN outperforms Transformer** on limited-size 3D MRI datasets; Transformers fail without task-specific adaptation | Do not use UNETR / SwinUNETR / nnFormer; stick to residual UNet / VNet |
| **ResUNet + 5-fold CV + ensemble** — top-3 teams all ensembled 5 sub-models | 5-fold CV + ensemble is mandatory for final submission; not for development runs |
| **SGD lr=0.01, 1000 epochs** — most common winning configuration | Our planned 150/200 epochs is too short; target ≥ 400 epochs; revisit optimiser |
| **Instance Norm** is critical for cross-center generalisation | Already in our MRI models; do not switch to Batch Norm for MRI |
| **Lightweight CNN (VNet, UMamba) matches large models** — no correlation between parameter count and DSC | Our ~7M-param VNet is well-chosen; no need to scale up |
| **Lightweight, task-specific models are preferable on small MRI sets** | Keep Stage 1 LA localisation and Stage 2 scar segmentation separate |

### Hardest failure modes (directly relevant to our tasks)

| Failure mode | Where it hurts us | Mitigation |
|---|---|---|
| **Domain shift** — wall/scar DSC drops ~10 pp at unseen center | Task 2 (cross-center LA segmentation) | InstanceNorm; TTA; test-time BN adaptation |
| **Post-ablation scar signal ≈ atrial wall** — models confuse the two | Task 1 (scar quantification) | Scar-only Stage 2; Tversky/Focal/spatially weighted losses for imbalance |
| **Superior/inferior slice degradation** — U-shaped DSC profile along z-axis; complex vascular junctions | All tasks | Deep supervision (NestedVNet); slice-position feature |
| **Segmentation leakage** — predictions bleed into adjacent structures | All tasks | Connected-component post-processing (keep largest component) |
| **Low-SNR images** — wall/scar DSC strongly correlated with SNR | Task 1 scar boundary accuracy | CLAHE preprocessing (`utils/mclahe.py` already exists) |

### UMamba as alternative backbone

UMambaBot (Mamba-based state-space model) achieved near-ResUNet performance with better computational efficiency. Worth evaluating as a drop-in backbone replacement for MRI tasks if VNet under-performs.

---

## Phase 4 — Training Loops ✅

### 4.1 MRI trainers (`trainer.py`)

Two separate trainer classes:

`CARE2026_MRI_Stage1_Trainer`:
- Binary LA segmentation loss (`Stage1MRILoss`: DiceCE on LA only).
- Metric: `la_dice` (local validation split).
- Checkpoint prefix `*-mri1`; monitors `la_dice`.

`CARE2026_MRI_Stage2_Trainer`:
- Scar-only loss (`ScarLoss`) on Task-1 samples, with no-scar Task-2 samples used as hard negatives.
- Metrics: `scar_dice`, `scar_acc`, `scar_sen`.
- Checkpoint prefix `*-scar`; monitors `scar_dice`.
- Supports `backbone="vnet"` and `backbone="nested_vnet"`.
- AMP: `use_amp=True`.

CLI:
```bash
python trainer.py --task mri --stage 1 --db-dir <data_root> --epochs 100
python trainer.py --task mri --stage 2 --db-dir <data_root> --epochs 150
```

### 4.2 CT trainer (`trainer.py`)

`CARE2026_CT_Trainer`:
- CPS loss: ramped consistency weight `λ_cps(t)`.
- Metric: `ct_dice_la`, `ct_dice_pv`, `ct_dice_laa`, `ct_mean_dice`.
- Labeled-only validation split.

```bash
python trainer.py --task ct --db-dir <data_root> --epochs 200
```

---

## Phase 5 — Prediction & Post-processing ✅

`predict.py` (core volume-level inference library):
- **`predict_mri_two_stage(img_path, stage1_model, stage2_model, ...)`** — true two-stage inference:
  1. Load NIfTI → resample to `MRI_CANONICAL_SHAPE` (576×576×44).
  2. Downsample canonical → `MRI_STAGE1_SHAPE` (144×144×44) → z-score → Stage-1 VNet → binary LA prob map → upsample to canonical.  This is the Task-2 LA output and also provides the Stage-2 crop centroid.
  3. Crop `MRI_STAGE2_CROP_SHAPE` (256×256×44) centred on centroid (zero-pad if near boundary).
  4. z-score → Stage-2 scar-only VNet → scar prob map.
  5. Strip padding → place scar back in canonical → resample LA/scar masks to original shape + affine.
- **CT inference** (`predict_ct`): sliding window `128³` patches, stride `64³`, Gaussian overlap weighting, soft-vote argmax.
- **TTA**: 8-fold flip averaging (all `{H, W, D}` axis combinations) for both stages / CT.
- **Post-processing**: `keep_largest_component`, `postprocess_mri_masks` (scar constrained within LA cavity), `postprocess_ct_mask` (per-class largest component).
- **CLAHE**: auto-detected from `stage2_model.train_config.apply_mclahe`; applied to the canonical image before Stage 1 downsample and Stage 2 crop.

`outputs.py`:
- `CARE2026Outputs` dataclass: `la_mask`, `scar_mask`, `ct_mask` + `source_affine` / `source_header`.
- `save_as_nifti(output_dir, record_id, task_num)`: writes `<task_dirname>/<record_id>/<record_id>_pred.nii.gz` with original affine.
- `package_submission(results_dir, team_name)`: creates challenge-compliant `CARE-Leftatrium-<team>.zip`.

`pipeline.py` (high-level orchestration + **unified CLI**):
- `run_task1_inference(stage1_model, stage2_model, ...)` / `run_task2_inference(...)`: per-task validation runners using `predict_mri_two_stage`.
- `run_task3_inference(ct_model, ...)`: CT validation runner.
- `run_all_tasks(mri_stage1_model, mri_stage2_model, ct_model, ...)`: convenience wrapper; `None` models safely skipped.
- Auto-discovers checkpoints (tries canonical names `mri_stage1_model.safetensors` etc., falls back to `BestModel_*` / `CARE2026_*` glob).
- CLI: `python pipeline.py --input_dir ... --output_dir ... --model_dir ... --tasks 1,2 [--run_name ...] [--team_name ...] [--package ...]`.
- Output is written to `<output_dir>/<run_name>/` (default run name is timestamped `run_YYYYMMDD_HHMMSS`).

**Validation data confirmed** (already on disk at `/Data1/wenh06/CARE2026-LeftAtrium/`):
- Task 1: 10 records (`val_1..val_10`), `enhanced.nii.gz`.
- Task 2: 20 records (`val_1..val_20`), `enhanced.nii.gz`.
- Task 3: 20 records (`val_1..val_20`), `NNNN.nii.gz` (4-digit zero-padded).

---

## Phase 6 — Training Runs 🏃

### 6.1 MRI Stage 1 — Coarse LA Localiser — ✅ Done

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  python trainer.py --task mri --stage 1 \
  --db-dir /Data1/wenh06/CARE2026-LeftAtrium --epochs 100 \
  2>&1 | tee log/mri1_train.log
```

Input: 144×144×44, batch_size=4.  Trained to epoch 100; checkpoint at `checkpoints/mri_stage1_model.safetensors`.

**CLAHE variant** also trained: `log/mri1_mclahe_train.log`, same architecture but with MCLAHE preprocessing enabled.

### 6.2 MRI Stage 2 — Scar-Only Segmenter — ⏳ Needs retraining

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  python trainer.py --task mri --stage 2 \
  --db-dir /Data1/wenh06/CARE2026-LeftAtrium --epochs 200 --mclahe true \
  2>&1 | tee log/scar_train.log
```

Input: 128×128×44 (resized from 256×256×44 crop), batch_size=4, AMP.
ScarLoss with spatial weight map (w₀=5, σ=2 mm).  LA cavity from Stage 1.
``no_scar_proportion=0.3`` keeps ~35/130 Task-2 (no-scar) records as hard
negatives; the ScarLoss penalises false scar predictions on these samples.

### 6.3 CT Semi-Supervised (Task 3) — 🔄 Iterating

All runs: SGD+poly LR (power=0.9), batch_size=2, 128³ patches,
HU clip [−200,800] → minmax [0,1], 0.5mm isotropic resampling.
Old augmentations (＂old aug＂): flips + 90° rotation + intensity
scaling [0.9,1.1] + Gaussian noise σ≤0.05, each at 50 % prob.

| # | Mode | Epochs | Aug | Class weights | Best | @Ep | Notes |
|---|------|--------|-----|---------------|------|-----|-------|
| 1 | CPS | 200 | old aug | — | 0.3904 | — | baseline |
| 2 | CPS | 200 | old aug | [0.2,1,6,2] | 0.4275 | — | PV ×6 |
| 3 | MT | 200 | old aug | [0.2,1,6,2] | 0.4655 | — | EMA 0.99 |
| 4 | MT | 300 | old aug | [0.2,1,6,2] | **0.5234** | 265 | +100 ep; also 0.4601 @197 ep |
| 5 | MT | 300 | new 8-way | [0.2,1,6,2] | 0.5013 | 296 | aug hurt (−2.1 pp vs run 4) |

Mean Teacher outperforms CPS by ~7.5 pp, likely because:
- Teacher EMA provides a more stable pseudo-label target than the
  cross-pseudo-supervision cycle (which can suffer confirmation bias).
- MSE on softmax preserves prediction uncertainty; CPS hard argmax
  discards it.

#### 6.3.1 V1 Diagnostic (2026-06-24)

Inference on the 50 labelled training records with the best V1 checkpoint
(MT, epoch 299) reveals severe underfitting — the model fails to learn
even the training set:

| Class | Train Dice | Sensitivity | Precision | Pred/GT |
|-------|-----------|-------------|-----------|---------|
| LA | 0.697 | 0.766 | 0.677 | 1.25× |
| PV | 0.329 | 0.527 | 0.255 | 2.23× |
| LAA | 0.378 | 0.369 | 0.471 | 0.90× |
| **Mean** | **0.468** | — | — | — |

Root causes (ranked by impact):

1. **BatchNorm + batch_size=2 is broken for 3D.**  BN computes
   statistics over 2 × spatial_samples; deep layers have only
   ~512 values/channel.  The MRI models use InstanceNorm and
   work well — CT should match.
2. **SGD lr=0.01 amplifies small-batch noise.**  MRI models use
   AdamW lr=3e-4.  With batch_size=2, SGD gradient estimates are
   too noisy for stable convergence.
3. **PV class weight 6× causes massive over-prediction.**  The
   model predicts 2.23× more PV voxels than GT (precision 0.255).
4. **Semi-supervised signal is negligible.**  Consistency loss
   ≈ 0.0003 vs supervised loss ≈ 0.4 — a 1000× gap.  The
   teacher (EMA of a bad student) produces garbage pseudo-labels
   that reinforce errors (confirmation bias).
5. **8-way augmentation degrades performance.**  Run 5 (0.5013)
   underperforms run 4 (0.5234); elastic deformation likely
   distorts thin PV structures.

**Plan → V2 (Section 6.3.2).**

#### 6.3.2 CT V2 — Supervised-First Redesign

Three architectural fixes applied in ``CT_TrainCfgV2`` +
``ModelCfg.vnet_ct_v2`` + ``CARE2026_CT_ModelV2``:

| Component | V1 | V2 | Rationale |
|-----------|----|----|-----------|
| Normalisation | BatchNorm | **InstanceNorm** | batch-size independent |
| Activation | ReLU | **Mish** | smoother gradients |
| Optimizer | SGD lr=1e-2 | **AdamW lr=3e-4** | adaptive LR, matches MRI |
| LR schedule | poly | **cosine** | matches MRI |
| Semi-supervised | CPS/MT | **supervised-only** | 50 labelled CTs sufficient; semi-supervised was adding noise |
| PV class weight | 6.0 | **3.0** | reduce over-prediction |
| Augmentation | 8 methods | **6** (−elastic, −low-res) | avoid distorting thin PV |
| Params | 13.7M (×2 models) | **6.9M** (×1 model) | lighter, faster |

Training command:
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  python trainer.py --task ct \
  --db-dir /Data1/wenh06/CARE2026-LeftAtrium --epochs 300 \
  2>&1 | tee log/ct_v2_train.log
```

Training command:
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  python trainer.py --task ct \
  --db-dir /Data1/wenh06/CARE2026-LeftAtrium --epochs 200 \
  --semi-mode mean_teacher \
  2>&1 | tee log/ct_mt_train.log
```

**MT extended (300 epochs, 2026-06-09):**

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  python trainer.py --task ct \
  --db-dir /Data1/wenh06/CARE2026-LeftAtrium --epochs 300 \
  --semi-mode mean_teacher \
  2>&1 | tee log/ct_mt_300ep_train.log
```

| Metric | Value | Epoch |
|--------|-------|-------|
| Best (overall) | **0.5234** | 265 |
| Best within 200 ep | 0.4601 | 197 |

Note: this run used the old hardcoded augmentations (flips, rotation,
intensity scaling, Gaussian noise) via ``aug_prob=0.5``.  The +5.8 pp
gain over the 200-epoch MT baseline (0.4655 → 0.5234) is attributable
to extended training (+100 epochs).  Planned: re-run with the new
8-method configurable augmentations enabled.

### 6.4 Validation & Submission

MRI models ready (Stage 1 + Stage 2 both with CLAHE variants). CT model iterating
(Mean Teacher best at 0.5234 with 300 epochs, no augmentation; planned re-run
with 8-way augmentation enabled).

```bash
# MRI only (Tasks 1 & 2):
python pipeline.py \
  --input_dir /Data1/wenh06/CARE2026-LeftAtrium \
  --output_dir /Data1/wenh06/CARE2026-LeftAtrium/output \
  --model_dir checkpoints/ \
  --tasks 1,2

# Full submission (all three tasks, once CT is final):
python pipeline.py \
  --input_dir /Data1/wenh06/CARE2026-LeftAtrium \
  --output_dir /Data1/wenh06/CARE2026-LeftAtrium/output \
  --model_dir checkpoints/ \
  --tasks 1,2,3

# Upload CARE-Leftatrium-REVENGER.zip to:
# http://zmic.org.cn/care_2026/eval/login?track=leftatrium
```

Local metrics to track:
- Task 1: G-DSC / ACC / SEN vs. epoch.
- Task 2: DSC and domain-shift gap (Center A train-val vs. Center C val).
- Task 3: per-class DSC (LA / PV / LAA); CPS pseudo-label quality curve.

---

## Phase 7 — Experiments & Ablations ⏳

### Core ablations

| Experiment | Goal |
|------------|------|
| Scar-only VNet vs. NestedVNet | Deep supervision benefit for sparse scar segmentation |
| Scar threshold sweep | Optimise Task 1 G-DSC / SEN trade-off |
| Stage-1 threshold sweep | Optimise Task 2 DSC and Task 1 scar constraint region |
| CPS vs. Mean Teacher (CT) | Measure semi-supervised benefit of EMA consistency vs CPS | ✅ Done — MT wins (0.4655 vs 0.3904) |
| CT augmentation sweep | Gamma + brightness/contrast + Gaussian blur; measure per-class Dice impact | 🔄 Next |
| Boundary Loss weight sweep | Optimise PV / LAA delineation (CT Task 3) |
| Patch size: 128³ vs. 64³ (CT) | Memory vs. accuracy trade-off |

### Experiments from MBAS2024 insights

| Experiment | Goal | Priority | Status |
|------------|------|----------|--------|
| **Extend MRI training: 150 → 400 epochs** | Close epoch-count gap vs. winning teams (1000 epochs) | 🔴 High | ⏳ |
| **CLAHE preprocessing** — enable `utils/mclahe.py` in MRI dataset | Improve low-SNR scar boundary accuracy | 🔴 High | ✅ Done (auto-detected in predict; `--mclahe` flag in trainer) |
| **Connected-component post-processing** — keep largest component per class | Eliminate segmentation leakage | 🔴 High | ✅ Done (`keep_largest_component`, `postprocess_mri_masks`, `postprocess_ct_mask` in `predict.py`) |
| **Test-time augmentation (TTA)** — flip + 90° rotations, average logits | Low-cost accuracy boost; especially useful for Task 2 domain generalisation | 🔴 High | ✅ Done (8-fold flip TTA in `predict.py`; toggle via `--tta` flag) |
| **5-fold CV + ensemble** — train 5 folds, ensemble predictions | Expected +1–2 pp DSC; mandatory for final submission | 🟡 Medium (Phase 8) | ⏳ |
| **SGD + polynomial LR for MRI** (vs. AdamW) | Winning teams used SGD lr=0.01; compare convergence | 🟡 Medium | ⏳ (CT uses SGD+poly; MRI still AdamW+cosine) |
| **Slice-position encoding** — append z-coordinate channel to input | Help model handle hard superior/inferior slices | 🟡 Medium | ⏳ |
| **Histogram matching / intensity standardisation** | Domain-shift mitigation for Task 2 unseen centers without changing the model | 🟡 Medium | ⏳ |
| **UMamba backbone** — replace VNet encoder with Mamba SSM | Near-ResUNet accuracy, more efficient; possible scar segmentation alternative | 🟢 Low | ⏳ |
| **Shape-constrained regularisation** — atlas-based prior or topology loss | Anatomy-aware design; reduce leakage at vascular junctions | 🟢 Low | ⏳ |

---

## Phase 8 — Challenge Submission Pipeline 🔄

- [x] Implement `post_docker_build.py`: cache trained weights from cloud storage (stub ready; needs cloud URLs).
- [x] Set up Docker CI (`docker-test.yml` with `status: pre`).
- [ ] Run full training (MRI: ≥ 400 epochs, CT: ≥ 400 epochs); save best checkpoints.
  - Development runs: MRI Stage 1 (100 epochs) ✅, MRI Stage 2 (150 epochs) ✅; CT (200 epochs) ⏳.
  - Final submission runs: 400+ epochs with 5-fold CV.
- [ ] End-to-end inference smoke test inside Docker container.
- [ ] Submit MRI validation predictions (Tasks 1 & 2) to official evaluation platform.
- [ ] Set `status: alpha` in `docker-test.yml`; enable dataset download step.
- [ ] Train CT model and submit Task 3 predictions.
- [ ] Set `status: final` for final submission.

---

## Immediate Next Steps

1. ~~**Train MRI Stage 1**~~ ✅ Done — checkpoint: `checkpoints/mri_stage1_model.safetensors` (also CLAHE variant trained).
2. ~~**Train MRI Stage 2**~~ ✅ Done — checkpoint: `checkpoints/mri_stage2_model.safetensors` (also CLAHE variant trained; epoch snapshots 147–149 retained).
3. ~~**MRI post-processing**~~ ✅ Done — `postprocess_mri_masks()` (scar constrained within dilated LA cavity; largest connected component).
4. ~~**CT post-processing**~~ ✅ Done — `postprocess_ct_mask()` (per-class largest connected component).
5. ~~**CLAHE ablation**~~ ✅ Done — MCLAHE wired into dataset (config flag `apply_mclahe`); auto-detected at inference time.
6. ~~**CT baseline training**~~ ✅ Done — CPS baseline + class weights + Mean Teacher all trained; MT best (0.5234 mean Dice).  V1 diagnostic (2026-06-24) identified BatchNorm + SGD + PV weight as root causes of poor performance.
7. **Train CT V2 (supervised-first)** — ``CT_TrainCfgV2`` + ``CARE2026_CT_ModelV2`` with InstanceNorm + Mish + AdamW:
   ```bash
   PYTORCH_ALLOC_CONF=expandable_segments:True \
     python trainer.py --task ct \
     --db-dir /Data1/wenh06/CARE2026-LeftAtrium --epochs 300 \
     2>&1 | tee log/ct_v2_train.log
   ```
8. **Run MRI validation inference** (Tasks 1 & 2):
   ```bash
   python pipeline.py \
     --input_dir /Data1/wenh06/CARE2026-LeftAtrium \
     --output_dir /Data1/wenh06/CARE2026-LeftAtrium/output \
     --model_dir checkpoints/ \
     --tasks 1,2
   ```
9. **5-fold CV + ensemble** (Phase 8): train 5 folds, ensemble predictions for final submission.
10. **Extend MRI training epochs** (400+): evaluate whether extended training closes the gap toward winning-team performance.
11. **SGD + poly LR for MRI**: test whether SGD (as used by top MBAS2024 teams) outperforms AdamW+cosine for MRI tasks.

---

## Notes & Open Questions

- **Task 2 domain shift**: Center B/C validation data (20 samples) is available for fine-tuning / histogram matching — check challenge rules whether this is permitted.
- **Task 3 label format**: challenge page says `cardiacSegImgMO.nii.gz` but training data uses `label_XXXX.nii.gz` — must handle both names in `data_reader.py` when validation/test data is released.
- **G-DSC for Task 1**: the scar label is used as a _continuous_ weight map in G-DSC computation; model's soft output (pre-sigmoid) may score better than hard threshold.
- **Prizes**: only Tasks 1 and 3 are prize-eligible; prioritise these two.
