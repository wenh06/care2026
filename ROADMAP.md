# CARE 2026 Left Atrium — Development Roadmap

> **Tasks**: (1) LA scar quantification from LGE-MRI; (2) LA cavity segmentation from LGE-MRI with cross-center domain shift; (3) Multi-structure LA segmentation from CT with semi-supervised learning.

---

## Approach Overview

Three tasks are evaluated independently; no cross-modal fusion is required.

---

## Dataset ID Quick Reference

| ID | Task | Content | CLAHE | Labels | Input | Creation Command |
|----|------|--------|:---:|------|------|------|
| 500 | CT 3 | multi-structure | — | 4-class | full volume | `python scripts/prep_nnunet_ct.py --db-dir ...` |
| 501 | MRI 1 | scar | — | binary scar | crop 256×256×44 | `python scripts/prep_nnunet_mri.py --db-dir ... --task 1` |
| 502 | MRI 2 | cavity | — | binary cavity | full 576×576×44 | `python scripts/prep_nnunet_mri.py --db-dir ... --task 2` |
| 503 | CT 3 | self-trained | — | 4-class | full volume | `python scripts/self_train_nnunet.py ... --dataset-id 503` |
| 511 | MRI 1 | scar | ✅ | binary scar | crop | `python scripts/prep_nnunet_mri.py --db-dir ... --task 1 --mclahe` |
| 512 | MRI 2 | cavity | ✅ | binary cavity | full | `python scripts/prep_nnunet_mri.py --db-dir ... --task 2 --mclahe` |
| 521 | MRI 1 | scar+cavity | — | 2-class | crop | `python scripts/prep_nnunet_mri.py --db-dir ... --task 1 --dataset-id-task1 521 --multi-class` |
| 531 | MRI 1 | scar+cavity | ✅ | 2-class | crop | `python scripts/prep_nnunet_mri.py --db-dir ... --task 1 --dataset-id-task1 531 --multi-class --mclahe` |
| 600 | MRI 1 | scar+cavity | — | 2-class | full volume | `python scripts/prep_nnunet_mri.py --db-dir ... --task 1 --dataset-id-task1 600 --multi-class --no-crop` |

## Best Model Configurations (for submission & inference)

### Task 1 — LA Scar Quantification

| Role | Dataset | Trainer | Notes |
|------|---------|----------|-------|
| **Stage 1** (cavity) | 502 | `nnUNetTrainer__nnUNetPlans__3d_fullres` | 190 cases, no CLAHE |
| **Stage 2** (scar) | 521 | `nnUNetTrainerScarGaussian` | 60 cases, multi-class (cav=1, scar=2), no CLAHE, GPU Gaussian spatial weight |
| Dilation | `None` | — | no LA cavity constraint |
| TTA | on | — | marginal gain (+0.0017 G-DSC) |

**Pipeline**: ``predict_mri_two_stage_hybrid`` (native Stage 1 + canonical Stage 2), 502→centroid crop→521 (ScarGaussian), ``scar_dilation=None``

### Task 2 — LA Cavity Segmentation

| Role | Dataset | Trainer | Notes |
|------|---------|----------|-------|
| **Model** | 502 | `nnUNetTrainer__nnUNetPlans__3d_fullres` | 190 cases, no CLAHE, DSC 0.8871 (native pipeline; 512 CLAHE not tested with native) |

**Pipeline**: ``predict_mri_two_stage`` (native), stage2_model=None (cavity-only)

### Task 3 — LA Multi-Structure Segmentation (CT)

| Role | Dataset | Trainer | Notes |
|------|---------|----------|-------|
| **Primary** | 500 | `nnUNetTrainer__nnUNetPlans__3d_fullres` | 50 labeled, no CLAHE, DSC 0.9563 (#2) |
| **Secondary** | 503 | `nnUNetTrainer__nnUNetPlans__3d_fullres` | self-trained on 150 cases, DSC 0.9745 (training-set; worse than 500 but complementary) |
| **Tested** | 500 | `nnUNetTrainerCTBoundary` | boundary loss (HausdorffER + CenterlineCE), training-set Mean DSC 0.9812 (−0.0011 vs baseline) |

**Pipeline**: ``predict_ct``.  Ensemble 500 + 503 if complementary errors provide gain.

---

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

- [x] Explore raw training data at `/path/to/CARE2026-LeftAtrium/`.
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
| `MRI_Stage1_TrainCfg.n_epochs` | 300 (SGD+polyLR) |
| `MRI_Stage2_TrainCfg.n_epochs` | 600 (SGD+polyLR) |
| `CT_TrainCfg.n_epochs` | 1000 |
| `MRI_Stage1_TrainCfg.batch_size` | 4 |
| `MRI_Stage2_TrainCfg.batch_size` | 4 |
| `CT_TrainCfg.batch_size` | 2 |
| `MRI_*.optimizer` | `"sgd"` (switched from adamw; MBAS2024 insight) |
| `CT_TrainCfg.optimizer` | `"sgd"` |
| `MRI_*.lr` | `1e-2` |
| `CT_TrainCfg.lr` | `1e-2` |
| `MRI_*.lr_scheduler` | `"poly"` (switched from cosine) |
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
- `_load_model()`: auto-detects VNet (``.safetensors`` file) vs nnUNet (directory with ``plans.json``).
- `PredictCfg` (``cfg.py``): centralised model paths — ``ct_model``, ``mri_stage1_model``, ``mri_stage2_model``, ``mri_apply_mclahe``.
- CLI: ``--ct-model``, ``--mri-stage1-model``, ``--mri-stage2-model`` per-model paths; ``--mri-mclahe`` for nnUNet.
- Auto-detects val/test phases (``val_`` / ``test_`` prefix) for record discovery and output naming.

**Validation data confirmed** (already on disk at `/path/to/CARE2026-LeftAtrium/`):
- Task 1: 10 records (`val_1..val_10`), `enhanced.nii.gz`.
- Task 2: 20 records (`val_1..val_20`), `enhanced.nii.gz`.
- Task 3: 20 records (`val_1..val_20`), `NNNN.nii.gz` (4-digit zero-padded).

---

## Phase 6 — Training Runs 🏃

### 6.1 MRI Stage 1 — Coarse LA Localiser — ✅ Done

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  python trainer.py --task mri --stage 1 \
  --db-dir /path/to/CARE2026-LeftAtrium --epochs 100 \
  2>&1 | tee log/mri1_train.log
```

Input: 144×144×44, batch_size=4.  Trained to epoch 100; checkpoint at `checkpoints/mri_stage1_model.safetensors`.

**CLAHE variant** also trained: `log/mri1_mclahe_train.log`, same architecture but with MCLAHE preprocessing enabled.

### 6.2 MRI Stage 2 — Scar-Only Segmenter — 🔄 Iterating

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  python trainer.py --task mri --stage 2 \
  --db-dir /path/to/CARE2026-LeftAtrium --epochs 200 --mclahe true \
  2>&1 | tee log/scar_train.log
```

Input: 128×128×44 (resized from 256×256×44 crop), batch_size=4, AMP.
ScarLoss with spatial weight map (w₀=5, σ=2 mm).  LA cavity from Stage 1.
``no_scar_proportion=0.3`` keeps ~35/130 Task-2 (no-scar) records as hard
negatives; the ScarLoss penalises false scar predictions on these samples.

**Backbone experiments (all with SGD+polyLR, CLAHE):**

| Backbone | Epochs | In Channels | Task 1 G-DSC | Notes |
|----------|--------|------------|-------------|-------|
| `vnet_stage2` | 300 | 1 (MRI only) | **0.2189** (val) / 0.3468 (train, none) | current best; dilation=none → +70% train |
| `vnet_stage2_2ch` | 600 | 2 (MRI+SDF) | 0.1882 | −3.1pp vs baseline; rolled back |
| `vnet_stage2_l` | — | 1 | — | underperformed baseline |
| `nested_vnet_stage2` | — | 1 | — | underperformed baseline; deep sup disabled |
| `nested_vnet_stage2_l` | — | 1 | — | underperformed baseline |

Conclusion: larger/wider backbones and 2ch SDF channel hurt Task 1 G-DSC.
The original ``vnet_stage2`` (single-channel, 16→32→64→128→256, 6.9M params)
with ScarLoss + SGD 300ep remains the best.  Switching to **nnUNet for MRI**
is the most promising next direction — the 6-stage ResEnc UNet with built-in
deep supervision may close the capacity gap without manual architecture tuning.

#### 6.2.1 nnUNet MRI — Breakthrough 🔄

nnUNet PlainConvUNet (6-stage, 30.6M) trained on Task 1 (scar, Dataset 501,
60 cropped 256×256×44 cases) and Task 2 (cavity, Dataset 502, 190 full-volume
576×576×44 cases).  Evaluated on labeled training data (in-domain):

**Task 1 — LA Scar (60 labeled cases, 5-fold ensemble, training-set evaluation):**

| S1 | S2 | Multi-class | MCLAHE | Dilation | Dice | Official G-DSC | ACC | SEN | Loss |
|----|----|:----------:|:------:|:--------:|------:|:--------------:|-----|-----|------|
| VNet S1 | VNet S2 | — | ✅ | 2mm | 0.2034 | — | 0.9997 | 0.1946 | ScarLoss |
| 502 | 501 | — | — | 2mm | 0.3529 | — | 0.9998 | 0.2573 | default[^nnunet-loss] |
| 512 | 511 | — | ✅ | 2mm | 0.3335 | — | 0.9998 | 0.2390 | default[^nnunet-loss] |
| 502 | 521 | ✅ | — | 2mm | 0.3533 | — | 0.9998 | 0.2577 | default[^nnunet-loss] |
| 502 | 501 | — | — | 5mm | 0.6211 | — | 0.9998 | 0.6242 | default[^nnunet-loss] |
| 502 | 521 | ✅ | — | 5mm | **0.6221** | — | 0.9998 | 0.6267 | default[^nnunet-loss] |
| 502 | 531 | ✅ | ✅ | 5mm | 0.6016 | — | 0.9998 | 0.5775 | default[^nnunet-loss] |
| 512 | 511 | — | ✅ | 5mm | 0.5875 | — | 0.9998 | — | default[^nnunet-loss] |
| 502 | 501 | — | — | none | 0.6317 | 0.6561 | 0.9998 | 0.6490 | default[^nnunet-loss] |
| 502 | 521 | ✅ | — | none | 0.6333 | 0.6563 | 0.9998 | 0.6517 | default[^nnunet-loss] |
| 512 | 511 | — | ✅ | none | 0.5976 | — | 0.9998 | 0.5788 | default[^nnunet-loss] |
| 502 | 521 | ✅ | — | none | **0.6631** | 0.6600 | 0.9998 | 0.6765 | ScarGaussian |
| 502 | 521 | ✅ | — | none | 0.6629 | 0.6630 | 0.9998 | **0.6837** | ScarCavityWall |
| 502 | 521 | ✅ | — | none | **0.6682** | 0.6682 | **0.9999** | 0.6779 | ScarGaussian + ResEnc M |
| 502 | 524 | ✅ | — | none | 0.6224 | 0.6225 | 0.9998 | 0.6387 | ScarGaussianSDM |

[^nnunet-loss]: nnUNet's default ``DC_and_CE_loss`` (Dice + Cross-Entropy).

**Backbone & Loss Ablation (2026-07-24) — training-set, no TTA, native pipeline:**

| Backbone | Loss | G-DSC | ACC | SEN | Δ vs baseline |
|----------|------|-------|-----|-----|:---:|
| PlainConvUNet | ScarGaussian | 0.6631 | 0.9998 | 0.6765 | — (baseline) |
| PlainConvUNet | ScarCavityWall | 0.6629 | 0.9998 | **0.6837** | −0.0002 |
| **ResEnc M** | ScarGaussian | **0.6682** | **0.9999** | 0.6779 | **+0.0051** |

**Pipeline comparison (native vs hybrid, no TTA):**

| Backbone | Loss | Native | Hybrid | Δ |
|----------|------|:---:|:---:|:---:|
| PlainConvUNet | ScarGaussian | 0.6631 | — | — |
| PlainConvUNet | ScarCavityWall | 0.6629 | 0.6386 | −0.0243 |
| ResEnc M | ScarGaussian | **0.6682** | 0.6422 | −0.0260 |

Key findings:
- **ResEnc M confirms residual encoder architecture gain** (+0.0051 G-DSC over PlainConvUNet).
- **CavityWall flat vs. ScarGaussian** — higher SEN (+0.0072) but G-DSC unchanged; recall gains offset by precision losses.
- **Hybrid pipeline harmful for both E1 and E2** (−0.024 to −0.026 vs native).  Double interpolation (native→canonical→nnUNet internal) degrades predictions when training spacing = test spacing.
- E3 (ResEnc M + CavityWall) not yet trained — orthogonal gains (architecture + higher SEN) could compound.

nnUNet (502+521, dilation=none) improves G-DSC by **+212 %** over VNet best on the same training data.

**Dilation sweep (same 60 cases):**

| Dilation | 502+501 (binary) | 502+521 (multi) | 512+511 (CLAHE bin) | 502+531 (CLAHE multi) |
|----------|------|------|------|------|
| none | **0.6317** | **0.6333** | 0.5976 | — |
| 2mm | 0.3529 | 0.3533 | 0.3138 | — |
| 5mm | 0.6211 | 0.6221 | 0.5875 | 0.6016 |
| 10mm | 0.6306 | 0.6321 | 0.5969 | — |

Key findings:
- **Remove LA constraint entirely** (dilation=none) — highest G-DSC across all models.
  LA cavity prediction errors clip true scar; even 10mm dilation loses ~0.1pp vs none.
- Dilation is the **dominant factor** (+0.28 G-DSC from 2mm→none).
- **CLAHE consistently harmful** (−3–4 pp at every dilation).  Multi-class partially
  compensates (+1.4 pp from 511→531) but cannot fully offset CLAHE damage (−2 pp vs 521).
- **Multi-class (521) ≈ binary (501)** — at best 0.15 pp difference, negligible.
- Default changed: ``postprocess_mri_masks(dilation_mm=None)`` — no LA constraint.
  Top-1 result is dilation, not data strategy.
  531 only tested at 5mm (``eval_all_models.py`` default at the time).

**6-stage hybrid dilation sweep (ScarGaussian, official G-DSC):**

| Dilation | G-DSC | ACC | SEN |
|----------|-------|------|------|
| none | 0.6372 | 0.9998 | 0.6553 |
| 3.0mm | 0.5704 | 0.9998 | 0.5237 |
| 5.0mm | 0.6268 | 0.9998 | 0.6317 |
| 7.0mm | 0.6336 | 0.9998 | 0.6481 |

Conclusion: `none` remains best; dilation still provides no benefit.

CLAHE is harmful for nnUNet (−5.5 % vs no CLAHE), consistent with the
observation that nnUNet's ZScoreNormalization already handles intensity
standardisation adequately.

**VNet leaderboard comparison**: VNet best on official validation leaderboard
was G-DSC 0.2189 — the training-set metric (0.2034) is a reasonable proxy.

**Task 2 — LA Cavity (190 labeled cases, 5-fold ensemble, full-volume inference):**

| Model | DSC |
|-------|-----|
| Dataset 502 (no CLAHE) | 0.9518 |
| Dataset 512 (CLAHE) | 0.9543 |

CLAHE gives marginal +0.25pp on cavity (not significant).  Training-set DSC (0.95)
vs. validation leaderboard (0.8832, nnUNet) suggests domain shift of ~6.8 pp
from Center B/C — consistent with the known challenge of multi-center LGE-MRI.

**Task 3 — CT (50 labeled cases, 5-fold ensemble, full-volume inference):**

| Model | LA | PV | LAA | Mean | Notes |
|-------|-----|-----|-----|------|-------|
| Dataset 500 (50 labeled) | **0.9938** | **0.9832** | **0.9700** | **0.9823** | nnUNet baseline |
| Dataset 500 CTBoundary | 0.9934 | 0.9820 | 0.9680 | 0.9812 | HausdorffER + CenterlineCE loss; −0.0011 vs baseline |
| Dataset 503 (self-trained, 150 cases) | 0.9916 | 0.9771 | 0.9547 | 0.9745 | hard pseudo-labels from 500 → hurts all classes |

Self-training with hard pseudo-labels **degrades performance across all classes**
(−0.78 pp mean DSC, LAA worst at −1.53 pp).  When the teacher (Dataset 500) is
already near-optimal (0.9823 training-set), hard pseudo-labels introduce noise
rather than new information.  The 100 unlabeled cases come from the same center
and distribution, so they provide no distributional diversity benefit.

This negative result motivates the boundary-aware trainer (B1,
``nnUNetTrainerCTBoundary``) — improving loss design rather than adding noisy
pseudo-labels.

**Overall leaderboard standings (2026-07-21):**

| Task | Our best | 1st place | Gap |
|------|----------|-----------|-----|
| 1 (scar) | G-DSC **0.4791** | OrganAgent **0.4907** | −0.0116 |
| 2 (cavity) | DSC **0.8871** | 0.8886 | −0.15 pp |
| 3 (CT) | DSC 0.9563 | 0.9579 | −0.16 pp |

**OrganAgent** (submitted 2026-07-19) surpassed us on Task 1 by +0.0116 G-DSC,
+0.0204 ACC, and +0.0407 SEN.  Their much higher sensitivity suggests better
scar recall — possibly from extra training data (LAScarQS 2022), a larger
backbone (SwinUNETR / ResEnc L), or more sophisticated anatomical priors (SDM).

**ResEnc M validation results (2026-07-24/25):** Sub12 (native pipeline) and Sub13
(hybrid pipeline) both underperformed our best PlainConvUNet submission (sub10,
G-DSC 0.4791).  ResEnc M achieves higher training-set G-DSC (0.6682 vs 0.6631)
but generalizes poorly to validation (0.4736 hybrid, 0.4324 native).  This confirms
the paper's finding that a properly configured Plain U-Net already lies near the
performance frontier, with further architectural complexity bringing no benefit.

### VNet 4-Stage Progression Experiments (2026-07-26)

All on VNet (4-stage, 6.9M params, SGD, CLAHE).  Training-set 5-fold CV without TTA.

| Configuration | Train Dice | Train G-DSC | Val G-DSC | Notes |
|---------------|:---:|:---:|:---:|-------|
| Baseline (1ch, DC+CE) | 0.3463 | 0.3465 | 0.2189 | sub 2 config |
| + Cavity SDM (2ch) | 0.3573 | 0.3574 | 0.1882 | sub 3 config |
| + 5mm cavity dilation | 0.3309 | 0.3310 | — | post-processing only |

Key: SDM improves train (+0.0110) but degrades val (−0.0307); hard dilation hurts both.

### Multi-Class vs Binary Training (2026-07-26)

Under DC+CE loss, 6-stage Plain U-Net, 5-fold CV:
- 502+501 binary scar: 0.6317
- 502+521 multi-class (cavity + scar): 0.6333 (+0.0016, +0.25%)

Task 2 native-resolution pipeline (sub 7, DSC 0.8871) closes the gap to 1st
place from 0.51 pp to 0.15 pp — the spacing fix clearly benefits Stage 1 on
multi-center data.  Several teams clustered between 0.8835–0.8886.

**Sub 7 breakthrough (2026-07-09):** G-DSC 0.4525 — first submission using
dilation=5mm, #1 by large margin.  **Sub 8 (2026-07-10):** dilation=None →
G-DSC 0.4743, ACC 0.7313, SEN 0.4627 — further +0.022 G-DSC over sub 7,
confirming the training-set trend that removing the LA cavity constraint
entirely is optimal.  All three metrics were #1 on the leaderboard.

**Sub 10 (2026-07-16):** ScarGaussian loss + hybrid pipeline (native Stage 1 +
canonical Stage 2) → G-DSC **0.4791**, ACC 0.736, SEN 0.4722 — +0.0048 G-DSC,
+0.0047 ACC, +0.0095 SEN over sub 8.  ScarGaussian contributes most of the gain;
hybrid pipeline adds stability by keeping Stage 2 at its training resolution.
All three personal-best metrics.

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

Root causes **proved wrong by follow-up experiments** (see Section 6.3.2).

#### 6.3.2 CT V2 + Ablation Experiments (2026-06-24–25)

V2 (IN+Mish+AdamW) was tested in three configurations to isolate
variables.  **V2 underperformed V1 in all settings.**  The
original V1 (BN+ReLU+SGD+MT) remains the best configuration.

| # | Model | Semi | Optimizer | Best Val | Δ vs V1 |
|---|-------|------|-----------|----------|---------|
| V1 orig | BN+ReLU | MT | SGD lr=0.01 | **0.5234** | — |
| V2 sup | IN+Mish | none | AdamW lr=3e-4 | 0.4660 | −5.7 pp |
| Exp A | BN+ReLU | MT | SGD lr=0.01 | 0.4947 | −2.9 pp |
| Exp B | IN+Mish | MT | AdamW lr=3e-4 | 0.4660 | −5.7 pp |

Key conclusions:
- **BN+ReLU+SGD > IN+Mish+AdamW** for this task — the original
  MBAS2024 insight ("Instance Norm is critical") does not apply here;
  the semi-supervised setting with 100 unlabelled volumes benefits
  from BatchNorm's cross-sample statistics.
- **MT semi-supervised helps** — removing it costs ~3–6 pp.
- **PV class weight 6.0 is correct** — reducing to 3.0 dropped
  PV Dice from 0.36 to 0.25–0.30.
- All models share a fundamental problem: **severe foreground
  over-prediction** (LA 1.7–2.5× GT, LAA 2.3–3.7×), suggesting
  the DiceCE loss alone is insufficient.

Config rolled back to match the best V1 checkpoint (Section 6.3).
Next directions: foreground-aware patch sampling, larger patches,
topology-aware losses (clCE/cbDice), FixMatch-style weak→strong
consistency.

#### 6.3.3 Two-Stage Coarse-to-Fine — Ruled Out (2026-06-25)

Unlike MRI where scar tightly surrounds the LA cavity, CT structures
are spatially dispersed.  Analysis of 50 labelled CTs:

| Metric | Value |
|--------|-------|
| Foreground bbox / total volume | **79.9%** |
| LA bbox / total volume | 25.3% |
| LAA bbox / total volume | 46.0% |
| PV centroid → LA centroid | **42.4 mm** |
| LAA centroid → LA centroid | 23.4 mm |

To fully contain LAA within an LA-bbox crop, a **50 mm margin** is
needed — at which point the crop volume ≈ original.  Two-stage
coarse-to-fine is **not viable** for CT.

#### 6.3.4 Loss & Sampling Improvements (2026-06-25)

Switch from DiceCE to FocalTverskyLoss (α=0.7, β=0.3, γ=0.75) to
penalise false positives and reduce foreground over-prediction.
Add Boundary Loss (HausdorffERLoss, weight=0.1) for PV/LAA edge
precision.  Implement clCE (CenterlineCELoss, MICCAI 2024) for
topology preservation of thin tubular structures (PV) — disabled
by default, enable with ``sup_clce=0.5``.

Sampling: increase patch_size 128→160, fg_bias 0.5→0.85 for better
PV coverage; switch to percentile normalisation (nnUNet-style).

| # | Change | Config key | Default | Status |
|---|--------|-----------|---------|--------|
| 1 | FocalTverskyLoss | `tversky_alpha/beta/gamma` | 0.7/0.3/0.75 | ✅ |
| 2 | Boundary loss + clCE | `sup_boundary` / `sup_clce` | 0.1 / 0.0 | ✅ |
| 3 | Percentile norm + class sampling + warm-up | — | — | ✅ |
| 4 | Pretrained encoder (nnUNet weight transfer) | `pretrained_encoder` | `ct_basemodel.safetensors` | ✅ |
| 5 | Full-volume CT val eval + val_ratio=0 | — | — | ✅ |
| 6 | Backbone stored in model_config | — | — | ✅ checkpoint migration done |

**MT extended (300 epochs, 2026-06-09):**

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
  python trainer.py --task ct \
  --db-dir /path/to/CARE2026-LeftAtrium --epochs 300 \
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

#### 6.3.5 nnUNet CT Breakthrough (2026-07-01)

nnUNet (6-stage PlainConvUNet, 30.6M params) achieves full-volume val
**Dice 0.945** (LA 0.976, PV 0.931, LAA 0.926) on the 50 labeled CTs,
dramatically exceeding our VNet-based pipeline (best: 0.81 with
pretrained encoder, 0.52 without).

Root causes of VNet underperformance identified:
- **Model capacity**: VNet 6.9M params vs nnUNet 30.6M — 4-5× gap.
- **Architecture depth**: 4 encoder stages vs nnUNet's 6; nnUNet's last
  stride [1,1,2] preserves XY resolution.
- **Deep supervision**: nnUNet always uses it; we disabled it for CT.
- **Training recipe**: SGD + polyLR 1000 epochs, 5-fold CV, EMA inference.

Two paths forward:
- `CARE2026_CT_nnUNet`: wraps nnUNetPredictor for inference-only use.
  Axis transpose nibabel(x,y,z) ↔ nnUNet(z,y,x) is critical.
- `CARE2026_CT_MT_nnUNet`: Mean Teacher with PlainConvUNet backbone,
  nnUNet pretrained weights as student initialization, 100 unlabeled
  CTs for consistency loss.  Patch [112,112,192], nnUNet normalization.

Self-training script (`scripts/self_train_nnunet.py`): N-fold ensemble
pseudo-labels on 100 unlabeled CTs → retrain nnUNet on 150 cases.

### 6.4 Validation & Submission

nnUNet models deployed for all three tasks.  VNet retained as fallback via
``PredictCfg`` in ``cfg.py``.  Unified CLI:

```bash
# nnUNet (no CLAHE) — best known combination for Task 1
python pipeline.py \
  --mri-stage1-model tmp/nnUNet_results/Dataset502_*/nnUNetTrainer__* \
  --mri-stage2-model tmp/nnUNet_results/Dataset501_*/nnUNetTrainer__* \
  --ct-model tmp/nnUNet_results/Dataset500_*/nnUNetTrainer__* \
  --input_dir /path/to/CARE2026-LeftAtrium \
  --output_dir results/ \
  --tasks 1,2,3

# VNet fallback (uses PredictCfg defaults → checkpoints/)
python pipeline.py \
  --input_dir /path/to/CARE2026-LeftAtrium \
  --output_dir results/ \
  --tasks 1,2,3
```

Output zip uploaded to official validation platform.
Results tracked in ``submissions`` (YAML log).

---

## Phase 7 — Experiments & Ablations ✅

### Core ablations

| Experiment | Goal |
|------------|------|
| Scar-only VNet vs. NestedVNet | Deep supervision benefit for sparse scar segmentation |
| Scar threshold sweep | Optimise Task 1 G-DSC / SEN trade-off |
| Stage-1 threshold sweep | Optimise Task 2 DSC and Task 1 scar constraint region |
| CPS vs. Mean Teacher (CT) | Measure semi-supervised benefit of EMA consistency vs CPS | ✅ Done — MT wins (0.4655 vs 0.3904) |
| CT augmentation sweep | Gamma + brightness/contrast + Gaussian blur; measure per-class Dice impact | Obsolete — nnUNet replaces VNet for CT |
| Boundary Loss weight sweep | Optimise PV / LAA delineation (CT Task 3) | Obsolete — nnUNet replaces VNet for CT |
| Patch size: 128³ vs. 64³ (CT) | Memory vs. accuracy trade-off | Obsolete — nnUNet replaces VNet for CT |

### Experiments from MBAS2024 insights

| Experiment | Goal | Priority | Status |
|------------|------|----------|--------|
| **Extend MRI training: 150 → 400 epochs** | Close epoch-count gap vs. winning teams (1000 epochs) | 🔴 High | ❌ Not pursued (competition ended) |
| **CLAHE preprocessing** — enable `utils/mclahe.py` in MRI dataset | Improve low-SNR scar boundary accuracy | 🔴 High | ✅ Done (auto-detected in predict; `--mclahe` flag in trainer) |
| **Connected-component post-processing** — keep largest component per class | Eliminate segmentation leakage | 🔴 High | ✅ Done (`keep_largest_component`, `postprocess_mri_masks`, `postprocess_ct_mask` in `predict.py`) |
| **Test-time augmentation (TTA)** — flip + 90° rotations, average logits | Low-cost accuracy boost; especially useful for Task 2 domain generalisation | 🔴 High | ✅ Done (8-fold flip TTA in `predict.py`; toggle via `--tta` flag) |
| **5-fold CV + ensemble** — train 5 folds, ensemble predictions | Expected +1–2 pp DSC; mandatory for final submission | ✅ Done — built into nnUNet (all 5 folds trained) | ✅ |
| **SGD + polynomial LR for MRI** (vs. AdamW) | Winning teams used SGD lr=0.01; compare convergence | 🟡 Medium | ✅ Done — S1 0.88→0.93, S2 0.44→0.48 |
| **Slice-position encoding** — append z-coordinate channel to input | Help model handle hard superior/inferior slices | 🟡 Medium | ❌ Not pursued (competition ended) |
| **Histogram matching / intensity standardisation** | Domain-shift mitigation for Task 2 unseen centers without changing the model | 🟡 Medium | ❌ Not pursued (competition ended) |
| **UMamba backbone** — replace VNet encoder with Mamba SSM | Near-ResUNet accuracy, more efficient; possible scar segmentation alternative | 🟢 Low | ❌ Not pursued (competition ended) |
| **Shape-constrained regularisation** — atlas-based prior or topology loss | Anatomy-aware design; reduce leakage at vascular junctions | 🟢 Low | ❌ Not pursued (competition ended) |

---

## Phase 8 — Challenge Submission Pipeline 🔄

- [x] Implement `post_docker_build.py`: validation script (stub replaced with working checker).
- [x] Set up Docker CI (`docker-test.yml` with `status: pre`).
- [x] CT model trained: nnUNet 5-fold ensemble → Task 3 DSC **0.9495** (validation leaderboard).
- [x] MRI nnUNet models trained: Task 1 scar (Dataset 501), Task 2 cavity (Dataset 502); MCLAHE variants (511/512) also done.
- [x] Submit all-task validation predictions to official evaluation platform (Tasks 1, 2, 3).
- [x] nnUNet inference pipeline complete: `CARE2026_MRI_nnUNet` + `PredictCfg` + `_load_model()`.
- [x] Docker end-to-end test.
- [x] Docker submit.
- [x] Docker CI `--shm-size=2g` fix.

---

## Online Validation Leaderboard (official evaluation platform)

### Task 1 (LA scar quantification) — G-DSC / ACC / SEN

| # | Time | G-DSC | ACC | SEN | Model |
|---|------|-------|-----|-----|-------|
| 1 | 2026-06-25 20:02 | 0.2092 | 0.5997 | 0.1997 | vnet_stage2 (1ch), SGD 300ep, CLAHE |
| 2 | 2026-06-29 18:58 | 0.2189 | 0.6041 | 0.2085 | vnet_stage2 (1ch), SGD 300ep, CLAHE, thresh=0.7 |
| 3 | 2026-07-01 22:40 | 0.1882 | 0.5766 | 0.1533 | vnet_stage2_2ch (MRI+SDF), SGD 600ep, CLAHE |
| 4 | 2026-07-06 23:26 | 0.2213 | 0.5744 | 0.1489 | nnUNet 502+501, dilation=2mm[^canonical] |
| 5 | 2026-07-08 01:27 | 0.2230 | 0.5747 | 0.1495 | nnUNet 502+501, TTA on, dilation=2mm[^canonical] |
| 6 | 2026-07-08 16:29 | 0.2236 | 0.5743 | 0.1487 | nnUNet 512+511 (CLAHE), TTA on, dilation=2mm[^canonical] |
| 7 | 2026-07-09 15:11 | 0.4525 | 0.7113 | 0.4227 | nnUNet 502+521, TTA on, dilation=5mm[^canonical] |
| 8 | 2026-07-10 01:53 | 0.4743 | 0.7313 | 0.4627 | nnUNet 502+521, TTA on, dilation=none[^canonical] |
| 9 | 2026-07-14 22:02 | 0.4411 | 0.6944 | 0.3889 | nnUNet 502+521, TTA on, dilation=none[^native] |
| 10 | 2026-07-16 22:02 | **0.4791** | **0.736** | **0.4722** | nnUNet 502+521, ScarGaussian, TTA on, dilation=none[^hybrid] |
| 11 | 2026-07-23 23:54 | 0.4767 | 0.7361 | 0.4722 | nnUNet 502+521, ScarCavityWall, TTA on, dilation=none[^hybrid] |
| 12 | 2026-07-24 13:36 | 0.4324 | 0.6886 | 0.3772 | nnUNet 502+521, ScarGaussian + ResEnc M, TTA on, dilation=none[^native] |
| 13 | 2026-07-25 00:53 | 0.4736 | 0.7309 | 0.4620 | nnUNet 502+521, ScarGaussian + ResEnc M, TTA on, dilation=none[^hybrid] |

### Task 2 (LA cavity segmentation) — DSC / HD (mm)

| # | Time | DSC | HD | Model |
|---|------|-----|----|-------|
| 1 | 2026-06-25 20:02 | 0.8538 | 21.9227 | vnet_stage1, SGD 300ep |
| 2 | 2026-06-29 18:58 | **0.8602** | **18.4552** | vnet_stage1, SGD 300ep, CLAHE |
| 3 | 2026-07-01 22:40 | 0.8602 | 18.4552 | same as #2 (Stage 1 unchanged) |
| 4 | 2026-07-06 23:26 | 0.8832 | 17.549 | nnUNet 502 (5-fold ensemble), 1000ep, no CLAHE[^canonical] |
| 5 | 2026-07-08 01:27 | 0.8828 | 17.6501 | nnUNet 502, TTA on[^canonical] |
| 6 | 2026-07-08 16:29 | 0.8835 | 18.5298 | nnUNet 512 (CLAHE), TTA on[^canonical] |
| 7 | 2026-07-14 22:02 | **0.8871** | **17.5901** | nnUNet 502, TTA on[^native] |

### Task 3 (LA multi-structure segmentation) — DSC / HD (mm)

| # | Time | DSC | HD | Model |
|---|------|-----|----|-------|
| 1 | 2026-06-25 20:02 | 0.4637 | 51.8558 | VNet (BN+ReLU, 6.9M), MT, SGD 300ep |
| 2 | 2026-06-27 22:03 | 0.6788 | 43.8951 | VNet + nnUNet pretrained enc, MT, SGD 800ep |
| 3 | 2026-06-29 18:58 | 0.7511 | 37.953 | nnUNet PlainConvUNet (30.6M), fold_0 only, ep974 |
| 4 | 2026-07-01 22:40 | 0.9495 | 17.9575 | nnUNet PlainConvUNet, fold_0 only |
| 5 | 2026-07-06 23:26 | **0.9558** | **13.3596** | nnUNet 500 (5-fold ensemble), 1000ep |
| 6 | 2026-07-08 01:27 | 0.9563 | 12.4547 | nnUNet 500 (5-fold ensemble), TTA on |
| 7 | 2026-07-13 22:42 | 0.9563 | 12.4561 | nnUNet 503 (self-trained, 5-fold ensemble), TTA on |

[^canonical]: **Canonical pipeline** (``predict_mri_two_stage_legacy``): image resampled to canonical grid (576×576×44), hardcoded spacing (0.625, 0.625, 2.5mm).  Produced sub 7–8 (G-DSC 0.4525–0.4743).
[^native]: **Native pipeline** (``predict_mri_two_stage``): spacing read from NIfTI header, inference at native resolution.  Produced sub 9 (G-DSC 0.4411) and Task 2 sub 7 (DSC 0.8871).
[^hybrid]: **Hybrid pipeline** (``predict_mri_two_stage_hybrid``): native Stage 1 + canonical Stage 2.  Produced sub 10 (G-DSC 0.4791) with ScarGaussian loss.

---
## Test Phase Leaderboard (2026-07-25)

Test set composition per [challenge page](care-webpages/CARE-Left Atrium.md):
- Task 1: 24 LGE-MRIs from Center A
- Task 2: 14 Center A + 20 Center B + 10 Center C (44 total)
- Task 3: 130 CTs from Center D

### Test-sub1 Results

| Task | Metric | Value | Val Best | Δ | Notes |
|------|--------|:---:|:---:|:---:|-------|
| 1 (scar) | G-DSC | **0.4298** | 0.4791 | −0.0493 | 24 Center A cases |
| | ACC | 0.7087 | 0.736 | −0.027 | |
| | SEN | 0.4176 | 0.4722 | −0.0546 | |
| 2 (cavity) | DSC | **0.8285** | 0.8871 | −0.0586 | 44 cases incl. Center B/C |
| | HD | 20.27 | 17.59 | +2.68 | test_4 DSC=0.0 (complete failure) |
| 3 (CT) | DSC | **0.9543** | 0.9563 | −0.0020 | 130 Center D cases |
| | HD | 9.72 | 12.45 | −2.73 | |

**Model**: nnUNet 502+521 (ScarGaussian, 5-fold, hybrid pipeline, TTA on) for Task 1; nnUNet 502 (5-fold, native pipeline, TTA on) for Task 2; nnUNet 500 (5-fold, TTA on) for Task 3.

**Key observations**:
- **Task 3 very stable** (Δ −0.0020 DSC): same-center test, nnUNet generalises well.
- **Task 2 severe drop** (−5.86 pp DSC, HD +2.68 mm): 68% of test cases from unseen centers (B/C) with different scanners and spacings.  test_4 DSC=0.0 suggests extreme spacing/scanner failure.
- **Task 1 moderate drop** (−4.93 pp G-DSC): all Center A but larger test set (24 vs 10 val) + Phase 1 errors propagate.  G-DSC range 0.239–0.561.

---
## Architecture — Why nnUNet > Custom VNet

nnUNet PlainConvUNet (6-stage, 30.6M params, deep supervision at every decoder level)
vs. our best VNet (4-stage, 6.9M params, no deep supervision).

| Stage | Features | Kernel | Stride | Note |
|-------|----------|--------|--------|------|
| 1 | 32 | 1×3×3 | 1,1,1 | Z stride=1 preserves 2.5mm spacing |
| 2 | 64 | 1×3×3 | 1,2,2 | |
| 3 | 128 | 3×3×3 | 1,2,2 | |
| 4 | 256 | 3×3×3 | 2,2,2 | |
| 5 | 320 | 3×3×3 | 2,2,2 | |
| 6 | 320 | 3×3×3 | 2,2,2 | bottleneck |

Gap is large enough (0.35 vs 0.20 G-DSC) that multiple factors compound — architecture depth,
training recipe (SGD+polyLR 1000ep), deep supervision, and data normalisation all contribute.
Ablation experiments below isolate each factor's contribution.

---
## Validation Data Spacing Variability (2026-07-13) ⚠️

**Finding**: Validation data has **different voxel spacings** than training data, even within the same center.  ``predict_mri_two_stage`` previously hardcoded Center A training spacing ``(0.625, 0.625, 2.5)`` for nnUNet inference, which was wrong for all validation cases.

Measured ``nii.header.get_zooms()`` from actual validation files:

| Data | Shape | Spacing (x, y, z) |
|------|-------|-------------------|
| Task 1/2 train (Center A) | 576×576×44 | (0.625, 0.625, 2.5) |
| Task 1 val (Center A) | 576×576×88 | **(1.0, 1.0, 1.0)** |
| Task 2 val_1-10 (Center A) | 864×864×~44 | **(0.347, 0.347, 2.0)** |
| Task 2 val_11+ (Center C) | 640×640×88 | **(1.0, 1.0, 1.0)** |
| Task 3 train (Center D) | variable | variable in-plane, 0.5 z |
| Task 3 val (Center D) | variable | variable in-plane, 0.5 z |

**Impact**: nnUNet predictor uses spacing to resample images to the target spacing defined in plans.json.  Hardcoding wrong spacing caused incorrect resampling → degraded prediction quality.

**Fix**: ``predict_mri_two_stage`` now reads spacing from the NIfTI header.  CT (``predict_ct``) was never affected — it always read spacing from the header.

---
## GT Scar Distribution Analysis (2026-07-12) ✅

**Purpose**: Verify how much of the GT scar tissue actually lies outside the GT LA cavity mask
at various dilation radii.  This is the data-analysis foundation for the paper's main finding.

**Method**: For each of 60 Task 1 training cases, compute the 3D Euclidean distance (in mm,
using spacing=(0.625, 0.625, 2.5) mm) from every GT scar voxel to the nearest GT LA cavity
voxel, using `scipy.ndimage.distance_transform_edt`.  Count the fraction of scar voxels with
distance ≤ dilation_mm as "inside dilated cavity".

**Results** (60 cases, only cases with non-zero scar counted):

| Dilation | % scar INSIDE dilated GT cavity | % scar OUTSIDE | Range (min–max inside) |
|----------|--------------------------------|----------------|------------------------|
| 0 mm     | 16.0 %                         | **84.0 %**     | 4 %–38 %               |
| 1 mm     | 40.6 %                         | 59.4 %         | 16 %–62 %              |
| **2 mm** | **80.7 %**                     | **19.3 %**     | 61 %–94 %              |
| 3 mm     | 93.4 %                         | 6.6 %          | 80 %–99 %              |
| 5 mm     | 98.9 %                         | 1.1 %          | 95 %–100 %             |
| 8 mm     | 99.6 %                         | 0.4 %          | 96 %–100 %             |
| 10 mm    | 99.7 %                         | 0.3 %          | 96 %–100 %             |

**Key insights**:
- **84% of GT scar is outside the GT LA lumen** at 0 mm (not 68% as previously estimated — correct the paper plan).
  This is because LA scar is myocardial fibrosis in the **atrial wall**, not inside the blood-pool lumen.
- Standard 2 mm dilation covers only 80.7% of scar; 19.3% is anatomically excluded even with a perfect Stage 1.
- At 3 mm (≈ atrial wall thickness upper bound) coverage reaches 93.4%.
- At 5 mm coverage is 98.9% — explains why sub 7 (dilation=5mm) dramatically outperformed sub 4-6 (dilation=2mm).
- Combined with Stage 1 prediction errors (DSC ~0.88), the 2 mm constraint effectively clips much more scar.

**Confirmed standard practice**:
- AtrialJSQnet (Li et al., MedIA 2022) explicitly dilates the cavity mask and uses it as a hard post-processing
  constraint (documented in their paper methods + Fig. 3/4).
- TESSLA (MICCAI-STACOM 2022) uses LA blood-pool mask to constrain scar.
- Closely related recent work: Kundu & Linte, "A Two Stage Pipeline for Left Atrial Wall Constrained Scar
  Segmentation" (arXiv:2604.27101, 2026-04-29) — uses SDMs as soft priors; their approach is a training-time
  soft constraint, complementary to our post-processing analysis.  Evaluated on LAScarQS 2022 (Dice 61.1%).

**Paper implication**: The 84% figure directly supports the paper's central claim.  The title should reflect
"scar beyond the lumen / cavity" rather than "dilation trap" (too jargon-heavy).
Suggested title: *"Scar Beyond the Lumen: Rethinking Anatomical Constraints for Left Atrial Scar Segmentation"*.

---
## Immediate Next Steps

### Priority: Task 3 CT — close 0.16 pp gap to 1st place

Current: nnUNet Dataset 500, 5-fold ensemble, DSC **0.9563** (#2).  Gap to #1: **0.16 pp**.

**Self-training (Dataset 503) — ❌ Failed.**  Training on 50 GT + 100 hard
pseudo-labels decreased mean DSC by −0.78 pp vs. the 50-label baseline.
Teacher (500) is too strong — pseudo-labels add noise, not signal.

**Next: Boundary-aware nnUNet (B1).**  ``nnUNetTrainerCTBoundary`` is ready
(``models/custom_nnunet.py``).  Adds HausdorffERLoss (PV/LAA boundary) +
CenterlineCELoss (PV topology) to standard DiceCE.  Train on Dataset 500:

```bash
export nnUNet_extTrainer="$PWD/models"
for f in 0 1 2 3 4; do
  nnUNetv2_train 500 3d_fullres $f -tr nnUNetTrainerCTBoundary
done
```

| # | Strategy | Rationale | Status |
|---|----------|-----------|--------|
| **B1** | **Boundary-aware nnUNet** | HausdorffERLoss + CenterlineCELoss for PV/LAA | ❌ Tested — training-set Mean DSC 0.9812 (−0.0011 vs baseline) |
| S1 | Soft pseudo-labels | KL divergence on unlabeled cases | De-prioritized after 503 failure |
| S3 | Ensemble 500 + 503 | Different training data → complementary errors | Low-cost, try after B1 |

### MRI — reclaim Task 1 #1 (surpassed by OrganAgent +0.0116 G-DSC on 2026-07-19)

| # | Task | Description | Status |
|---|------|-------------|--------|
| D2 | Dataset 600 | nnUNet on 2-class full volume (backup) | ❌ Cancelled (competition ended) |
| D3 | Custom nnUNet loss | `nnUNetTrainerScarGaussian` — test on 521 | ✅ Done — sub 10 G-DSC 0.4791 (+0.0048) |
| E1 | **ResEnc M backbone** | `nnUNetPlannerResEncM` + ScarGaussian on Dataset 521 — residual encoder for architecture gain | ✅ Done — Training-set G-DSC 0.6682 (+0.0051 vs PlainConvUNet); Val: native 0.4324 (sub12), hybrid 0.4736 (sub13). ResEnc M **does not improve** over PlainConvUNet on validation (best Plain 0.4791 > ResEnc M 0.4736). |
| E2 | **Cavity-wall spatial loss** | `nnUNetTrainerScarCavityWall` on Dataset 521 — cavity blur weight instead of scar blur | ✅ Done — Val leaderboard G-DSC 0.4767 (−0.0024 vs ScarGaussian 0.4791), ACC 0.7361, SEN 0.4722. CavityWall does not improve over ScarGaussian. |
| E3 | **ResEnc M + CavityWall** | Combined: ResEnc M plans + CavityWall trainer | ❌ Cancelled — ResEnc M alone does not improve validation; combining with CavityWall unlikely to help. |
| E4 | **ResEnc L (M confirmed gain)** | ResEnc L + ScarGaussian on Dataset 521 — scale up now that M proves +0.0051 G-DSC | ❌ Cancelled — ResEnc M does not improve validation; scaling to L not justified. |
| B1 | VNet + nnUNet recipe | Train VNet on Dataset 501, SGD+polyLR 1000ep | ❌ Cancelled (competition ended) |
| B2 | 4-stage nnUNet | PlainConvUNet reduced to 4 stages | ❌ Cancelled (competition ended) |

### Infrastructure

| # | Task | Description |
|---|------|-------------|
| C1 | Docker build & test | ✅ Finalized: `post_docker_build.py`, local smoke test, CI `--shm-size=2g` fix |
| C2 | Docker submit | ✅ Verified: all 3 tasks match best submissions (T1 Dice 1.0000, T2 Dice 1.0000, T3 DSC 0.9929). Output dir fixed (task1/task2/task3). Rebuild + smoke test pending. Emails drafted. |

---

## Notes & Open Questions

- **Task 2 cross-scanner domain shift**: The spacing fix resolves physical resampling errors, but the remaining gap to 0.89+ DSC is likely due to scanner-level intensity differences (Center A: Siemens 1.5T/3T vs. Centers B/C: Philips 1.5T).  Possible inference-time mitigations (no retraining needed, challenge-compliant):
  - **Histogram matching**: map test image intensity CDF to a Center A template → Philips images "look like" Siemens.
  - **Per-case Z-score**: ``(img - mean) / std`` per individual volume instead of global normalization.
  - **BN Test-Time Adaptation**: forward pass on test data to update BN running stats (no weight update).

- **Task 3 label format**: challenge page says `cardiacSegImgMO.nii.gz` but training data uses `label_XXXX.nii.gz` — must handle both names in `data_reader.py` when validation/test data is released.
- **G-DSC for Task 1**: evaluated alongside ACC and SEN which require binary (0/1) predictions.  Soft probability maps are NOT accepted — must submit uint8 binary mask.  G-DSC weight w_c = 1/V_c² gives the rare scar class orders of magnitude more influence than background.
- **Prizes**: only Tasks 1 and 3 are prize-eligible; prioritise these two.
