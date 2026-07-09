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

### 6.2 MRI Stage 2 — Scar-Only Segmenter — 🔄 Iterating

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

**Backbone experiments (all with SGD+polyLR, CLAHE):**

| Backbone | Epochs | In Channels | Task 1 G-DSC | Notes |
|----------|--------|------------|-------------|-------|
| `vnet_stage2` | 300 | 1 (MRI only) | **0.2189** | current best |
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

**Task 1 — LA Scar (60 labeled cases, 5-fold ensemble):**

| Model | G-DSC | ACC | SEN |
|-------|-------|-----|-----|
| VNet (S1+S2, CLAHE, thresh=0.5) | 0.2034 | 0.9997 | 0.1946 |
| nnUNet 502+501 (no CLAHE) | 0.3529 | 0.9998 | 0.2573 |
| nnUNet 512+511 (CLAHE) | 0.3335 | 0.9998 | 0.2390 |
| nnUNet 502+521 (multi-class, no CLAHE) | 0.3533 | 0.9998 | 0.2577 |
| nnUNet 502+521, dilation=5mm | **0.6221** | 0.9998 | 0.6267 |
| nnUNet 502+501, dilation=5mm | 0.6211 | 0.9998 | 0.6242 |

nnUNet no-CLAHE improves G-DSC by **+73 %** over VNet on the same training data.

**Dilation ablation (training set, 60 cases):**

| Dilation | 502+501 (binary) | 502+521 (multi-class) | 512+511 (CLAHE) |
|----------|------|------|------|
| none | **0.6317** | **0.6333** | 0.5976 |
| 2mm (old default) | 0.3529 | 0.3533 | 0.3138 |
| 5mm | 0.6211 | 0.6221 | 0.5875 |
| 10mm | 0.6306 | 0.6321 | 0.5969 |

Key findings:
- **Remove LA constraint entirely** (dilation=none) — highest G-DSC across all models.
  LA cavity prediction errors clip true scar; even 10mm dilation loses ~0.1pp vs none.
- Dilation is the **dominant factor** (+0.28 G-DSC from 2mm→none).
- **CLAHE consistently harmful** (~3-4 pp lower at every dilation).
- **Multi-class (521) ≈ binary (501)** — at best 0.15 pp difference, negligible.
- Default changed: ``postprocess_mri_masks(dilation_mm=None)`` — no LA constraint. Top-1
result is dilation, not data strategy.

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

| Class | DSC |
|-------|-----|
| LA (left atrium) | 0.9938 |
| PV (pulmonary veins) | 0.9832 |
| LAA (left atrial appendage) | 0.9700 |
| **Mean** | **0.9823** |

Training-set mean DSC 0.9823 vs. validation leaderboard 0.9558 (nnUNet 5-fold).
Competitive with 1st place (0.9579) — only 0.21 pp away on the leaderboard.

**Overall leaderboard standings (2026-07-06 submission):**

| Task | nnUNet | 1st place | Gap |
|------|--------|-----------|-----|
| 1 (scar) | G-DSC 0.2213 | 0.4409 | −49 % |
| 2 (cavity) | DSC 0.8832 | 0.8886 | −0.6 % |
| 3 (CT) | DSC 0.9558 | 0.9579 | −0.2 % |

Task 2 and 3 are competitive.  Task 1 (scar) is the primary bottleneck.
The 5mm dilation fix (2mm → 5mm) raises training-set G-DSC from 0.35 to
0.62, suggesting the old post-processing constraint was the dominant problem
rather than model quality.  Next validation submission expected to close
significant portion of the gap to 1st place (0.4409).

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
  --input_dir /Data1/wenh06/CARE2026-LeftAtrium \
  --output_dir results/ \
  --tasks 1,2,3

# VNet fallback (uses PredictCfg defaults → checkpoints/)
python pipeline.py \
  --input_dir /Data1/wenh06/CARE2026-LeftAtrium \
  --output_dir results/ \
  --tasks 1,2,3
```

Output zip uploaded to official validation platform.
Results tracked in ``submissions`` (YAML log).

---

## Phase 7 — Experiments & Ablations ⏳

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
| **Extend MRI training: 150 → 400 epochs** | Close epoch-count gap vs. winning teams (1000 epochs) | 🔴 High | ⏳ |
| **CLAHE preprocessing** — enable `utils/mclahe.py` in MRI dataset | Improve low-SNR scar boundary accuracy | 🔴 High | ✅ Done (auto-detected in predict; `--mclahe` flag in trainer) |
| **Connected-component post-processing** — keep largest component per class | Eliminate segmentation leakage | 🔴 High | ✅ Done (`keep_largest_component`, `postprocess_mri_masks`, `postprocess_ct_mask` in `predict.py`) |
| **Test-time augmentation (TTA)** — flip + 90° rotations, average logits | Low-cost accuracy boost; especially useful for Task 2 domain generalisation | 🔴 High | ✅ Done (8-fold flip TTA in `predict.py`; toggle via `--tta` flag) |
| **5-fold CV + ensemble** — train 5 folds, ensemble predictions | Expected +1–2 pp DSC; mandatory for final submission | ✅ Done — built into nnUNet (all 5 folds trained) | ✅ |
| **SGD + polynomial LR for MRI** (vs. AdamW) | Winning teams used SGD lr=0.01; compare convergence | 🟡 Medium | ✅ Done — S1 0.88→0.93, S2 0.44→0.48 |
| **Slice-position encoding** — append z-coordinate channel to input | Help model handle hard superior/inferior slices | 🟡 Medium | ⏳ |
| **Histogram matching / intensity standardisation** | Domain-shift mitigation for Task 2 unseen centers without changing the model | 🟡 Medium | ⏳ |
| **UMamba backbone** — replace VNet encoder with Mamba SSM | Near-ResUNet accuracy, more efficient; possible scar segmentation alternative | 🟢 Low | ⏳ |
| **Shape-constrained regularisation** — atlas-based prior or topology loss | Anatomy-aware design; reduce leakage at vascular junctions | 🟢 Low | ⏳ |

---

## Phase 8 — Challenge Submission Pipeline 🔄

- [x] Implement `post_docker_build.py`: validation script (stub replaced with working checker).
- [x] Set up Docker CI (`docker-test.yml` with `status: pre`).
- [x] CT model trained: nnUNet 5-fold ensemble → Task 3 DSC **0.9495** (validation leaderboard).
- [x] MRI nnUNet models trained: Task 1 scar (Dataset 501), Task 2 cavity (Dataset 502); MCLAHE variants (511/512) also done.
- [x] Submit all-task validation predictions to official evaluation platform (Tasks 1, 2, 3).
- [x] nnUNet inference pipeline complete: `CARE2026_MRI_nnUNet` + `PredictCfg` + `_load_model()`.
- [ ] Docker end-to-end test.
- [ ] Docker submit.

---

## Online Validation Leaderboard (official evaluation platform)

### Task 1 (LA scar quantification) — G-DSC / ACC / SEN

| # | Time | G-DSC | ACC | SEN | Model |
|---|------|-------|-----|-----|-------|
| 1 | 2026-06-25 20:02 | 0.2092 | 0.5997 | 0.1997 | vnet_stage2 (1ch), SGD 300ep, CLAHE |
| 2 | 2026-06-29 18:58 | **0.2189** | 0.6041 | 0.2085 | vnet_stage2 (1ch), SGD 300ep, CLAHE, thresh=0.7 |
| 3 | 2026-07-01 22:40 | 0.1882 | 0.5766 | 0.1533 | vnet_stage2_2ch (MRI+SDF), SGD 600ep, CLAHE |
| 4 | 2026-07-06 23:26 | 0.2213 | 0.5744 | 0.1489 | nnUNet 502+501, dilation=2mm |
| 5 | 2026-07-08 01:27 | 0.2230 | 0.5747 | 0.1495 | nnUNet 502+501, TTA on, dilation=2mm |
| 6 | 2026-07-08 16:29 | **0.2236** | 0.5743 | 0.1487 | nnUNet 512+511 (CLAHE), TTA on, dilation=2mm |
| 7 | 2026-07-08 ??:?? | — | — | — | nnUNet 502+521, TTA on, dilation=5mm |

### Task 2 (LA cavity segmentation) — DSC / HD (mm)

| # | Time | DSC | HD | Model |
|---|------|-----|----|-------|
| 1 | 2026-06-25 20:02 | 0.8538 | 21.9227 | vnet_stage1, SGD 300ep |
| 2 | 2026-06-29 18:58 | **0.8602** | **18.4552** | vnet_stage1, SGD 300ep, CLAHE |
| 3 | 2026-07-01 22:40 | 0.8602 | 18.4552 | same as #2 (Stage 1 unchanged) |
| 4 | 2026-07-06 23:26 | **0.8832** | **17.549** | nnUNet 502 (5-fold ensemble), 1000ep, no CLAHE |

### Task 3 (LA multi-structure segmentation) — DSC / HD (mm)

| # | Time | DSC | HD | Model |
|---|------|-----|----|-------|
| 1 | 2026-06-25 20:02 | 0.4637 | 51.8558 | VNet (BN+ReLU, 6.9M), MT, SGD 300ep |
| 2 | 2026-06-27 22:03 | 0.6788 | 43.8951 | VNet + nnUNet pretrained enc, MT, SGD 800ep |
| 3 | 2026-06-29 18:58 | 0.7511 | 37.953 | nnUNet PlainConvUNet (30.6M), fold_0 only, ep974 |
| 4 | 2026-07-01 22:40 | 0.9495 | 17.9575 | nnUNet PlainConvUNet, fold_0 only |
| 5 | 2026-07-06 23:26 | **0.9558** | **13.3596** | nnUNet 500 (5-fold ensemble), 1000ep |

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
## Immediate Next Steps

### Analysis (zero training cost)

| # | Task | Description |
|---|------|-------------|
| A1 | Convergence curves | Parse nnUNet `training_log_*.txt` vs VNet CSV → prove nnUNet converges faster to higher ceiling (`utils/parse_nnunet_log.py` ready) |
| A2 | Per-case DSC | 60 cases individually — which gain most (large/small LA, SNR)? |
| A3 | Failure analysis | Cases where even nnUNet fails — thin wall, low contrast, boundary ambiguity? |
| A4 | MCLAHE ablation write-up | Why CLAHE helps VNet but hurts nnUNet (ZScoreNormalization vs global z-score) |

### Training (need GPU time)

| # | Task | Description |
|---|------|-------------|
| **D1** | **Dataset 521** | nnUNet on 2-class crop (cavity+scar joint label) — ✅ Done. G-DSC 0.3533 vs 0.3529 (501) at 2mm, 0.6221 vs 0.6211 at 5mm. Negligible difference; dilation, not label strategy, is the key factor. |
| **D2** | **Dataset 600** | nnUNet on 2-class full volume (backup: 60 cases may not be enough) |
| **D3** | **Custom nnUNet loss** | `nnUNetTrainerScarWeighted` (class weights) + `nnUNetTrainerScarGaussian` (Gaussian spatial weight map) — ✅ Ready |
| B1 | VNet + nnUNet recipe | Train VNet on Dataset 501, SGD+polyLR 1000ep → isolate "architecture" from "training recipe" |
| B2 | 4-stage nnUNet | PlainConvUNet reduced to 4 stages (~12M params) → isolate depth + deep supervision |
| B3 | CT self-training | `scripts/self_train_nnunet.py` → 5-fold pseudo-labels on 100 unlabeled CT → retrain nnUNet on 150 cases |

### Infrastructure

| # | Task | Description |
|---|------|-------------|
| C1 | Docker build & test | Finalize `post_docker_build.py`, build image, end-to-end smoke test |
| C2 | Docker submit | Three images (one per task) |

---

## Notes & Open Questions

- **Task 2 domain shift**: Center B/C validation data (20 samples) is available for fine-tuning / histogram matching — check challenge rules whether this is permitted.
- **Task 3 label format**: challenge page says `cardiacSegImgMO.nii.gz` but training data uses `label_XXXX.nii.gz` — must handle both names in `data_reader.py` when validation/test data is released.
- **G-DSC for Task 1**: evaluated alongside ACC and SEN which require binary (0/1) predictions.  Soft probability maps are NOT accepted — must submit uint8 binary mask.  G-DSC weight w_c = 1/V_c² gives scar class enormous influence (~40,000× vs background).
- **Prizes**: only Tasks 1 and 3 are prize-eligible; prioritise these two.
