# CARE 2026 Left Atrium — Development Roadmap

> **Tasks**: (1) LA scar quantification from LGE-MRI; (2) LA cavity segmentation from LGE-MRI with cross-center domain shift; (3) Multi-structure LA segmentation from CT with semi-supervised learning.

---

## Approach Overview

Three tasks are evaluated independently; no cross-modal fusion is required by the challenge.

### Tasks 1 & 2 — LGE-MRI (Multi-task V-Net)

We train a **single multi-output 3D V-Net** on all 190 MRI samples.  The network has two segmentation heads branching from a shared encoder:

```
LGE-MRI volume (H, W, D)
       │
       ├─ MCLAHE preprocessing → intensity normalisation [0, 1]
       │
       ├─ Shared 3D V-Net encoder + decoder body
       │
       ├─ Head A: LA cavity (sigmoid binary) — trained on all 190 samples (Tasks 1 + 2)
       │
       └─ Head B: LA scar   (sigmoid binary) — trained on  60 samples  (Task 1 only)
```

Why multi-task?  Task 1 comes with both LA cavity and scar labels; Task 2 has only LA cavity labels.  The shared encoder benefits from 190 training samples for geometry understanding, while the scar head uses the 60 annotated samples only.  At inference, a two-stage crop-and-refine strategy is applied: first predict LA cavity (Head A), then crop to the bounding box and re-run to predict scars (Head B).

Loss (Task 1 combined head):

```
L_total = λ₁ · L_dice(LA) + λ₂ · L_dice(scar) + λ₃ · L_boundary(scar) + λ₄ · L_focal(scar)
```

The scar class is heavily imbalanced; Boundary Loss + Focal Loss compensate.

Domain generalisation for Task 2 test set (Centers B & C unseen during training):
- **Instance Normalisation** instead of Batch Norm in the encoder.
- **Aggressive augmentation**: random gamma, random intensity shift, elastic deformation, random flip.
- **Histogram matching**: match Center B/C validation images to Center A statistics at inference.

### Task 3 — CT (Semi-supervised 3D U-Net)

100 of 150 training CTs have no labels; the model must leverage unlabelled data.

**Cross Pseudo Supervision (CPS)** with two parallel 3D U-Nets:

```
Labelled batch
       ├─ Model 1 forward → supervised loss (Dice + CE)
       └─ Model 2 forward → supervised loss (Dice + CE)

Unlabelled batch
       ├─ Model 1 forward → pseudo-labels → supervise Model 2
       └─ Model 2 forward → pseudo-labels → supervise Model 1
```

Loss:

```
L_total = L_sup(M1) + L_sup(M2) + λ_cps · [L_cps(M1←M2) + L_cps(M2←M1)]
```

`λ_cps` is ramped up from 0 → 1 over 20 epochs to avoid early noisy pseudo-labels.

Additional tricks:
- CT windowing to cardiac soft-tissue window (W=350, L=50 HU).
- Resample all volumes to uniform 0.5 × 0.5 × 0.5 mm isotropic.
- Deep supervision at each decoder scale.
- Boundary Loss for PV (pulmonary veins) and LAA (left atrial appendage) which are thin / small.

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

Define project-wide constants:

| Constant | Value | Notes |
|----------|-------|-------|
| `MRI_CENTERS` | `["A"]` | Training centers for MRI |
| `CT_CENTERS` | `["D"]` | Training centers for CT |
| `MRI_SPACING` | `(0.625, 0.625, 2.5)` | mm, Center A |
| `CT_SPACING_ISOTROPIC` | `(0.5, 0.5, 0.5)` | mm, target resampling |
| `MRI_PATCH_SIZE` | `(128, 128, 32)` | training patch size |
| `CT_PATCH_SIZE` | `(128, 128, 128)` | training patch size |
| `MRI_CLASS_MAP_T1` | `{0: "bg", 1: "LA scar"}` | Task 1 |
| `MRI_CLASS_MAP_T2` | `{0: "bg", 1: "LA cavity"}` | Task 2 |
| `CT_CLASS_MAP` | `{0: "bg", 1: "LA", 2: "PV", 3: "LAA"}` | Task 3 |
| `CT_LABELED_COUNT` | `50` | labelled training CTs |
| `CT_UNLABELED_COUNT` | `100` | unlabelled training CTs |
| `TASK1_TRAIN_COUNT` | `60` | Task 1 training MRIs |
| `TASK2_TRAIN_COUNT` | `130` | Task 2 training MRIs |

### 2.2 `cfg.py`

Implement `BaseCfg`, `TrainCfg`, `ModelCfg` using `torch_ecg.cfg.CFG`.

Key settings:

| Config | Value |
|--------|-------|
| `TrainCfg.n_epochs` | 150 (MRI), 200 (CT) |
| `TrainCfg.batch_size` | 2 (MRI), 2 (CT) |
| `TrainCfg.patch_size` | per `const.py` |
| `TrainCfg.optimizer` | `"adamw_amsgrad"` |
| `TrainCfg.lr` | `1e-3` |
| `TrainCfg.lr_scheduler` | `"cosine_annealing"` |
| `ModelCfg.vnet` | V-Net config dict (channels, blocks, etc.) |
| `ModelCfg.loss` | compound loss weights `λ₁..λ₄` |

### 2.3 `dataset.py`

Implement PyTorch Dataset classes:

- `CARE2026_MRI_Dataset`: random 3D patch cropping around LA bounding box; online augmentation (flip, elastic, gamma); returns `(image, la_mask, scar_mask, has_scar)` — `has_scar=False` for Task 2 samples.
- `CARE2026_CT_Dataset`: CT windowing; resample to isotropic 0.5 mm; random patch; returns `(image, mask, is_labeled)`.
- `collate_fn_mri` and `collate_fn_ct`.

---

## Phase 3 — Model Implementation ⏳

### 3.1 `models/vnet.py` — multi-output V-Net for MRI

Extend the stub with:
- Two decoder heads (LA cavity + scar) branching from the bottleneck.
- Config-driven: `num_heads`, `head_channels`, `shared_depth` (how far up the decoder is shared).
- Instance Norm option (replacing Batch Norm) for domain generalisation.
- Deep supervision: auxiliary losses at each decoder scale.

### 3.2 `models/nested_vnet.py` — nested / V-Net++ for MRI (optional)

Nested skip connections (similar to UNet++). May improve scar boundary delineation.

### 3.3 CT U-Net (`models/unet3d.py`)

New file.  Symmetric 3D U-Net with:
- Residual blocks in encoder and decoder.
- Deep supervision.
- Batch Norm (CT single-center, no domain shift needed within training).
- Multi-class softmax output (4 classes).

### 3.4 `models/loss/`

| File | Contents |
|------|----------|
| `region_loss.py` | `DiceLoss`, `FocalLoss`, `CrossEntropyLoss3D` |
| `boundary_loss.py` | `BoundaryLoss` (distance-transform-based) |
| `distribution_loss.py` | `KLDivergenceLoss` (for consistency regularisation in CPS) |
| `compound_loss.py` | `CompoundLoss`: configurable weighted sum of any combination |

---

## Phase 4 — Training Loops ⏳

### 4.1 MRI trainer (`trainer.py`)

`CARE2026_MRI_Trainer`:
- Multi-task loss: applies Head A loss on all batches, Head B loss only when `has_scar=True`.
- Metric: DSC (Task 2 / LA head); G-DSC + ACC + SEN (Task 1 / scar head).
- Validation set: 10 Center A samples (Task 1), 20 samples (Task 2, including 10 Center C for domain shift check).
- Checkpointing: save best scar DSC for Task 1, best LA DSC for Task 2.

### 4.2 CT trainer (`trainer.py`)

`CARE2026_CT_Trainer`:
- CPS loss: ramped consistency weight `λ_cps(t)`.
- Maintain two independent models (M1, M2) and two optimisers.
- Metric: per-class DSC (LA / PV / LAA) and mean HD.
- Save best mean DSC checkpoint.

---

## Phase 5 — Prediction & Post-processing ⏳

`predict.py`:
- **MRI inference**: slide window or crop-to-bbox, ensemble flip TTA.
- **MRI post-processing (scar)**: keep only scar voxels within LA cavity mask.
- **CT inference**: sliding window (128³, stride 64³), soft-max vote over overlapping patches.
- **CT post-processing**: largest connected component per class; morphological closing on LAA.

`outputs.py`: `CARE2026Outputs` dataclass wrapping predictions and file-save helpers (`save_as_nifti`, `evaluate`).

`pipeline.py`: end-to-end `run(record, task, model, output_dir)`.

---

## Phase 6 — Validation & Analysis ⏳

After training converges:

- Compute per-task metrics on local validation split.
- Task 1: G-DSC / ACC / SEN curves vs. epoch.
- Task 2: domain shift analysis — Center A val vs. Center C val DSC gap.
- Task 3: per-class DSC (LA / PV / LAA) breakdown; CPS unlabelled sample quality curve.
- Visualise predictions with `data_reader.view_data()`.

---

## Phase 7 — Experiments & Ablations ⏳

| Experiment | Goal |
|------------|------|
| Multi-task MRI vs. separate models | Verify joint training helps scar head |
| Instance Norm vs. Batch Norm (MRI) | Quantify domain generalisation gain |
| CPS vs. supervised-only (CT) | Measure semi-supervised benefit |
| Boundary Loss weight sweep | Optimise PV / LAA delineation |
| TTA (flip / rotation) | Measure inference-time accuracy boost |
| Patch size: 128³ vs. 64³ (CT) | Memory vs. accuracy trade-off |

---

## Phase 8 — Challenge Submission Pipeline ⏳

- [ ] Implement `post_docker_build.py`: cache trained weights from cloud storage.
- [ ] Verify Docker build (`docker-test.yml`) passes with `status: pre`.
- [ ] Run full training (MRI: 150 epochs, CT: 200 epochs); save best checkpoints.
- [ ] End-to-end inference smoke test inside Docker container.
- [ ] Set `status: alpha` in `docker-test.yml`; enable dataset download step.
- [ ] Submit to official evaluation platform.
- [ ] Set `status: final` for final submission.

---

## Immediate Next Steps

1. **`const.py`**: define all constants listed in Phase 2.1.
2. **`cfg.py`**: implement `BaseCfg` / `TrainCfg` / `ModelCfg` with V-Net and U-Net config blocks.
3. **`dataset.py`**: `CARE2026_MRI_Dataset` with bounding-box crop + augmentation; `CARE2026_CT_Dataset` with CT windowing + patch sampling.
4. **`models/vnet.py`**: extend stub to dual-head multi-task V-Net (shared encoder, two decoder heads).
5. **`models/unet3d.py`**: new file — 3D residual U-Net for CT semi-supervised training.
6. **`models/loss/region_loss.py`** and **`models/loss/compound_loss.py`**: implement Dice, Focal, CE, compound loss.
7. **`trainer.py`**: MRI multi-task trainer first (simpler, no CPS); CT CPS trainer second.

---

## Notes & Open Questions

- **Task 2 domain shift**: Center B/C validation data (20 samples) is available for fine-tuning / histogram matching — check challenge rules whether this is permitted.
- **Task 3 label format**: challenge page says `cardiacSegImgMO.nii.gz` but training data uses `label_XXXX.nii.gz` — must handle both names in `data_reader.py` when validation/test data is released.
- **G-DSC for Task 1**: the scar label is used as a _continuous_ weight map in G-DSC computation; model's soft output (pre-sigmoid) may score better than hard threshold.
- **Prizes**: only Tasks 1 and 3 are prize-eligible; prioritise these two.
