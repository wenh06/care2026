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

## Phase 3 — Model Implementation ✅

### 3.1 `models/layers.py` — shared 3D building blocks

`ConvNormAct`, `ResBlock3D`, `DownBlock3D`, `UpBlock3D`, `NestedUpBlock3D`.

### 3.2 `models/vnet.py` — V-Net for MRI & CT

- `_SegEncoder3D`: shared encoder (stem + DownBlock3D stack).
- `DualHeadVNet`: shared encoder → two independent decoders (LA cavity + scar). **Instance Norm** for domain generalisation.
- `VNet`: shared encoder → single decoder. **Batch Norm**. Used as both Model 1 and Model 2 in CPS for CT.

### 3.3 `models/nested_vnet.py` — Nested V-Net (UNet++) for MRI

`DualHeadNestedVNet`: same `_SegEncoder3D` encoder as above + two `_NestedDecoder` paths.
- Dense skip connections across all encoder levels (UNet++ node grid).
- **Deep supervision**: returns one logit tensor per decoder level (coarse → fine); loss averaged across levels. Particularly helpful for superior/inferior slice regions (see MBAS2024 insights below).

### 3.4 `models/loss/`

| File | Contents |
|------|----------|
| `dice_loss.py` | `SoftDiceLoss`, `DiceCELoss`, `TverskyLoss`, `FocalTverskyLoss` |
| `boundary_loss.py` | `BoundaryLoss` (on-the-fly scipy distance transform) |
| `__init__.py` | `MRILoss` (LA DiceCE + scar Tversky/Focal/Boundary), `CTLoss` (supervised DiceCE + CPS CE) |

### 3.5 `models/__init__.py` — model wrappers

`CARE2026_MRI_Model`: wraps `DualHeadVNet` or `DualHeadNestedVNet` (controlled by `backbone=` arg); loss computed inside `forward()` when labels are provided; handles deep-supervision list output.

`CARE2026_CT_Model`: wraps two `VNet` instances for CPS; loss includes supervised DiceCE + consistency CE.

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
| **Labeling strategy matters**: separating cavity and wall heads consistently outperformed joint multi-class models | Validates our dual-head design (LA cavity head + scar head) |

### Hardest failure modes (directly relevant to our tasks)

| Failure mode | Where it hurts us | Mitigation |
|---|---|---|
| **Domain shift** — wall/scar DSC drops ~10 pp at unseen center | Task 2 (cross-center LA segmentation) | InstanceNorm; TTA; test-time BN adaptation |
| **Post-ablation scar signal ≈ atrial wall** — models confuse the two | Task 1 (scar quantification) | Dual-head avoids mixing; Tversky/Focal losses for imbalance |
| **Superior/inferior slice degradation** — U-shaped DSC profile along z-axis; complex vascular junctions | All tasks | Deep supervision (NestedVNet); slice-position feature |
| **Segmentation leakage** — predictions bleed into adjacent structures | All tasks | Connected-component post-processing (keep largest component) |
| **Low-SNR images** — wall/scar DSC strongly correlated with SNR | Task 1 scar boundary accuracy | CLAHE preprocessing (`utils/mclahe.py` already exists) |

### UMamba as alternative backbone

UMambaBot (Mamba-based state-space model) achieved near-ResUNet performance with better computational efficiency. Worth evaluating as a drop-in backbone replacement for MRI tasks if VNet under-performs.

---

## Phase 4 — Training Loops ✅

### 4.1 MRI trainer (`trainer.py`)

`CARE2026_MRI_Trainer`:
- Multi-task loss: applies Head A loss on all batches, Head B loss only when `has_scar=True`.
- Metric: local validation `la_dice`, `scar_dice`, `scar_acc`, `scar_sen`.
- Config-driven monitor key (default `la_dice`) for checkpoint selection.
- Supports `backbone="vnet"` and `backbone="nested_vnet"` via the wrapped MRI model.
- Uses `BaseTrainer` logging / checkpointing, with model-contained loss.

### 4.2 CT trainer (`trainer.py`)

`CARE2026_CT_Trainer`:
- CPS loss: ramped consistency weight `λ_cps(t)`.
- Wrapped model maintains two internal VNet branches (`model1`, `model2`) under one optimiser.
- Metric: local validation `ct_dice_la`, `ct_dice_pv`, `ct_dice_laa`, `ct_mean_dice`.
- Uses mixed labeled + unlabeled training split, labeled-only validation split.

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

### Core ablations

| Experiment | Goal |
|------------|------|
| Multi-task MRI vs. separate models | Verify joint training helps scar head |
| `DualHeadVNet` vs. `DualHeadNestedVNet` | Deep supervision benefit; especially for superior/inferior slices |
| Instance Norm vs. Batch Norm (MRI) | Quantify domain generalisation gain for Task 2 |
| CPS vs. supervised-only (CT) | Measure semi-supervised benefit on 50 labelled / 100 unlabelled |
| Boundary Loss weight sweep | Optimise PV / LAA delineation (CT Task 3) |
| Patch size: 128³ vs. 64³ (CT) | Memory vs. accuracy trade-off |

### Experiments from MBAS2024 insights

| Experiment | Goal | Priority |
|------------|------|----------|
| **Extend MRI training: 150 → 400 epochs** | Close epoch-count gap vs. winning teams (1000 epochs) | 🔴 High |
| **CLAHE preprocessing** — enable `utils/mclahe.py` in MRI dataset | Improve low-SNR scar boundary accuracy | 🔴 High |
| **Connected-component post-processing** — keep largest component per class | Eliminate segmentation leakage | 🔴 High |
| **Test-time augmentation (TTA)** — flip + 90° rotations, average logits | Low-cost accuracy boost; especially useful for Task 2 domain generalisation | 🔴 High |
| **5-fold CV + ensemble** — train 5 folds, ensemble predictions | Expected +1–2 pp DSC; mandatory for final submission | 🟡 Medium (Phase 8) |
| **SGD + polynomial LR for MRI** (vs. AdamW) | Winning teams used SGD lr=0.01; compare convergence | 🟡 Medium |
| **Slice-position encoding** — append z-coordinate channel to input | Help model handle hard superior/inferior slices | 🟡 Medium |
| **Test-time BN/IN adaptation** — update norm stats on test volume | Domain-shift mitigation for Task 2 unseen centers | 🟡 Medium |
| **UMamba backbone** — replace VNet encoder with Mamba SSM | Near-ResUNet accuracy, more efficient; possible Task 2 alternative | 🟢 Low |
| **Shape-constrained regularisation** — atlas-based prior or topology loss | Anatomy-aware design; reduce leakage at vascular junctions | 🟢 Low |

---

## Phase 8 — Challenge Submission Pipeline ⏳

- [ ] Implement `post_docker_build.py`: cache trained weights from cloud storage.
- [ ] Verify Docker build (`docker-test.yml`) passes with `status: pre`.
- [ ] Run full training (MRI: ≥ 400 epochs, CT: ≥ 400 epochs); save best checkpoints.
  - Development runs: 150/200 epochs; final submission runs: 400+ epochs with 5-fold CV.
- [ ] End-to-end inference smoke test inside Docker container.
- [ ] Set `status: alpha` in `docker-test.yml`; enable dataset download step.
- [ ] Submit to official evaluation platform.
- [ ] Set `status: final` for final submission.

---

## Immediate Next Steps

1. **`predict.py`**: implement MRI/CT inference entrypoints and task routing.
2. **MRI post-processing**: scar constrained by LA cavity; connected-component cleanup.
3. **CT post-processing**: largest connected component per class; optional morphological closing for LAA.
4. **`pipeline.py`**: end-to-end `run(record, task, model, output_dir)` wrapper.
5. **Validation pass**: run a small real-data train/val smoke test for both trainers.
6. **CLAHE ablation**: wire `utils/mclahe.py` into the MRI dataset and compare.
7. **TTA**: add inference-time flip/rotation averaging, especially for Task 2.

---

## Notes & Open Questions

- **Task 2 domain shift**: Center B/C validation data (20 samples) is available for fine-tuning / histogram matching — check challenge rules whether this is permitted.
- **Task 3 label format**: challenge page says `cardiacSegImgMO.nii.gz` but training data uses `label_XXXX.nii.gz` — must handle both names in `data_reader.py` when validation/test data is released.
- **G-DSC for Task 1**: the scar label is used as a _continuous_ weight map in G-DSC computation; model's soft output (pre-sigmoid) may score better than hard threshold.
- **Prizes**: only Tasks 1 and 3 are prize-eligible; prioritise these two.
