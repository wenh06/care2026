"""
Configurations for models, training, etc.
"""

import pathlib
from copy import deepcopy

import numpy as np
import torch
from torch_ecg.cfg import CFG

from const import (
    CT_NUM_CLASSES,
    CT_PATCH_SIZE,
    DEFAULT_VAL_RATIO,
    MRI_CANONICAL_SHAPE,
    MRI_STAGE1_SHAPE,
    MRI_STAGE2_CACHE_SHAPE,
    MRI_STAGE2_CENTROID_JITTER,
    MRI_STAGE2_CROP_SHAPE,
)

__all__ = [
    "BaseCfg",
    "MRI_Stage1_TrainCfg",
    "MRI_Stage2_TrainCfg",
    "CT_TrainCfg",
    "CT_TrainCfgV2",
    "CT_TrainCfg_nnUNet",
    "CT_TrainCfg_MT_nnUNet",
    "NNUNET_CT_ARCH_CONFIG",
    "ModelCfg",
]

_BASE_DIR = pathlib.Path(__file__).absolute().parent

# ---------------------------------------------------------------------------
# Base config
# ---------------------------------------------------------------------------

BaseCfg = CFG()
BaseCfg.db_dir = None
BaseCfg.working_dir = None
BaseCfg.project_dir = _BASE_DIR
BaseCfg.log_dir = _BASE_DIR / "log"
BaseCfg.model_dir = _BASE_DIR / "checkpoints"
BaseCfg.results_dir = _BASE_DIR / "results"
BaseCfg.log_dir.mkdir(exist_ok=True)
BaseCfg.model_dir.mkdir(exist_ok=True)
BaseCfg.results_dir.mkdir(exist_ok=True)

BaseCfg.torch_dtype = torch.float32
BaseCfg.np_dtype = np.float32

BaseCfg.val_ratio = DEFAULT_VAL_RATIO
BaseCfg.random_seed = 42

# ---------------------------------------------------------------------------
# MRI Stage 1 training configuration (coarse LA localisation)
# ---------------------------------------------------------------------------
# Stage 1 model sees the full volume downsampled to MRI_STAGE1_SHAPE (144×144×44)
# and produces a binary LA segmentation used only to locate the LA centroid.

MRI_Stage1_TrainCfg = deepcopy(BaseCfg)

MRI_Stage1_TrainCfg.task = "mri"
MRI_Stage1_TrainCfg.stage = 1

# Volume shape after resampling to canonical then downsampling to Stage 1 resolution
MRI_Stage1_TrainCfg.canonical_shape = MRI_CANONICAL_SHAPE  # (576, 576, 44)
MRI_Stage1_TrainCfg.patch_shape = MRI_STAGE1_SHAPE  # (144, 144, 44)

# No HW sub-crop during training: Stage 1 input is already small
MRI_Stage1_TrainCfg.train_crop_hw = 0

# Training duration and batch
# MRI_Stage1_TrainCfg.n_epochs = 100
MRI_Stage1_TrainCfg.n_epochs = 300  # extended for SGD convergence
MRI_Stage1_TrainCfg.batch_size = 4  # small input → larger batch fits fine
MRI_Stage1_TrainCfg.use_amp = True
MRI_Stage1_TrainCfg.accumulate_grad_batches = 1

# SGD + poly LR (MBAS2024: all top teams use SGD)
# MRI_Stage1_TrainCfg.optimizer = "adamw"
# MRI_Stage1_TrainCfg.betas = (0.9, 0.999)
# MRI_Stage1_TrainCfg.decay = 1e-2
# MRI_Stage1_TrainCfg.learning_rate = 3e-4
# MRI_Stage1_TrainCfg.lr = MRI_Stage1_TrainCfg.learning_rate
# MRI_Stage1_TrainCfg.lr_scheduler = "cosine"
# MRI_Stage1_TrainCfg.lr_min = 1e-6
MRI_Stage1_TrainCfg.optimizer = "sgd"
MRI_Stage1_TrainCfg.momentum = 0.9
MRI_Stage1_TrainCfg.decay = 1e-4
MRI_Stage1_TrainCfg.learning_rate = 1e-2
MRI_Stage1_TrainCfg.lr = MRI_Stage1_TrainCfg.learning_rate
MRI_Stage1_TrainCfg.lr_scheduler = "poly"
MRI_Stage1_TrainCfg.lr_poly_power = 0.9

# Augmentation
MRI_Stage1_TrainCfg.augmentation = CFG(
    flips=CFG(prob=0.5),
    rotation=CFG(prob=0.5),
    gamma=CFG(prob=0.5, range=[0.7, 1.5]),
    gaussian_noise=CFG(prob=0.5, std_range=[0.0, 0.1]),
)

# Loss weights (Stage 1 only predicts binary LA → no scar head)
MRI_Stage1_TrainCfg.loss_weights = CFG(la_dice=1.0)

# Checkpointing
MRI_Stage1_TrainCfg.keep_checkpoint_max = 3
MRI_Stage1_TrainCfg.log_step = 10
MRI_Stage1_TrainCfg.debug = True  # evaluate on train set too (set False to skip)

# CLAHE preprocessing (disabled by default; enable for ablation)
MRI_Stage1_TrainCfg.apply_mclahe = False

# ---------------------------------------------------------------------------
# MRI Stage 2 training configuration (scar-only segmentation)
# ---------------------------------------------------------------------------
# Stage 1 provides the LA cavity mask; Stage 2 focuses entirely on scar
# using ScarLoss with Gaussian spatial weighting to handle extreme class
# imbalance (~2.4 % of LA voxels are scar).

MRI_Stage2_TrainCfg = deepcopy(BaseCfg)

MRI_Stage2_TrainCfg.task = "mri"
MRI_Stage2_TrainCfg.stage = 2
# MRI_Stage2_TrainCfg.backbone = "vnet_stage2"
MRI_Stage2_TrainCfg.backbone = "vnet_stage2_2ch"  # "vnet_stage2" | "nested_vnet_stage2" | "vnet_stage2_2ch"

# Volume shape: canonical → crop centred on LA centroid → resize to 128×128×44
MRI_Stage2_TrainCfg.canonical_shape = MRI_CANONICAL_SHAPE
MRI_Stage2_TrainCfg.cache_shape = MRI_STAGE2_CACHE_SHAPE
MRI_Stage2_TrainCfg.patch_shape = MRI_STAGE2_CROP_SHAPE
MRI_Stage2_TrainCfg.centroid_jitter = MRI_STAGE2_CENTROID_JITTER
MRI_Stage2_TrainCfg.train_crop_hw = 128  # model input at training time

# MRI_Stage2_TrainCfg.n_epochs = 200
# MRI_Stage2_TrainCfg.n_epochs = 400
MRI_Stage2_TrainCfg.n_epochs = 600  # extended for SGD convergence
MRI_Stage2_TrainCfg.batch_size = 4
MRI_Stage2_TrainCfg.use_amp = True
MRI_Stage2_TrainCfg.accumulate_grad_batches = 1

# SGD + poly LR (MBAS2024: all top teams use SGD)
# MRI_Stage2_TrainCfg.optimizer = "adamw"
# MRI_Stage2_TrainCfg.betas = (0.9, 0.999)
# MRI_Stage2_TrainCfg.decay = 1e-2
# MRI_Stage2_TrainCfg.learning_rate = 3e-4
# MRI_Stage2_TrainCfg.lr = MRI_Stage2_TrainCfg.learning_rate
# MRI_Stage2_TrainCfg.lr_scheduler = "cosine"
# MRI_Stage2_TrainCfg.lr_min = 1e-6
MRI_Stage2_TrainCfg.optimizer = "sgd"
MRI_Stage2_TrainCfg.momentum = 0.9
MRI_Stage2_TrainCfg.decay = 1e-4
MRI_Stage2_TrainCfg.learning_rate = 1e-2
MRI_Stage2_TrainCfg.lr = MRI_Stage2_TrainCfg.learning_rate
MRI_Stage2_TrainCfg.lr_scheduler = "poly"
MRI_Stage2_TrainCfg.lr_poly_power = 0.9

# Aggressive augmentation for domain generalisation (Task 2 unseen centres)
MRI_Stage2_TrainCfg.augmentation = CFG(
    flips=CFG(prob=0.5),
    rotation=CFG(prob=0.5),
    gamma=CFG(prob=0.5, range=[0.7, 1.5]),
    gaussian_noise=CFG(prob=0.5, std_range=[0.0, 0.1]),
    brightness_contrast=CFG(prob=0.5, contrast_range=[0.85, 1.15], brightness_range=[-0.1, 0.1]),
    gaussian_blur=CFG(prob=0.2, sigma_range=[0.5, 1.0]),
    elastic_deformation=CFG(prob=0.2, alpha_range=[0, 200], sigma_range=[9, 13]),
    low_resolution=CFG(prob=0.2, zoom_range=[0.5, 1.0]),
)

# Scar-only loss weights (no LA head; LA from Stage 1)
MRI_Stage2_TrainCfg.loss_weights = CFG(
    scar_dice=1.0,
    scar_focal=0.5,
    scar_boundary=0.0,  # set >0 to enable BoundaryLoss (thin wall precision)
    spatial_w0=5.0,
    spatial_sigma_mm=2.0,
)

MRI_Stage2_TrainCfg.no_scar_proportion = 0.3  # fraction of no-scar (Task 2) records to include as hard negatives
MRI_Stage2_TrainCfg.keep_checkpoint_max = 3
MRI_Stage2_TrainCfg.log_step = 10
MRI_Stage2_TrainCfg.debug = True  # evaluate on train set too (set False to skip)
MRI_Stage2_TrainCfg.apply_mclahe = False

# ---------------------------------------------------------------------------
# CT training configuration (Task 3 — semi-supervised VNet)
# ---------------------------------------------------------------------------

CT_TrainCfg = deepcopy(BaseCfg)

CT_TrainCfg.task = "ct"
CT_TrainCfg.backbone = "vnet_ct"  # "vnet_ct" | "nested_vnet_ct"

# Patch size (isotropic cube) after resampling to 0.5 mm isotropic
CT_TrainCfg.patch_size = CT_PATCH_SIZE  # 128³ (default; try 160 for more context)
CT_TrainCfg.fg_bias = 0.85  # 0.5  # foreground-biased patch sampling (try 0.85 for PV/LAA)
# Class-aware patch sampling: per-class probabilities for patch centre.
# None → use fg_bias + random sampling.  List of 4 floats [random, LA, PV, LAA]
# summing to 1.0 enables explicit per-class sampling, guaranteeing PV/LAA exposure.
CT_TrainCfg.class_sampling_probs = [0.15, 0.30, 0.35, 0.20]  # [random, LA, PV, LAA]
# CT_TrainCfg.pretrained_encoder = "checkpoints/vnet_ct_nnunet_enc.safetensors"
CT_TrainCfg.pretrained_encoder = "checkpoints/ct_basemodel.safetensors"  # best supervised ckpt

# Training duration and batch
# CT_TrainCfg.n_epochs = 200
CT_TrainCfg.n_epochs = 1000
CT_TrainCfg.batch_size = 2
CT_TrainCfg.use_amp = True
CT_TrainCfg.accumulate_grad_batches = 1

# Optimizer (SGD + polynomial LR, standard for semi-supervised 3-D seg)
CT_TrainCfg.optimizer = "sgd"
CT_TrainCfg.momentum = 0.9
CT_TrainCfg.decay = 1e-4
CT_TrainCfg.learning_rate = 1e-2
CT_TrainCfg.lr = CT_TrainCfg.learning_rate

# Polynomial LR decay (power = 0.9)
CT_TrainCfg.lr_scheduler = "poly"
CT_TrainCfg.lr_poly_power = 0.9

# CT intensity normalisation — configurable mode
# "minmax"     : fixed HU clip then scale to [0,1]
# "percentile" : per-volume percentile clip then scale to [0,1]  (nnUNet-style, default)
# "zscore"     : per-volume z-score  (μ=0, σ=1), no clipping
CT_TrainCfg.normalization = CFG(
    mode="percentile",
    hu_min=-200.0,
    hu_max=800.0,  # for "minmax"
    p_low=0.5,
    p_high=99.5,  # for "percentile"
)

# Augmentation — restored to best-run config (0.5234, epoch 265).
# Note: the actual augmentations applied during that run were controlled
# by the hardcoded `aug_prob=0.5` system (flips + rotation + intensity
# scaling + Gaussian noise); the per-method config below reflects
# CT_TrainCfg as saved in the checkpoint.
CT_TrainCfg.augmentation = CFG(
    flips=CFG(prob=0.5),
    rotation=CFG(prob=0.5),
    gamma=CFG(prob=0.5, range=[0.7, 1.5]),
    gaussian_noise=CFG(prob=0.5, std_range=[0.0, 0.05]),
    brightness_contrast=CFG(prob=0.5, contrast_range=[0.85, 1.15], brightness_range=[-0.1, 0.1]),
    gaussian_blur=CFG(prob=0.5, sigma_range=[0.5, 1.5]),
    elastic_deformation=CFG(prob=0.2, alpha_range=[0, 200], sigma_range=[9, 13]),
    low_resolution=CFG(prob=0.2, zoom_range=[0.5, 1.0]),
)

# Semi-supervised mode: "cps" or "mean_teacher"
# CT_TrainCfg.semi_supervised_mode = "supervised"
CT_TrainCfg.semi_supervised_mode = "mean_teacher"
# Warm-up: run pure supervised first, then ramp up MT consistency
CT_TrainCfg.mt_warmup_epochs = 400  # 0 = no warmup
CT_TrainCfg.mt_rampup_epochs = 100  # ramp-up duration *after* warmup
# Legacy key (used by trainer._get_cps_weight):
CT_TrainCfg.consistency_rampup_epochs = CT_TrainCfg.mt_rampup_epochs
# CPS
CT_TrainCfg.cps_lambda_max = 1.0
# Mean Teacher (Tarvainen & Valpola, NeurIPS 2017)
# Teacher EMA decay: θ_t ← α·θ_t + (1−α)·θ_s (α=0.99 per step)
CT_TrainCfg.mt_ema_decay = 0.99
CT_TrainCfg.mt_consistency_weight = 1.0  # λ_consist in L_total = L_sup + λ·L_consist

# Loss weights (FocalTversky + optional CE + optional boundary loss).
# FocalTversky α=0.7, β=0.3 penalises false positives more than false
# negatives, counteracting the model's tendency to massively over-predict
# foreground (LA 2–3×, LAA 3–4× GT).
# Per-class CE weights: inverse-frequency to counteract LA dominance.
# Train-set fg ratios: LA≈75% PV≈6% LAA≈19% → weights ≈ 1/ratio, norm to LA=1
CT_TrainCfg.loss_weights = CFG(
    tversky_alpha=0.7,
    tversky_beta=0.3,
    tversky_gamma=0.75,
    sup_ce=0.0,  # set >0 to add CE with class weights
    sup_boundary=0.1,  # set >0 to enable HausdorffERLoss (PV/LAA boundary precision)
    sup_clce=0.2,  # set >0 to enable CenterlineCELoss (PV topology preservation)
    clce_start_epoch=100,  # delay clCE until this epoch (0 = from start; try 100)
    clce_classes=[2],  # None = all classes; e.g. [2] for PV-only topology supervision
    cps=1.0,
    ce_class_weight=[0.2, 1.0, 6.0, 2.0],  # [bg, LA, PV, LAA]
)

# Checkpointing
CT_TrainCfg.keep_checkpoint_max = 3
CT_TrainCfg.log_step = 10
CT_TrainCfg.debug = True  # evaluate on train set too (set False to skip)

# ---------------------------------------------------------------------------
# CT V2 training configuration (supervised-only, InstanceNorm + Mish + AdamW)
# ---------------------------------------------------------------------------
# Fixes the three root causes of poor CT performance:
#   1. BatchNorm → InstanceNorm  (batch_size=2 friendly)
#   2. ReLU → Mish              (better gradient flow for segmentation)
#   3. SGD → AdamW              (adaptive LR handles small-batch noise)
#
# Semi-supervised (CPS / Mean Teacher) is removed — the consistency signal
# was 1000× weaker than the supervised signal (0.0003 vs 0.4), meaning
# unlabelled data contributed virtually nothing useful.  A strong supervised
# baseline on the 50 labelled CTs is the right first step.

CT_TrainCfgV2 = deepcopy(BaseCfg)

CT_TrainCfgV2.task = "ct"
CT_TrainCfgV2.backbone = "vnet_ct_v2"

# Patch size (isotropic cube) after resampling to 0.5 mm isotropic
CT_TrainCfgV2.patch_size = CT_PATCH_SIZE  # 128³

# Training duration and batch
CT_TrainCfgV2.n_epochs = 300
CT_TrainCfgV2.batch_size = 2
CT_TrainCfgV2.use_amp = True
CT_TrainCfgV2.accumulate_grad_batches = 1

# ── Optimizer: AdamW (matching MRI models) ──────────────────────────────
CT_TrainCfgV2.optimizer = "adamw"
CT_TrainCfgV2.betas = (0.9, 0.999)
CT_TrainCfgV2.decay = 1e-2
CT_TrainCfgV2.learning_rate = 3e-4
CT_TrainCfgV2.lr = CT_TrainCfgV2.learning_rate

# Cosine annealing schedule (matching MRI models)
CT_TrainCfgV2.lr_scheduler = "cosine"
CT_TrainCfgV2.lr_min = 1e-6

# ── CT intensity normalisation ───────────────────────────────────────────
CT_TrainCfgV2.normalization = CFG(
    mode="minmax",
    hu_min=-200.0,
    hu_max=800.0,
    p_low=0.5,
    p_high=99.5,
)

# ── Augmentation — aligned with CT_TrainCfg (4-method, matching old aug) ─
CT_TrainCfgV2.augmentation = CFG(
    flips=CFG(prob=0.5),
    rotation=CFG(prob=0.5),
    gamma=CFG(prob=0.5, range=[0.7, 1.5]),
    gaussian_noise=CFG(prob=0.5, std_range=[0.0, 0.05]),
)

# ── Semi-supervised: off  ────────────────────────────────────────────────
# "supervised" mode uses only labelled data.  CPS / mean_teacher are
# available if needed later, once the supervised baseline is solid.
CT_TrainCfgV2.semi_supervised_mode = "supervised"
CT_TrainCfgV2.consistency_rampup_epochs = 30
CT_TrainCfgV2.cps_lambda_max = 1.0
CT_TrainCfgV2.mt_ema_decay = 0.99
CT_TrainCfgV2.mt_consistency_weight = 1.0

# ── Loss weights ─────────────────────────────────────────────────────────
# Class weights: PV weight reduced from 6.0 → 3.0.  The old 6× weight
# caused the model to massively over-predict PV (2.23× more voxels
# than GT, precision 0.255).
CT_TrainCfgV2.loss_weights = CFG(
    sup_dice=0.5,
    sup_ce=0.5,
    sup_boundary=0.0,
    ce_class_weight=[0.1, 1.0, 3.0, 2.0],  # [bg, LA, PV, LAA]
)

# Checkpointing
CT_TrainCfgV2.keep_checkpoint_max = 3
CT_TrainCfgV2.log_step = 10
CT_TrainCfgV2.debug = True  # evaluate on train set too (set False to skip)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

ModelCfg = CFG()

# -- Single-head V-Net for MRI Stage 1 (coarse LA localisation) -------------

ModelCfg.vnet_stage1 = CFG(
    in_channels=1,
    num_classes=2,  # binary: background + LA cavity
    norm="instance",
    activation="mish",
    use_eca_skip=False,
    input_conv=CFG(channels=16, kernel_size=5),
    down_conv=CFG(
        channels=[32, 64, 128, 256],
        kernel_size=[3, 3, 3, 3],
        dropout=[0.0, 0.0, 0.3, 0.3],
    ),
    up_conv=CFG(
        channels=[128, 64, 32, 16],
        kernel_size=[3, 3, 3, 3],
        dropout=[0.0, 0.0, 0.0, 0.0],
    ),
    output_conv=CFG(kernel_size=1),
    bottleneck_transformer=None,
)

# -- Single-head V-Net for MRI Stage 2 (scar-only, cropped region) ------------

ModelCfg.vnet_stage2 = CFG(
    in_channels=1,
    num_classes=2,
    norm="instance",
    activation="mish",
    use_eca_skip=False,
    input_conv=CFG(channels=16, kernel_size=5),
    down_conv=CFG(
        channels=[32, 64, 128, 256],
        kernel_size=[3, 3, 3, 3],
        dropout=[0.0, 0.0, 0.3, 0.3],
    ),
    up_conv=CFG(
        channels=[128, 64, 32, 16],
        kernel_size=[3, 3, 3, 3],
        dropout=[0.0, 0.0, 0.0, 0.0],
    ),
    output_conv=CFG(kernel_size=1),
    bottleneck_transformer=None,
)

# -- Large V-Net for MRI Stage 2 (wider + deeper) -------------------------------

ModelCfg.vnet_stage2_l = CFG(
    in_channels=1,
    num_classes=2,
    norm="instance",
    activation="mish",
    use_eca_skip=False,
    input_conv=CFG(channels=16, kernel_size=5),
    down_conv=CFG(
        channels=[32, 64, 128, 256],
        kernel_size=[3, 3, 3, 3],
        blocks=[1, 2, 2, 2],
        dropout=[0.0, 0.0, 0.3, 0.3],
    ),
    up_conv=CFG(
        channels=[256, 128, 64, 32],
        kernel_size=[3, 3, 3, 3],
        blocks=[2, 2, 1, 1],
        dropout=[0.0, 0.0, 0.0, 0.0],
    ),
    output_conv=CFG(kernel_size=1),
    bottleneck_transformer=None,
)

# -- VNet for MRI Stage 2 with anatomical prior channel (distance transform) -----

ModelCfg.vnet_stage2_2ch = CFG(
    in_channels=2,
    num_classes=2,
    norm="instance",
    activation="mish",
    use_eca_skip=False,
    input_conv=CFG(channels=16, kernel_size=5),
    down_conv=CFG(
        channels=[32, 64, 128, 256],
        kernel_size=[3, 3, 3, 3],
        dropout=[0.0, 0.0, 0.3, 0.3],
    ),
    up_conv=CFG(
        channels=[128, 64, 32, 16],
        kernel_size=[3, 3, 3, 3],
        dropout=[0.0, 0.0, 0.0, 0.0],
    ),
    output_conv=CFG(kernel_size=1),
    bottleneck_transformer=None,
)

# -- Large VNet 2ch -------------------------------------------------------------

ModelCfg.vnet_stage2_2ch_l = CFG(
    in_channels=2,
    num_classes=2,
    norm="instance",
    activation="mish",
    use_eca_skip=False,
    input_conv=CFG(channels=16, kernel_size=5),
    down_conv=CFG(
        channels=[32, 64, 128, 256],
        kernel_size=[3, 3, 3, 3],
        blocks=[1, 2, 2, 2],
        dropout=[0.0, 0.0, 0.3, 0.3],
    ),
    up_conv=CFG(
        channels=[256, 128, 64, 32],
        kernel_size=[3, 3, 3, 3],
        blocks=[2, 2, 1, 1],
        dropout=[0.0, 0.0, 0.0, 0.0],
    ),
    output_conv=CFG(kernel_size=1),
    bottleneck_transformer=None,
)

# -- Nested V-Net (UNet++) for MRI Stage 2 (deep supervision variant) ---------

ModelCfg.nested_vnet_stage2 = CFG(
    in_channels=1,
    num_classes=2,
    norm="instance",
    activation="mish",
    use_eca_skip=False,
    input_conv=CFG(channels=16, kernel_size=5),
    down_conv=CFG(
        channels=[32, 64, 128, 256],
        kernel_size=[3, 3, 3, 3],
        dropout=[0.0, 0.0, 0.3, 0.3],
    ),
    up_conv=CFG(
        channels=[128, 64, 32, 16],
        kernel_size=[3, 3, 3, 3],
        dropout=[0.0, 0.0, 0.0, 0.0],
    ),
    output_conv=CFG(kernel_size=1),
    deep_supervision=False,
    bottleneck_transformer=None,
)

# -- Large Nested V-Net (UNet++) for MRI Stage 2 (wider + deeper) --------------

ModelCfg.nested_vnet_stage2_l = CFG(
    in_channels=1,
    num_classes=2,
    norm="instance",
    activation="mish",
    use_eca_skip=False,
    input_conv=CFG(channels=16, kernel_size=5),
    down_conv=CFG(
        channels=[32, 64, 128, 256],
        kernel_size=[3, 3, 3, 3],
        blocks=[1, 2, 2, 2],
        dropout=[0.0, 0.0, 0.3, 0.3],
    ),
    up_conv=CFG(
        channels=[256, 128, 64, 32],
        kernel_size=[3, 3, 3, 3],
        blocks=[2, 2, 1, 1],
        dropout=[0.3, 0.3, 0.0, 0.0],
    ),
    output_conv=CFG(kernel_size=1),
    deep_supervision=False,
    bottleneck_transformer=None,
)

# -- VNet (single-head) for CT CPS (Task 3) ----------------------------------

ModelCfg.vnet_ct = CFG(
    in_channels=1,
    num_classes=CT_NUM_CLASSES,  # 4
    norm="batch",
    activation="relu",
    use_eca_skip=False,
    input_conv=CFG(channels=16, kernel_size=3),
    down_conv=CFG(
        channels=[32, 64, 128, 256],
        kernel_size=[3, 3, 3, 3],
        blocks=[1, 2, 2, 2],
        dropout=[0.0, 0.0, 0.0, 0.2],
    ),
    up_conv=CFG(
        channels=[256, 128, 64, 32],
        kernel_size=[3, 3, 3, 3],
        blocks=[2, 2, 1, 1],
        dropout=[0.0, 0.0, 0.0, 0.0],
    ),
    output_conv=CFG(kernel_size=1),
    bottleneck_transformer=None,
)

# -- Nested V-Net (UNet++) for CT (deep supervision variant) --------------------

ModelCfg.nested_vnet_ct = CFG(
    in_channels=1,
    num_classes=CT_NUM_CLASSES,  # 4
    norm="batch",
    activation="relu",
    use_eca_skip=False,
    input_conv=CFG(channels=16, kernel_size=3),
    down_conv=CFG(
        channels=[32, 64, 128, 256],
        kernel_size=[3, 3, 3, 3],
        blocks=[1, 2, 2, 2],
        dropout=[0.0, 0.0, 0.0, 0.2],
    ),
    up_conv=CFG(
        channels=[256, 128, 64, 32],
        kernel_size=[3, 3, 3, 3],
        blocks=[2, 2, 1, 1],
        dropout=[0.0, 0.0, 0.0, 0.0],
    ),
    output_conv=CFG(kernel_size=1),
    deep_supervision=False,
    bottleneck_transformer=None,
)

# -- VNet for CT V2 (supervised-only, InstanceNorm + Mish) ---------------------
# Same encoder-decoder depth as vnet_ct but with:
#   - InstanceNorm (batch-size independent, matches MRI models)
#   - Mish activation (smoother gradients than ReLU)
#   - Slightly heavier dropout for regularisation on 50 labelled volumes

ModelCfg.vnet_ct_v2 = CFG(
    in_channels=1,
    num_classes=CT_NUM_CLASSES,  # 4: BG, LA, PV, LAA
    norm="instance",
    activation="mish",
    use_eca_skip=False,
    input_conv=CFG(channels=16, kernel_size=3),
    down_conv=CFG(
        channels=[32, 64, 128, 256],
        kernel_size=[3, 3, 3, 3],
        dropout=[0.0, 0.0, 0.3, 0.3],  # slightly heavier than v1 (0.0→0.3 at deeper levels)
    ),
    up_conv=CFG(
        channels=[128, 64, 32, 16],
        kernel_size=[3, 3, 3, 3],
        dropout=[0.0, 0.0, 0.0, 0.0],
    ),
    output_conv=CFG(kernel_size=1),
    bottleneck_transformer=None,
)

# ---------------------------------------------------------------------------
# nnUNet CT architecture config — from nnUNetv2_plan_and_preprocess
# Dataset500_CARE2026CT 3d_fullres.  6 encoder stages (vs. VNet's 4),
# last stride [1,1,2] preserves XY, deep supervision always on.
# ---------------------------------------------------------------------------

NNUNET_CT_ARCH_CONFIG = dict(
    n_stages=6,
    features_per_stage=[32, 64, 128, 256, 320, 320],
    kernel_sizes=[[3, 3, 3]] * 6,
    strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 1, 2]],
    n_conv_per_stage=[2, 2, 2, 2, 2, 2],
    n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
    conv_bias=True,
    norm_op_kwargs=dict(eps=1e-5, affine=True),
    nonlin_kwargs=dict(inplace=True),
)

# ---------------------------------------------------------------------------
# CT nnUNet training / inference configuration
# ---------------------------------------------------------------------------
# Wraps nnUNet's PlainConvUNet (ResEnc U-Net) as the backbone.
# Designed for inference with pretrained nnUNet weights from
# nnUNetv2_train.  Uses non-isotropic patch [112,112,192] and
# CTNormalization (percentile clip → z-score).

CT_TrainCfg_nnUNet = deepcopy(BaseCfg)

CT_TrainCfg_nnUNet.task = "ct"
CT_TrainCfg_nnUNet.backbone = "nnunet"

# Non-isotropic patch matching nnUNet plan (more Z context for PV/LAA)
CT_TrainCfg_nnUNet.patch_shape = [112, 112, 192]

# Training duration and batch (matching nnUNet plan)
CT_TrainCfg_nnUNet.n_epochs = 1000
CT_TrainCfg_nnUNet.batch_size = 2
CT_TrainCfg_nnUNet.use_amp = True
CT_TrainCfg_nnUNet.accumulate_grad_batches = 1

# SGD + poly LR (matching nnUNet recipe)
CT_TrainCfg_nnUNet.optimizer = "sgd"
CT_TrainCfg_nnUNet.momentum = 0.99
CT_TrainCfg_nnUNet.decay = 3e-5
CT_TrainCfg_nnUNet.learning_rate = 1e-2
CT_TrainCfg_nnUNet.lr = CT_TrainCfg_nnUNet.learning_rate
CT_TrainCfg_nnUNet.lr_scheduler = "poly"
CT_TrainCfg_nnUNet.lr_poly_power = 0.9

# nnUNet CTNormalization: per-volume percentile clip → global z-score
CT_TrainCfg_nnUNet.normalization = CFG(
    mode="nnunet",
    p_low=0.5,
    p_high=99.5,
    # Global foreground stats from plan.json (fallback if per-volume fails)
    global_clip_min=1122.0,
    global_clip_max=2018.0,
    global_mean=1542.61,
    global_std=188.64,
)

# Architecture config — stored in model_config metadata so
# from_checkpoint can reconstruct PlainConvUNet.
CT_TrainCfg_nnUNet.arch_config = NNUNET_CT_ARCH_CONFIG

# Augmentation (nnUNet-style, extensive)
CT_TrainCfg_nnUNet.augmentation = CFG(
    flips=CFG(prob=0.5),
    rotation=CFG(prob=0.5),
    gamma=CFG(prob=0.5, range=[0.7, 1.5]),
    gaussian_noise=CFG(prob=0.5, std_range=[0.0, 0.05]),
    brightness_contrast=CFG(prob=0.5, contrast_range=[0.85, 1.15], brightness_range=[-0.1, 0.1]),
    gaussian_blur=CFG(prob=0.2, sigma_range=[0.5, 1.0]),
    elastic_deformation=CFG(prob=0.2, alpha_range=[0, 200], sigma_range=[9, 13]),
    low_resolution=CFG(prob=0.2, zoom_range=[0.5, 1.0]),
)

# Loss — Dice + CE matching nnUNet
CT_TrainCfg_nnUNet.loss_weights = CFG(
    sup_dice=0.5,
    sup_ce=0.5,
    ce_class_weight=[0.1, 1.0, 6.0, 2.0],  # [bg, LA, PV, LAA]
)

# Supervised-only (nnUNet doesn't use semi-supervised)
CT_TrainCfg_nnUNet.semi_supervised_mode = "supervised"

# Checkpointing
CT_TrainCfg_nnUNet.keep_checkpoint_max = 3
CT_TrainCfg_nnUNet.log_step = 10
CT_TrainCfg_nnUNet.debug = True

# Class-aware patch sampling
CT_TrainCfg_nnUNet.fg_bias = 0.85
CT_TrainCfg_nnUNet.class_sampling_probs = [0.15, 0.30, 0.35, 0.20]

# nnUNet target spacing (near-isotropic, Z preserved)
CT_TrainCfg_nnUNet.target_spacing = [0.5, 0.496, 0.496]

# nnUNet results directory — auto-discovers trainer/folds/checkpoint
CT_TrainCfg_nnUNet.nnunet_model_dir = "checkpoints/ct_model"
CT_TrainCfg_nnUNet.nnunet_folds = None  # None = auto-detect
CT_TrainCfg_nnUNet.nnunet_checkpoint = None  # None = auto-detect

# ---------------------------------------------------------------------------
# CT Mean Teacher with nnUNet backbone (PlainConvUNet)
# ---------------------------------------------------------------------------
# Uses nnUNet's 6-stage PlainConvUNet as student/teacher, initialized
# from a pretrained nnUNet checkpoint.  Mean Teacher consistency loss
# on 100 unlabeled CTs; warmup first N epochs with supervised-only.

CT_TrainCfg_MT_nnUNet = deepcopy(CT_TrainCfg_nnUNet)

# Override: semi-supervised mode
CT_TrainCfg_MT_nnUNet.semi_supervised_mode = "mean_teacher"
CT_TrainCfg_MT_nnUNet.mt_warmup_epochs = 200  # supervised warmup
CT_TrainCfg_MT_nnUNet.mt_rampup_epochs = 100  # ramp-up duration
CT_TrainCfg_MT_nnUNet.mt_ema_decay = 0.99
CT_TrainCfg_MT_nnUNet.mt_consistency_weight = 1.0  # λ in L = L_sup + λ*L_consist

# Pretrained nnUNet weights for the student (same format as nnUNet checkpoint .pth)
CT_TrainCfg_MT_nnUNet.pretrained_encoder = None  # e.g. "tmp/nnUNet_results/.../fold_0/checkpoint_best.pth"

# Loss: Dice+CE matching nnUNet, deep supervision per-level, MSE consistency
CT_TrainCfg_MT_nnUNet.loss_weights = CFG(
    loss_mode="dice_ce",  # nnUNet-style DiceCELoss (vs "focal_tversky" for VNet)
    sup_dice=1.0,  # Dice weight in DiceCELoss
    sup_ce=1.0,  # CE weight in DiceCELoss
    consist=1.0,  # MT consistency weight
    deep_supervision=True,  # per-level loss (nnUNet style)
    ce_class_weight=[0.1, 1.0, 6.0, 2.0],
)
