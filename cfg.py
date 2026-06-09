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
MRI_Stage1_TrainCfg.n_epochs = 100
MRI_Stage1_TrainCfg.batch_size = 4  # small input → larger batch fits fine
MRI_Stage1_TrainCfg.use_amp = True
MRI_Stage1_TrainCfg.accumulate_grad_batches = 1

# Optimizer
MRI_Stage1_TrainCfg.optimizer = "adamw"
MRI_Stage1_TrainCfg.betas = (0.9, 0.999)
MRI_Stage1_TrainCfg.decay = 1e-2
MRI_Stage1_TrainCfg.learning_rate = 3e-4
MRI_Stage1_TrainCfg.lr = MRI_Stage1_TrainCfg.learning_rate

# Cosine annealing schedule
MRI_Stage1_TrainCfg.lr_scheduler = "cosine"
MRI_Stage1_TrainCfg.lr_min = 1e-6

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
MRI_Stage1_TrainCfg.debug = False

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
MRI_Stage2_TrainCfg.backbone = "vnet_stage2"  # "vnet_stage2" | "nested_vnet_stage2"

# Volume shape: canonical → crop centred on LA centroid → resize to 128×128×44
MRI_Stage2_TrainCfg.canonical_shape = MRI_CANONICAL_SHAPE
MRI_Stage2_TrainCfg.cache_shape = MRI_STAGE2_CACHE_SHAPE
MRI_Stage2_TrainCfg.patch_shape = MRI_STAGE2_CROP_SHAPE
MRI_Stage2_TrainCfg.centroid_jitter = MRI_STAGE2_CENTROID_JITTER
MRI_Stage2_TrainCfg.train_crop_hw = 128  # model input at training time

MRI_Stage2_TrainCfg.n_epochs = 200
MRI_Stage2_TrainCfg.batch_size = 4
MRI_Stage2_TrainCfg.use_amp = True
MRI_Stage2_TrainCfg.accumulate_grad_batches = 1

MRI_Stage2_TrainCfg.optimizer = "adamw"
MRI_Stage2_TrainCfg.betas = (0.9, 0.999)
MRI_Stage2_TrainCfg.decay = 1e-2
MRI_Stage2_TrainCfg.learning_rate = 3e-4
MRI_Stage2_TrainCfg.lr = MRI_Stage2_TrainCfg.learning_rate
MRI_Stage2_TrainCfg.lr_scheduler = "cosine"
MRI_Stage2_TrainCfg.lr_min = 1e-6

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
MRI_Stage2_TrainCfg.debug = False
MRI_Stage2_TrainCfg.apply_mclahe = False

# ---------------------------------------------------------------------------
# CT training configuration (Task 3 — semi-supervised VNet)
# ---------------------------------------------------------------------------

CT_TrainCfg = deepcopy(BaseCfg)

CT_TrainCfg.task = "ct"
CT_TrainCfg.backbone = "vnet_ct"  # "vnet_ct" | "nested_vnet_ct"

# Patch size (isotropic cube) after resampling to 0.5 mm isotropic
CT_TrainCfg.patch_size = CT_PATCH_SIZE  # 128³

# Training duration and batch
CT_TrainCfg.n_epochs = 200
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
# "minmax"     : fixed HU clip then scale to [0,1]  (current default)
# "percentile" : per-volume percentile clip then scale to [0,1]  (nnUNet-style)
# "zscore"     : per-volume z-score  (μ=0, σ=1), no clipping
CT_TrainCfg.normalization = CFG(
    mode="minmax",
    hu_min=-200.0,
    hu_max=800.0,  # for "minmax"
    p_low=0.5,
    p_high=99.5,  # for "percentile"
)

# Augmentation (nnUNet-inspired, configurable per method)
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
CT_TrainCfg.semi_supervised_mode = "cps"
CT_TrainCfg.consistency_rampup_epochs = 30
# CPS
CT_TrainCfg.cps_lambda_max = 1.0
# Mean Teacher (Tarvainen & Valpola, NeurIPS 2017)
# Teacher EMA decay: θ_t ← α·θ_t + (1−α)·θ_s (α=0.99 per step)
CT_TrainCfg.mt_ema_decay = 0.99
CT_TrainCfg.mt_consistency_weight = 1.0  # λ_consist in L_total = L_sup + λ·L_consist

# Loss weights (supervised Dice + CE on labeled, CPS on all)
# Per-class CE weights: inverse-frequency to counteract LA dominance.
# Train-set fg ratios: LA≈75% PV≈6% LAA≈19% → weights ≈ 1/ratio, norm to LA=1
CT_TrainCfg.loss_weights = CFG(
    sup_dice=0.5,
    sup_ce=0.5,
    sup_boundary=0.0,  # set >0 to enable HausdorffERLoss (PV/LAA boundary precision)
    cps=1.0,
    ce_class_weight=[0.2, 1.0, 6.0, 2.0],  # [bg, LA, PV, LAA] — PV 6x higher
)

# Checkpointing
CT_TrainCfg.keep_checkpoint_max = 3
CT_TrainCfg.log_step = 10
CT_TrainCfg.debug = False

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
# Stage 1 provides the LA cavity mask and centroid; Stage 2 is a single-head
# VNet trained on the centroid-cropped region (128×128×44) to segment scar only.

ModelCfg.vnet_stage2 = CFG(
    in_channels=1,
    num_classes=2,  # binary: background + scar
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

# -- Nested V-Net (UNet++) for MRI Stage 2 (deep supervision variant) ---------

ModelCfg.nested_vnet_stage2 = CFG(
    in_channels=1,
    num_classes=2,  # binary: background + scar
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
    deep_supervision=True,
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
        dropout=[0.0, 0.0, 0.0, 0.2],
    ),
    up_conv=CFG(
        channels=[128, 64, 32, 16],
        kernel_size=[3, 3, 3, 3],
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
        dropout=[0.0, 0.0, 0.0, 0.2],
    ),
    up_conv=CFG(
        channels=[128, 64, 32, 16],
        kernel_size=[3, 3, 3, 3],
        dropout=[0.0, 0.0, 0.0, 0.0],
    ),
    output_conv=CFG(kernel_size=1),
    deep_supervision=True,
    bottleneck_transformer=None,
)
