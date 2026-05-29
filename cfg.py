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
    "MRI_TrainCfg",  # alias for MRI_Stage2_TrainCfg
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
MRI_Stage1_TrainCfg.aug_prob = 0.5

# Loss weights (Stage 1 only predicts binary LA → no scar head)
MRI_Stage1_TrainCfg.loss_weights = CFG(la_dice=1.0)

# Checkpointing
MRI_Stage1_TrainCfg.keep_checkpoint_max = 3
MRI_Stage1_TrainCfg.log_step = 10
MRI_Stage1_TrainCfg.debug = False

# ---------------------------------------------------------------------------
# MRI Stage 2 training configuration (fine LA + scar segmentation)
# ---------------------------------------------------------------------------
# Stage 2 model operates on a fixed-size crop centred on the GT LA centroid
# (+ random jitter during training to match Stage 1 prediction uncertainty).

MRI_Stage2_TrainCfg = deepcopy(BaseCfg)

MRI_Stage2_TrainCfg.task = "mri"
MRI_Stage2_TrainCfg.stage = 2

# Volume shape: canonical → crop centred on LA centroid
MRI_Stage2_TrainCfg.canonical_shape = MRI_CANONICAL_SHAPE  # (576, 576, 44)
MRI_Stage2_TrainCfg.cache_shape = MRI_STAGE2_CACHE_SHAPE  # (320, 320, 44) generous cache
MRI_Stage2_TrainCfg.patch_shape = MRI_STAGE2_CROP_SHAPE  # (256, 256, 44) model input
MRI_Stage2_TrainCfg.centroid_jitter = MRI_STAGE2_CENTROID_JITTER  # (32, 32, 0)

# Training patch: optional further HW sub-crop to reduce GPU memory
MRI_Stage2_TrainCfg.train_crop_hw = 128  # 128×128×44 per sample with AMP on 16 GB GPU

# Training duration and batch
MRI_Stage2_TrainCfg.n_epochs = 150
MRI_Stage2_TrainCfg.batch_size = 1
MRI_Stage2_TrainCfg.use_amp = True
MRI_Stage2_TrainCfg.accumulate_grad_batches = 2  # effective batch = 2

# Optimizer
MRI_Stage2_TrainCfg.optimizer = "adamw"
MRI_Stage2_TrainCfg.betas = (0.9, 0.999)
MRI_Stage2_TrainCfg.decay = 1e-2
MRI_Stage2_TrainCfg.learning_rate = 3e-4
MRI_Stage2_TrainCfg.lr = MRI_Stage2_TrainCfg.learning_rate

# Cosine annealing schedule
MRI_Stage2_TrainCfg.lr_scheduler = "cosine"
MRI_Stage2_TrainCfg.lr_min = 1e-6

# Augmentation
MRI_Stage2_TrainCfg.aug_prob = 0.5

# Multi-task loss weights (same as before)
MRI_Stage2_TrainCfg.loss_weights = CFG(
    la_dice=1.0,
    scar_dice=2.0,
    scar_boundary=0.0,
    scar_focal=0.5,
)

# Checkpointing
MRI_Stage2_TrainCfg.keep_checkpoint_max = 3
MRI_Stage2_TrainCfg.log_step = 10
MRI_Stage2_TrainCfg.debug = False

# Backward-compatibility alias
MRI_TrainCfg = MRI_Stage2_TrainCfg

# ---------------------------------------------------------------------------
# CT training configuration (Task 3 — CPS semi-supervised UNet3D)
# ---------------------------------------------------------------------------

CT_TrainCfg = deepcopy(BaseCfg)

CT_TrainCfg.task = "ct"

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

# Augmentation
CT_TrainCfg.aug_prob = 0.5

# Cross Pseudo Supervision (CPS) settings
CT_TrainCfg.cps_lambda_max = 1.0  # maximum CPS consistency weight
CT_TrainCfg.cps_rampup_epochs = 30  # ramp λ_cps linearly from 0 → cps_lambda_max

# Loss weights (supervised Dice + CE on labeled, CPS on all)
CT_TrainCfg.loss_weights = CFG(
    sup_dice=0.5,
    sup_ce=0.5,
    cps=1.0,
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
)

# -- Dual-head V-Net for MRI (Tasks 1 & 2) ----------------------------------

ModelCfg.vnet = CFG(
    in_channels=1,
    # Instance Norm for domain generalisation (Task 2 test set has new centres)
    norm="instance",
    activation="mish",
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
    # Two output heads: LA cavity (binary) and LA scar (binary)
    heads=CFG(
        la=CFG(out_channels=2),  # 2-class: background + LA cavity
        scar=CFG(out_channels=2),  # 2-class: background + scar
    ),
)

# -- NestedV-Net (deep supervision) for MRI ---------------------------------

ModelCfg.nested_vnet = CFG(
    in_channels=1,
    norm="instance",
    activation="mish",
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
    heads=CFG(
        la=CFG(out_channels=2),
        scar=CFG(out_channels=2),
    ),
)

# -- VNet (single-head) for CT CPS (Task 3) ----------------------------------

ModelCfg.vnet_ct = CFG(
    in_channels=1,
    num_classes=CT_NUM_CLASSES,  # 4
    norm="batch",
    activation="relu",
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
)
