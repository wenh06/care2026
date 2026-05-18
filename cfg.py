"""
Configurations for models, training, etc.
"""

import pathlib
from copy import deepcopy

import numpy as np
import torch
from torch_ecg.cfg import CFG

from const import CT_NUM_CLASSES, CT_PATCH_SIZE, DEFAULT_VAL_RATIO, MRI_PATCH_SHAPE

__all__ = [
    "BaseCfg",
    "MRI_TrainCfg",
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
# MRI training configuration (Tasks 1 & 2 — dual-head V-Net)
# ---------------------------------------------------------------------------

MRI_TrainCfg = deepcopy(BaseCfg)

MRI_TrainCfg.task = "mri"

# Volume shape after crop + resize
MRI_TrainCfg.patch_shape = MRI_PATCH_SHAPE  # (H, W, D)

# Training duration and batch
MRI_TrainCfg.n_epochs = 150
MRI_TrainCfg.batch_size = 2  # limited by GPU memory (full 256×256×44 volumes)

# Optimizer
MRI_TrainCfg.optimizer = "adamw"
MRI_TrainCfg.betas = (0.9, 0.999)
MRI_TrainCfg.decay = 1e-2
MRI_TrainCfg.learning_rate = 3e-4
MRI_TrainCfg.lr = MRI_TrainCfg.learning_rate

# Cosine annealing schedule
MRI_TrainCfg.lr_scheduler = "cosine"
MRI_TrainCfg.lr_min = 1e-6

# Augmentation probability (applied per-sample in __getitem__)
MRI_TrainCfg.aug_prob = 0.5

# Multi-task loss weights:
#   la_dice       -- Dice loss on the LA cavity head (all 190 samples)
#   scar_dice     -- Dice loss on the scar head (Task 1 samples only)
#   scar_boundary -- boundary / surface loss on the scar prediction
#   scar_focal    -- focal loss on scar (class imbalance)
MRI_TrainCfg.loss_weights = CFG(
    la_dice=1.0,
    scar_dice=2.0,
    scar_boundary=0.0,   # disabled by default (costly scipy EDT); enable for fine-tuning
    scar_focal=0.5,
)

# Checkpointing
MRI_TrainCfg.keep_checkpoint_max = 5
MRI_TrainCfg.log_step = 10
MRI_TrainCfg.debug = False

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
CT_TrainCfg.cps_lambda_max = 1.0    # maximum CPS consistency weight
CT_TrainCfg.cps_rampup_epochs = 30  # ramp λ_cps linearly from 0 → cps_lambda_max

# Loss weights (supervised Dice + CE on labeled, CPS on all)
CT_TrainCfg.loss_weights = CFG(
    sup_dice=0.5,
    sup_ce=0.5,
    cps=1.0,
)

# Checkpointing
CT_TrainCfg.keep_checkpoint_max = 5
CT_TrainCfg.log_step = 10
CT_TrainCfg.debug = False

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

ModelCfg = CFG()

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
        la=CFG(out_channels=2),   # 2-class: background + LA cavity
        scar=CFG(out_channels=2), # 2-class: background + scar
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
