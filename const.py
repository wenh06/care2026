"""
Constant definitions for the CARE 2026 Left Atrium challenge.

Includes cache directories, spatial resolution constants, patch sizes,
HU windowing parameters, class maps, dataset size constants, and more.
"""

import os
from pathlib import Path

__all__ = [
    "MODEL_CACHE_DIR",
    "DATA_CACHE_DIR",
    "MRI_CENTER_A_SPACING",
    "CT_NATIVE_Z_SPACING",
    "CT_TARGET_SPACING",
    "MRI_PATCH_SHAPE",
    "CT_PATCH_SIZE",
    "CT_HU_MIN",
    "CT_HU_MAX",
    "TASK1_CLASS_MAP",
    "TASK2_CLASS_MAP",
    "CT_CLASS_MAP",
    "CT_NUM_CLASSES",
    "TASK1_TRAIN_COUNT",
    "TASK2_TRAIN_COUNT",
    "CT_TOTAL_COUNT",
    "CT_LABELED_COUNT",
    "CT_UNLABELED_COUNT",
    "DEFAULT_VAL_RATIO",
    "REMOTE_MODELS",
]

# ---------------------------------------------------------------------------
# Cache directories (override via environment variables)
# ---------------------------------------------------------------------------

MODEL_CACHE_DIR = Path(
    os.environ.get("CARE2026_MODEL_CACHE", Path.home() / ".cache" / "care2026" / "models")
)
DATA_CACHE_DIR = Path(
    os.environ.get("CARE2026_DATA_CACHE", Path.home() / ".cache" / "care2026" / "data")
)

MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Spatial resolution constants (mm)
# ---------------------------------------------------------------------------

# LGE-MRI from Center A (Siemens scanner): in-plane 0.625 mm, slice 2.5 mm
MRI_CENTER_A_SPACING = (0.625, 0.625, 2.5)  # (x, y, z) mm

# CT from Center D: z-spacing is consistently 0.5 mm; in-plane is variable
CT_NATIVE_Z_SPACING = 0.5  # mm
CT_TARGET_SPACING = (0.5, 0.5, 0.5)  # isotropic target for resampling (mm)

# ---------------------------------------------------------------------------
# Patch / volume shape constants
# ---------------------------------------------------------------------------

# MRI: crop to the LA bounding box then resize to this shape (H, W, D)
MRI_PATCH_SHAPE = (256, 256, 44)  # matches typical LGE-MRI slice count

# CT: random 3-D patch size during training (isotropic cube), in voxels
CT_PATCH_SIZE = 128  # voxels per side after isotropic resampling

# ---------------------------------------------------------------------------
# CT Hounsfield Unit windowing
# ---------------------------------------------------------------------------

CT_HU_MIN = -200.0  # lower clip value (HU)
CT_HU_MAX = 800.0   # upper clip value (HU)

# ---------------------------------------------------------------------------
# Class mappings
# ---------------------------------------------------------------------------

TASK1_CLASS_MAP = {
    0: "background",
    1: "LA scar",
}

TASK2_CLASS_MAP = {
    0: "background",
    1: "left atrium",
}

CT_CLASS_MAP = {
    0: "background",
    1: "left atrium",
    2: "pulmonary veins",
    3: "left atrial appendage",
}

CT_NUM_CLASSES = len(CT_CLASS_MAP)  # 4

# ---------------------------------------------------------------------------
# Dataset size constants
# ---------------------------------------------------------------------------

TASK1_TRAIN_COUNT = 60    # Task 1 MRI samples (scar + cavity labels)
TASK2_TRAIN_COUNT = 130   # Task 2 MRI samples (cavity label only)
CT_TOTAL_COUNT = 150      # CT samples total
CT_LABELED_COUNT = 50     # CT samples with ground-truth labels (train_1..train_50)
CT_UNLABELED_COUNT = 100  # CT samples without labels (train_51..train_150)

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

DEFAULT_VAL_RATIO = 0.1  # 10 % of each dataset held out for validation

# Remote model weights (empty until models are trained and published)
REMOTE_MODELS: dict = {}
