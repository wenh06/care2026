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
    # Two-stage MRI pipeline shapes
    "MRI_CANONICAL_SHAPE",
    "MRI_STAGE1_SHAPE",
    "MRI_STAGE2_CROP_SHAPE",
    "MRI_STAGE2_CACHE_SHAPE",
    "MRI_STAGE2_CENTROID_JITTER",
    "MRI_PATCH_SHAPE",  # alias for MRI_STAGE2_CROP_SHAPE
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

MODEL_CACHE_DIR = Path(os.environ.get("CARE2026_MODEL_CACHE", Path.home() / ".cache" / "care2026" / "models"))
DATA_CACHE_DIR = Path(os.environ.get("CARE2026_DATA_CACHE", Path.home() / ".cache" / "care2026" / "data"))

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
# Two-stage MRI pipeline shape constants (H × W × D, in voxels)
# ---------------------------------------------------------------------------
# Derived from CARE2026 training-set statistics (N=190 LGE-MRI volumes):
#   - Raw image mode: 576×576×44 at 0.625×0.625×2.5 mm spacing
#   - LA bbox p95:  (200 × 123 × 62) voxels in raw space
#
# All raw volumes are first resampled to MRI_CANONICAL_SHAPE, then processed
# through two stages.  Changing any of these constants propagates automatically
# through dataset / model / inference code without touching implementation logic.

# Step 0 – resample every raw volume to this common grid
MRI_CANONICAL_SHAPE = (576, 576, 44)

# Stage 1 – 4× downsampled in H,W; D kept intact (input to coarse LA localiser)
MRI_STAGE1_SHAPE = (144, 144, 44)

# Stage 2 – crop centred on the predicted LA centroid (input to fine segmenter)
MRI_STAGE2_CROP_SHAPE = (256, 256, 44)

# Internal cache shape for Stage 2 dataset: generous margin around GT centroid
# so that ±MRI_STAGE2_CENTROID_JITTER random crop offsets remain in bounds.
# Margin = (cache_hw - crop_hw) / 2 = (320 - 256) / 2 = 32  (== jitter_max)
MRI_STAGE2_CACHE_SHAPE = (320, 320, 44)

# Maximum per-axis centroid jitter (H, W, D) during Stage 2 training.
# Simulates Stage 1 localisation errors; must satisfy jitter ≤ margin above.
MRI_STAGE2_CENTROID_JITTER = (32, 32, 0)

# Backward-compatibility alias  (= Stage 2 crop shape)
MRI_PATCH_SHAPE = MRI_STAGE2_CROP_SHAPE

# CT: random 3-D patch size during training (isotropic cube), in voxels
CT_PATCH_SIZE = 128  # voxels per side after isotropic resampling

# ---------------------------------------------------------------------------
# CT Hounsfield Unit windowing
# ---------------------------------------------------------------------------

CT_HU_MIN = -200.0  # lower clip value (HU)
CT_HU_MAX = 800.0  # upper clip value (HU)

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

TASK1_TRAIN_COUNT = 60  # Task 1 MRI samples (scar + cavity labels)
TASK2_TRAIN_COUNT = 130  # Task 2 MRI samples (cavity label only)
CT_TOTAL_COUNT = 150  # CT samples total
CT_LABELED_COUNT = 50  # CT samples with ground-truth labels (train_1..train_50)
CT_UNLABELED_COUNT = 100  # CT samples without labels (train_51..train_150)

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

DEFAULT_VAL_RATIO = 0.1  # 10 % of each dataset held out for validation

# Remote model weights (empty until models are trained and published)
REMOTE_MODELS: dict = {}
