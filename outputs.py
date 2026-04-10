"""
Output container definitions for model predictions.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["CARE2026Outputs"]


@dataclass
class CARE2026Outputs:
    """Container for CARE2026 model predictions.

    Parameters
    ----------
    task : str
        Either "mri" or "ct".
    la_mask : np.ndarray, optional
        LA cavity segmentation, shape (B, H, W, D), dtype uint8.
        Present for task="mri".
    scar_mask : np.ndarray, optional
        LA scar segmentation, shape (B, H, W, D), dtype uint8.
        Present for task="mri".
    ct_mask : np.ndarray, optional
        CT multi-structure segmentation, shape (B, H, W, D), dtype uint8.
        Present for task="ct".
    """

    task: str
    la_mask: Optional[np.ndarray] = None
    scar_mask: Optional[np.ndarray] = None
    ct_mask: Optional[np.ndarray] = None
