"""
Compound (combined) loss functions for 3-D segmentation.

All losses accept:
    logits : torch.Tensor, shape (B, C, *spatial)
    target : torch.Tensor, shape (B, *spatial), dtype long (integer class indices)
"""

import torch
import torch.nn as nn

from .boundary_loss import BoundaryLoss
from .dice_loss import SoftDiceLoss
from .distribution_loss import FocalLoss, TopKCELoss

__all__ = ["DiceFocalLoss", "DiceBoundaryLoss", "DiceTopKLoss"]


class DiceFocalLoss(nn.Module):
    """Dice + Focal loss — better than Dice+CE for heavily imbalanced classes.

    The focal component automatically suppresses easy-background gradients,
    letting the Dice term focus on rare foreground (e.g., atrial scar).

    Parameters
    ----------
    dice_weight : float, default 0.5
    focal_weight : float, default 0.5
    gamma : float, default 2.0
        Focal loss focusing parameter.
    do_bg : bool, default False
    smooth : float, default 1e-5
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        gamma: float = 2.0,
        do_bg: bool = False,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dice = SoftDiceLoss(do_bg=do_bg, smooth=smooth)
        self.focal = FocalLoss(gamma=gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + self.focal_weight * self.focal(logits, target)


class DiceBoundaryLoss(nn.Module):
    """Dice + Boundary (distance-transform) loss.

    Boundary loss penalises predictions far from the true contour, while the
    Dice term handles volumetric overlap.  The combination helps sharpen thin
    structures such as the atrial wall.

    Parameters
    ----------
    dice_weight : float, default 0.5
    boundary_weight : float, default 0.5
    do_bg : bool, default False
    smooth : float, default 1e-5
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        boundary_weight: float = 0.5,
        do_bg: bool = False,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dice = SoftDiceLoss(do_bg=do_bg, smooth=smooth)
        self.boundary = BoundaryLoss()
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + self.boundary_weight * self.boundary(logits, target)


class DiceTopKLoss(nn.Module):
    """Dice + TopK cross-entropy loss (nnUNet variant).

    TopK CE trains only on the hardest k% voxels (by per-voxel CE loss),
    providing a robust alternative to focal loss that avoids tuning gamma.

    Parameters
    ----------
    dice_weight : float, default 0.5
    topk_weight : float, default 0.5
    k : float, default 10.0
        Percentage of hardest voxels to keep.
    do_bg : bool, default False
    smooth : float, default 1e-5
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        topk_weight: float = 0.5,
        k: float = 10.0,
        do_bg: bool = False,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dice = SoftDiceLoss(do_bg=do_bg, smooth=smooth)
        self.topk = TopKCELoss(k=k)
        self.dice_weight = dice_weight
        self.topk_weight = topk_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + self.topk_weight * self.topk(logits, target)
