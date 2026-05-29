"""
Loss functions for 3-D segmentation: SoftDiceLoss, DiceCELoss, TverskyLoss, FocalTverskyLoss.

All losses accept:
    logits : torch.Tensor, shape (B, C, H, W, D)
    target : torch.Tensor, shape (B, H, W, D), dtype long (integer class indices)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SoftDiceLoss", "DiceCELoss", "TverskyLoss", "FocalTverskyLoss"]


def _one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer label map to one-hot encoding.

    Parameters
    ----------
    target : torch.Tensor
        Integer class label map of shape (B, H, W, D).
    num_classes : int
        Number of classes.

    Returns
    -------
    torch.Tensor
        One-hot tensor of shape (B, C, H, W, D).
    """
    B = target.shape[0]
    spatial = target.shape[1:]
    one_hot = torch.zeros(B, num_classes, *spatial, device=target.device, dtype=torch.float32)
    one_hot.scatter_(1, target.unsqueeze(1).long(), 1.0)
    return one_hot


class SoftDiceLoss(nn.Module):
    """Soft Dice loss for multi-class volumetric segmentation.

    Parameters
    ----------
    do_bg : bool, default False
        Whether to include the background class (index 0) in the loss.
    smooth : float, default 1e-5
        Smoothing term for numerical stability.
    """

    def __init__(self, do_bg: bool = False, smooth: float = 1e-5) -> None:
        super().__init__()
        self.do_bg = do_bg
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor, shape (B, C, H, W, D)
        target : torch.Tensor, shape (B, H, W, D), dtype long

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # (B, C, H, W, D)
        one_hot = _one_hot(target, num_classes)  # (B, C, H, W, D)
        axes = tuple(range(2, logits.ndim))  # spatial dims
        intersection = (probs * one_hot).sum(dim=axes)  # (B, C)
        union = probs.sum(dim=axes) + one_hot.sum(dim=axes)  # (B, C)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (B, C)
        if not self.do_bg:
            dice = dice[:, 1:]
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss (nnUNet default).

    Parameters
    ----------
    dice_weight : float, default 0.5
    ce_weight : float, default 0.5
    do_bg : bool, default False
    smooth : float, default 1e-5
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        do_bg: bool = False,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dice = SoftDiceLoss(do_bg=do_bg, smooth=smooth)
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + self.ce_weight * self.ce(logits, target.long())


class TverskyLoss(nn.Module):
    """Tversky loss for class imbalance (FN-penalizing for sparse scar).

    Parameters
    ----------
    alpha : float, default 0.3
        Weight for false positives.
    beta : float, default 0.7
        Weight for false negatives.
    do_bg : bool, default False
    smooth : float, default 1e-5
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        do_bg: bool = False,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.do_bg = do_bg
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        one_hot = _one_hot(target, num_classes)
        axes = tuple(range(2, logits.ndim))
        tp = (probs * one_hot).sum(dim=axes)
        fp = (probs * (1.0 - one_hot)).sum(dim=axes)
        fn = ((1.0 - probs) * one_hot).sum(dim=axes)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        if not self.do_bg:
            tversky = tversky[:, 1:]
        return 1.0 - tversky.mean()


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss: Tversky index raised to power gamma.

    Parameters
    ----------
    alpha : float, default 0.3
    beta : float, default 0.7
    gamma : float, default 0.75
        Focusing parameter; < 1 focuses on hard examples.
    do_bg : bool, default False
    smooth : float, default 1e-5
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        do_bg: bool = False,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self._tversky = TverskyLoss(alpha=alpha, beta=beta, do_bg=do_bg, smooth=smooth)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tversky_loss = self._tversky(logits, target)
        tversky_index = 1.0 - tversky_loss
        return 1.0 - tversky_index**self.gamma
