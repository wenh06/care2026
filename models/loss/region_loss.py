"""
Region-based loss functions for 3-D segmentation.

All losses accept:
    logits : torch.Tensor, shape (B, C, *spatial)
    target : torch.Tensor, shape (B, *spatial), dtype long (integer class indices)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dice_loss import _one_hot

__all__ = ["IoULoss", "GeneralizedDiceLoss", "LovaszSoftmaxLoss"]


class IoULoss(nn.Module):
    """Soft Intersection-over-Union (Jaccard) loss.

    IoU = TP / (TP + FP + FN)

    IoU = Dice / (2 - Dice), so this loss has slightly sharper gradients than
    SoftDiceLoss for low-overlap predictions.

    Parameters
    ----------
    do_bg : bool, default False
        Whether to include the background class (index 0).
    smooth : float, default 1e-5
    """

    def __init__(self, do_bg: bool = False, smooth: float = 1e-5) -> None:
        super().__init__()
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
        iou = (tp + self.smooth) / (tp + fp + fn + self.smooth)
        if not self.do_bg:
            iou = iou[:, 1:]
        return 1.0 - iou.mean()


class GeneralizedDiceLoss(nn.Module):
    """Generalized Dice loss with inverse-frequency class weighting.

    Rare classes (e.g., scar voxels) automatically receive higher weights,
    which is preferable to manual tuning for heavily imbalanced segmentation.

    Paper: "Generalised Dice overlap as a deep learning loss function for
    highly unbalanced segmentations" (Sudre et al., 2017).
    https://arxiv.org/abs/1707.03237

    Parameters
    ----------
    do_bg : bool, default False
    smooth : float, default 1e-5
    """

    def __init__(self, do_bg: bool = False, smooth: float = 1e-5) -> None:
        super().__init__()
        self.do_bg = do_bg
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        one_hot = _one_hot(target, num_classes)
        if not self.do_bg:
            probs = probs[:, 1:]
            one_hot = one_hot[:, 1:]
        axes = tuple(range(2, logits.ndim))
        with torch.no_grad():
            vol = one_hot.sum(dim=axes)  # (B, C')
            weight = 1.0 / (vol**2 + self.smooth)
        intersection = (weight * (probs * one_hot).sum(dim=axes)).sum(dim=1)
        union = (weight * (probs + one_hot).sum(dim=axes)).sum(dim=1)
        gdc = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - gdc.mean()


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Lovász extension gradient (Algorithm 1 in the paper)."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmaxLoss(nn.Module):
    """Lovász-Softmax loss — differentiable surrogate for the IoU metric.

    Provides exact gradient w.r.t. IoU via the Lovász extension.  Strong
    alternative to Dice/IoU losses when metric-aligned training is desired.

    Paper: "The Lovász-Softmax loss" (Berman et al., 2018).
    https://arxiv.org/abs/1705.08790

    Parameters
    ----------
    reduction : {"mean", "sum", "none"}, default "mean"
    do_bg : bool, default False
    """

    def __init__(self, reduction: str = "mean", do_bg: bool = False) -> None:
        super().__init__()
        self.reduction = reduction
        self.do_bg = do_bg

    def _lovasz_softmax_flat(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        probs   : (N, C) probabilities (softmax applied)
        targets : (N,)   integer labels
        """
        num_classes = probs.size(1)
        start = 0 if self.do_bg else 1
        losses = []
        for c in range(start, num_classes):
            target_c = (targets == c).float()
            pred_c = probs[:, c]
            error = (target_c - pred_c).abs()
            error_sorted, idx = torch.sort(error, descending=True)
            target_sorted = target_c[idx]
            grad = _lovasz_grad(target_sorted)
            losses.append(torch.dot(error_sorted, grad))
        losses = torch.stack(losses)
        if self.reduction == "none":
            return losses
        elif self.reduction == "sum":
            return losses.sum()
        return losses.mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor, shape (B, C, *spatial)
        target : torch.Tensor, shape (B, *spatial), dtype long
        """
        probs = F.softmax(logits, dim=1)
        C = probs.shape[1]
        probs_flat = probs.permute(0, *range(2, probs.ndim), 1).contiguous().view(-1, C)
        target_flat = target.long().view(-1)
        return self._lovasz_softmax_flat(probs_flat, target_flat)
