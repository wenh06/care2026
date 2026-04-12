"""
Distribution-based loss functions for 3-D segmentation.

All losses accept:
    logits : torch.Tensor, shape (B, C, *spatial)
    target : torch.Tensor, shape (B, *spatial), dtype long (integer class indices)
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FocalLoss", "TopKCELoss"]


class FocalLoss(nn.Module):
    """Multi-class focal loss for N-D volumetric tensors.

    Focal_Loss = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    https://arxiv.org/abs/1708.02002

    Parameters
    ----------
    gamma : float, default 2.0
        Focusing parameter; higher values down-weight easy examples more.
    alpha : float or list of float, optional
        Class weights.  If a single float, it is used as the foreground weight
        and ``(1 - alpha)`` for background.  If a list, must have length C.
        If None, all classes are equally weighted.
    reduction : {"mean", "sum", "none"}, default "mean"
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, List[float]]] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor, shape (B, C, *spatial)
        target : torch.Tensor, shape (B, *spatial), dtype long

        Returns
        -------
        torch.Tensor
            Scalar loss (or per-element tensor if reduction="none").
        """
        log_probs = F.log_softmax(logits, dim=1)  # (B, C, *spatial)
        probs = log_probs.exp()

        t = target.long().unsqueeze(1)  # (B, 1, *spatial)
        log_pt = log_probs.gather(1, t).squeeze(1)  # (B, *spatial)
        pt = probs.gather(1, t).squeeze(1)  # (B, *spatial)

        focal_weight = (1.0 - pt) ** self.gamma
        loss = -focal_weight * log_pt  # (B, *spatial)

        if self.alpha is not None:
            if isinstance(self.alpha, float):
                alpha_t = torch.where(
                    target.long() == 0,
                    torch.full_like(loss, 1.0 - self.alpha),
                    torch.full_like(loss, self.alpha),
                )
            else:
                alpha_tensor = logits.new_tensor(self.alpha)  # (C,)
                alpha_t = alpha_tensor[target.long()]  # (B, *spatial)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class TopKCELoss(nn.Module):
    """Cross-entropy loss computed only on the top-k% hardest voxels.

    Focuses training on hard examples without explicit focal weighting.
    Part of the nnUNet loss suite.

    Parameters
    ----------
    k : float, default 10.0
        Percentage of voxels to keep (those with the highest per-voxel CE loss).
    """

    def __init__(self, k: float = 10.0) -> None:
        super().__init__()
        self.k = k

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor, shape (B, C, *spatial)
        target : torch.Tensor, shape (B, *spatial), dtype long

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        ce = F.cross_entropy(logits, target.long(), reduction="none")  # (B, *spatial)
        ce_flat = ce.view(-1)
        topk_n = max(1, int(ce_flat.numel() * self.k / 100.0))
        topk_vals, _ = torch.topk(ce_flat, topk_n, sorted=False)
        return topk_vals.mean()
