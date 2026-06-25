"""
Centerline-based loss for vessel-like structure segmentation.

clCE (Acebes et al., MICCAI 2024): Cross-Entropy over soft-skeletonized
probability maps, combined with standard Dice loss.  Improves topology
preservation of thin tubular structures (PV) without sacrificing Dice.

Reference: https://arxiv.org/abs/2407.01517 (cbDice)
           https://github.com/cesaracebes/centerline_CE (clCE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dice_loss import SoftDiceLoss, _one_hot

__all__ = ["CenterlineCELoss"]


class CenterlineCELoss(nn.Module):
    """Centerline Cross-Entropy loss for topology-aware segmentation.

    Computes soft skeletons of both prediction and ground truth via
    differentiable min-pooling + ReLU, then applies Cross-Entropy
    between the skeleton maps.  Combine with Dice loss for best
    results: ``L_total = L_dice + lambda * L_clCE``.

    Parameters
    ----------
    kernel_size : int, default 5
        Pooling kernel size for soft-skeleton computation.
        Larger values produce wider skeletons.
    lambda_clce : float, default 1.0
        Weight of the clCE term relative to Dice.
    do_bg : bool, default False
    smooth : float, default 1e-5
    """

    def __init__(
        self,
        kernel_size: int = 5,
        lambda_clce: float = 0.5,
        do_bg: bool = False,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.lambda_clce = lambda_clce
        self.do_bg = do_bg
        self.smooth = smooth
        self.dice = SoftDiceLoss(do_bg=do_bg, smooth=smooth)

    @staticmethod
    def _min_pool(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Min-pooling via negated max-pooling: min(x) = -max(-x)."""
        return -F.max_pool3d(-x, kernel_size, stride=1, padding=kernel_size // 2)

    def soft_skel(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable soft-skeleton via morphological thinning.

        ``skel = ReLU(x - min_pool(x))`` keeps voxels that are locally
        maximal — the ridges of the probability map corresponding to
        vessel centerlines.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, H, W, D)

        Returns
        -------
        torch.Tensor, same shape as *x*
        """
        if self.kernel_size <= 1:
            return x
        min_pooled = self._min_pool(x, self.kernel_size)
        return F.relu(x - min_pooled) + self.smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor, shape (B, C, H, W, D)
        target : torch.Tensor, shape (B, H, W, D), dtype long

        Returns
        -------
        torch.Tensor
            Scalar loss = L_dice + λ_clCE * L_clCE.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        one_hot = _one_hot(target, num_classes)

        dice_loss = self.dice(logits, target)

        clce_total = logits.new_zeros(())
        count = 0
        start = 0 if self.do_bg else 1

        for c in range(start, num_classes):
            p_c = probs[:, c : c + 1]  # (B, 1, H, W, D)
            g_c = one_hot[:, c : c + 1]

            p_skel = self.soft_skel(p_c)
            g_skel = self.soft_skel(g_c)

            # CE between skeleton maps: treat skeleton as a spatial
            # distribution and measure KL divergence.  Normalise by
            # log(N) to keep the CE term in [0, 1] range, comparable
            # to the Dice loss.
            p_flat = p_skel.reshape(p_skel.shape[0], -1)
            g_flat = g_skel.reshape(g_skel.shape[0], -1)
            p_dist = p_flat / (p_flat.sum(dim=1, keepdim=True) + self.smooth)
            g_dist = g_flat / (g_flat.sum(dim=1, keepdim=True) + self.smooth)
            # CE = -sum(g * log(p)) / log(N)  (normalise to [0, 1])
            n_voxels = p_flat.shape[1]
            ce = -(g_dist * torch.log(p_dist + self.smooth)).sum(dim=1) / max(1.0, float(n_voxels))
            clce_total = clce_total + ce.mean()
            count += 1

        clce = clce_total / max(count, 1)
        return dice_loss + self.lambda_clce * clce
