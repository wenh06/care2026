"""
Boundary-based loss functions for 3-D segmentation.

- BoundaryLoss      : signed-distance-map weighted Dice (Kervadec et al., 2019)
- HausdorffDTLoss   : DT-based approximation of Hausdorff distance loss
- HausdorffERLoss   : erosion-based Hausdorff loss (GPU-friendly, no scipy)

All losses accept:
    logits : torch.Tensor, shape (B, C, *spatial)
    target : torch.Tensor, shape (B, *spatial), dtype long (integer class indices)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from .dice_loss import _one_hot

__all__ = ["BoundaryLoss", "HausdorffDTLoss", "HausdorffERLoss"]


def _compute_distance_map(seg: np.ndarray) -> np.ndarray:
    """Compute signed distance map for a binary segmentation mask.

    Positive inside foreground, negative outside.
    Returns zero array if mask is empty or full.

    Parameters
    ----------
    seg : np.ndarray
        Binary mask of shape (H, W, D).
    """
    if seg.max() == 0 or seg.min() == 1:
        return np.zeros_like(seg, dtype=np.float32)
    dist_inside = distance_transform_edt(seg)
    dist_outside = distance_transform_edt(1 - seg)
    # Positive outside, negative inside — matches Kervadec et al. (2019):
    # minimising mean(prob_fg * phi) pushes prob_fg up inside (phi < 0) and
    # down outside (phi > 0).
    return (dist_outside - dist_inside).astype(np.float32)


class BoundaryLoss(nn.Module):
    """Boundary loss for binary segmentation.

    Computes distance maps on-the-fly from ground truth.
    Default weight in cfg is 0 (disabled); activate by setting scar_boundary > 0.

    loss = mean(softmax_prob_foreground * distance_map)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor, shape (B, 2, H, W, D)
            Binary logits.
        target : torch.Tensor, shape (B, H, W, D), dtype long

        Returns
        -------
        torch.Tensor
            Scalar boundary loss.
        """
        probs_fg = F.softmax(logits, dim=1)[:, 1]  # (B, H, W, D)
        target_np = target.detach().cpu().numpy().astype(np.uint8)
        dist_maps = [_compute_distance_map(target_np[i]) for i in range(target_np.shape[0])]
        dist_tensor = torch.from_numpy(np.stack(dist_maps, axis=0)).to(device=logits.device, dtype=logits.dtype)
        return (probs_fg * dist_tensor).mean()


class HausdorffDTLoss(nn.Module):
    """Multi-class Hausdorff distance loss via distance transform (DT).

    For each foreground class, computes DT fields on both the prediction and
    the ground truth, then penalises their disagreement weighted by distance.

    Paper: "How Distance Transform Maps Boost Segmentation CNNs"
    (Karimi & Salcudean, 2019). https://arxiv.org/abs/1904.10030

    .. warning::
        Runs scipy DT on CPU for every sample × class per iteration.
        Recommended for fine-tuning stages only, not full training.

    Parameters
    ----------
    alpha : float, default 2.0
        Exponent applied to the distance fields; higher values penalise distant
        errors more strongly.
    do_bg : bool, default False
    """

    def __init__(self, alpha: float = 2.0, do_bg: bool = False) -> None:
        super().__init__()
        self.alpha = alpha
        self.do_bg = do_bg

    @staticmethod
    def _dt_field(binary_mask: np.ndarray) -> np.ndarray:
        """DT field = DT(fg) + DT(bg) for each sample.

        Parameters
        ----------
        binary_mask : np.ndarray, shape (B, *spatial), values in [0, 1]

        Returns
        -------
        np.ndarray, shape (B, *spatial)
        """
        field = np.zeros_like(binary_mask, dtype=np.float32)
        for b in range(binary_mask.shape[0]):
            fg = binary_mask[b] > 0.5
            if fg.any() and not fg.all():
                field[b] = distance_transform_edt(fg) + distance_transform_edt(~fg)
        return field

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor, shape (B, C, *spatial)
        target : torch.Tensor, shape (B, *spatial), dtype long
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        one_hot = _one_hot(target, num_classes)

        start = 0 if self.do_bg else 1
        total = logits.new_zeros(())
        count = 0
        for c in range(start, num_classes):
            pred_c_np = probs[:, c].detach().cpu().numpy()
            tgt_c_np = one_hot[:, c].detach().cpu().numpy()
            pred_dt = torch.from_numpy(self._dt_field(pred_c_np)).to(device=logits.device, dtype=logits.dtype)
            tgt_dt = torch.from_numpy(self._dt_field(tgt_c_np)).to(device=logits.device, dtype=logits.dtype)
            pred_error = (probs[:, c] - one_hot[:, c]) ** 2
            distance = pred_dt**self.alpha + tgt_dt**self.alpha
            total = total + (pred_error * distance).mean()
            count += 1
        return total / max(count, 1)


class HausdorffERLoss(nn.Module):
    """Multi-class Hausdorff loss via iterative morphological erosion (GPU-friendly).

    Approximates the Hausdorff distance using repeated convolution-based
    erosion, avoiding scipy and running entirely on the GPU.

    Paper: "Boundary-weighted Domain Adaptive Neural Network for Prostate MRI
    Segmentation" (Wang et al., 2019). Erosion variant from the original
    HausdorffLoss repository.

    Parameters
    ----------
    alpha : float, default 2.0
    erosions : int, default 10
        Number of erosion steps.
    do_bg : bool, default False
    """

    def __init__(self, alpha: float = 2.0, erosions: int = 10, do_bg: bool = False) -> None:
        super().__init__()
        self.alpha = alpha
        self.erosions = erosions
        self.do_bg = do_bg

        cross = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32)
        bound = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        self.register_buffer("kernel2D", cross.unsqueeze(0).unsqueeze(0) / 5.0)
        self.register_buffer("kernel3D", torch.stack([bound, cross, bound]).unsqueeze(0).unsqueeze(0) / 7.0)

    def _erode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            return F.conv3d(x, self.kernel3D.to(dtype=x.dtype, device=x.device), padding=1)
        return F.conv2d(x, self.kernel2D.to(dtype=x.dtype, device=x.device), padding=1)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor, shape (B, C, *spatial)
        target : torch.Tensor, shape (B, *spatial), dtype long
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        one_hot = _one_hot(target, num_classes)

        start = 0 if self.do_bg else 1
        total = logits.new_zeros(())
        count = 0
        spatial_dims = list(range(2, logits.dim()))
        for c in range(start, num_classes):
            bound = (probs[:, c : c + 1] - one_hot[:, c : c + 1]) ** 2
            eroded = torch.zeros_like(bound)
            for k in range(self.erosions):
                bound = self._erode(bound)
                erosion = F.relu(bound - 0.5)
                ptp = erosion.amax(dim=spatial_dims, keepdim=True) - erosion.amin(dim=spatial_dims, keepdim=True) + 1e-6
                eroded = eroded + erosion / ptp * (k + 1) ** self.alpha
            total = total + eroded.mean()
            count += 1
        return total / max(count, 1)
