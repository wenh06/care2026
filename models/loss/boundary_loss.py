"""
Boundary loss (Kervadec et al., 2019) with on-the-fly distance transform.
No precomputed tensors required.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

__all__ = ["BoundaryLoss"]


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
    return (dist_inside - dist_outside).astype(np.float32)


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
