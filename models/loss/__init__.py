"""
Loss function module for CARE2026.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .boundary_loss import BoundaryLoss, HausdorffDTLoss, HausdorffERLoss
from .compound_loss import DiceBoundaryLoss, DiceFocalLoss, DiceTopKLoss
from .dice_loss import DiceCELoss, FocalTverskyLoss, SoftDiceLoss, TverskyLoss
from .distribution_loss import FocalLoss, TopKCELoss
from .region_loss import GeneralizedDiceLoss, IoULoss, LovaszSoftmaxLoss

__all__ = [
    # dice / region — dice_loss.py
    "SoftDiceLoss",
    "DiceCELoss",
    "TverskyLoss",
    "FocalTverskyLoss",
    # region — region_loss.py
    "IoULoss",
    "GeneralizedDiceLoss",
    "LovaszSoftmaxLoss",
    # distribution — distribution_loss.py
    "FocalLoss",
    "TopKCELoss",
    # boundary — boundary_loss.py
    "BoundaryLoss",
    "HausdorffDTLoss",
    "HausdorffERLoss",
    # compound — compound_loss.py
    "DiceFocalLoss",
    "DiceBoundaryLoss",
    "DiceTopKLoss",
    # task-level compound wrappers
    "Stage1MRILoss",
    "ScarLoss",
    "CTLoss",
]


class Stage1MRILoss(nn.Module):
    """Loss for Stage 1 MRI coarse localisation (binary LA only).

    Uses DiceCELoss with equal Dice/CE weighting.  No scar head.

    Parameters
    ----------
    cfg : CFG
        Training configuration.  Only ``loss_weights.la_dice`` is used.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.criterion = DiceCELoss(dice_weight=0.5, ce_weight=0.5)
        w = cfg.loss_weights
        self.w_la = w.get("la_dice", 1.0)

    def forward(self, la_logits: torch.Tensor, la_target: torch.Tensor) -> dict:
        """Compute Stage 1 loss.

        Parameters
        ----------
        la_logits : torch.Tensor, shape (B, 2, H, W, D)
        la_target : torch.Tensor, shape (B, H, W, D), dtype long

        Returns
        -------
        dict with keys: la_loss, total_loss
        """
        la_loss = self.criterion(la_logits, la_target.long())
        return {"la_loss": la_loss, "total_loss": self.w_la * la_loss}


class ScarLoss(nn.Module):
    """Scar-only loss with Gaussian spatial weighting.

    Scar is extremely sparse (~2.4 % of LA voxels) and located in the
    thin atrial wall.  This loss combines an unweighted Dice+Focal
    component with a spatially-weighted cross-entropy term that
    up-weights voxels near GT scar::

        w(x) = 1 + w₀ · exp(−d(x)² / 2σ²)

    where *d(x)* is the Euclidean distance to the nearest scar voxel.

    Parameters
    ----------
    cfg : CFG
        ``loss_weights``: ``scar_dice`` (default 1.0), ``scar_focal``
        (default 0.5), ``spatial_w0`` (default 5.0),
        ``spatial_sigma_mm`` (default 2.0).
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        w = cfg.loss_weights
        self.dice_loss = DiceCELoss(dice_weight=0.5, ce_weight=0.5)
        self.focal_loss = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)
        self.w_dice = w.get("scar_dice", 1.0)
        self.w_focal = w.get("scar_focal", 0.5)
        self.spatial_w0 = w.get("spatial_w0", 5.0)
        self.sigma_mm = w.get("spatial_sigma_mm", 2.0)

    def forward(
        self,
        scar_logits: torch.Tensor,
        scar_target: torch.Tensor,
        has_scar: torch.Tensor,
        spacing: Tuple[float, float, float] = (0.625, 0.625, 2.5),
    ) -> dict:
        if not has_scar.any():
            return {
                "scar_loss": torch.tensor(0.0, device=scar_logits.device),
                "total_loss": torch.tensor(0.0, device=scar_logits.device),
            }

        from scipy.ndimage import distance_transform_edt

        total = scar_logits.sum() * 0.0
        n = 0
        sigma_px = max(1.0, self.sigma_mm / min(spacing[:2]))
        for b in range(scar_logits.shape[0]):
            if not has_scar[b]:
                continue
            sl = scar_logits[b : b + 1]
            st = scar_target[b : b + 1].long()
            st_np = st.squeeze().cpu().numpy().astype(np.uint8)
            if st_np.sum() == 0:
                continue
            # Spatial weight map
            d = distance_transform_edt(1 - st_np).astype(np.float32)
            w_map = 1.0 + self.spatial_w0 * np.exp(-(d**2) / (2 * sigma_px**2))
            w_t = torch.from_numpy(w_map).to(sl.device).unsqueeze(0)  # (1,H,W,D)

            # Weighted CE: CE loss weighted per-voxel by w_map
            logp = torch.log_softmax(sl, dim=1)  # (1, 2, H, W, D)
            ce_voxel = -logp.gather(1, st.unsqueeze(1)).squeeze(1)  # (1, H, W, D)
            weighted_ce = (ce_voxel * w_t).mean()

            dice = self.dice_loss(sl, st)
            focal = self.focal_loss(sl, st)
            total += self.w_dice * dice + self.w_focal * focal + self.spatial_w0 * 0.1 * weighted_ce
            n += 1

        total = total / max(n, 1)
        return {"scar_loss": total, "total_loss": total}


class CTLoss(nn.Module):
    """Loss for CT semi-supervised CPS training.

    Supervised component: DiceCELoss on labeled samples.
    CPS component: CrossEntropyLoss between model predictions.

    Parameters
    ----------
    cfg : CFG
        Training configuration with loss_weights: sup_dice, sup_ce, cps.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        w = cfg.loss_weights
        self.supervised_loss = DiceCELoss(
            dice_weight=w.get("sup_dice", 0.5),
            ce_weight=w.get("sup_ce", 0.5),
        )
        self.cps_loss_fn = nn.CrossEntropyLoss()
        self.w_cps = w.get("cps", 1.0)

    def forward(
        self,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        labeled_mask: Optional[torch.Tensor] = None,
        cps_weight: float = 1.0,
    ) -> dict:
        """Compute CT combined loss.

        Parameters
        ----------
        logits1 : torch.Tensor, shape (B, 4, H, W, D)
        logits2 : torch.Tensor, shape (B, 4, H, W, D)
        target : torch.Tensor, optional, shape (B, H, W, D), dtype long
        labeled_mask : torch.Tensor, optional, shape (B,), dtype bool
        cps_weight : float, default 1.0
            Ramp-up factor (0→1).

        Returns
        -------
        dict
            Keys: sup_loss, cps_loss, total_loss.
        """
        sup_loss = torch.tensor(0.0, device=logits1.device)
        if target is not None and labeled_mask is not None and labeled_mask.any():
            l1 = logits1[labeled_mask]
            l2 = logits2[labeled_mask]
            tgt = target[labeled_mask].long()
            sup_loss = 0.5 * (self.supervised_loss(l1, tgt) + self.supervised_loss(l2, tgt))

        with torch.no_grad():
            pseudo1 = logits1.argmax(dim=1)
            pseudo2 = logits2.argmax(dim=1)
        cps = 0.5 * (self.cps_loss_fn(logits1, pseudo2) + self.cps_loss_fn(logits2, pseudo1))
        cps_loss = self.w_cps * cps_weight * cps

        return {"sup_loss": sup_loss, "cps_loss": cps_loss, "total_loss": sup_loss + cps_loss}
