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
        self.boundary_loss = BoundaryLoss() if w.get("scar_boundary", 0.0) > 0 else None
        self.w_dice = w.get("scar_dice", 1.0)
        self.w_focal = w.get("scar_focal", 0.5)
        self.w_boundary = w.get("scar_boundary", 0.0)
        self.spatial_w0 = w.get("spatial_w0", 5.0)
        self.sigma_mm = w.get("spatial_sigma_mm", 2.0)

    def forward(
        self,
        scar_logits: torch.Tensor,
        scar_target: torch.Tensor,
        has_scar: torch.Tensor,
        spacing: Tuple[float, float, float] = (0.625, 0.625, 2.5),
    ) -> dict:
        from scipy.ndimage import distance_transform_edt

        total = scar_logits.sum() * 0.0
        n_pos, n_neg = 0, 0
        sigma_px = max(1.0, self.sigma_mm / min(spacing[:2]))
        for b in range(scar_logits.shape[0]):
            sl = scar_logits[b : b + 1]  # (1, 2, H, W, D)
            if has_scar[b]:
                st = scar_target[b : b + 1].long()
                st_np = st.squeeze().cpu().numpy().astype(np.uint8)
                if st_np.sum() == 0:
                    continue
                d = distance_transform_edt(1 - st_np).astype(np.float32)
                w_map = 1.0 + self.spatial_w0 * np.exp(-(d**2) / (2 * sigma_px**2))
                w_t = torch.from_numpy(w_map).to(sl.device).unsqueeze(0)  # (1,H,W,D)
                logp = torch.log_softmax(sl, dim=1)
                ce_voxel = -logp.gather(1, st.unsqueeze(1)).squeeze(1)
                weighted_ce = (ce_voxel * w_t).mean()
                dice = self.dice_loss(sl, st)
                focal = self.focal_loss(sl, st)
                term = self.w_dice * dice + self.w_focal * focal + self.spatial_w0 * 0.1 * weighted_ce
                if self.boundary_loss is not None:
                    term = term + self.w_boundary * self.boundary_loss(sl, st)
                total = total + term
                n_pos += 1
            else:
                # No-scar sample: push scar probability towards zero everywhere
                scar_prob = torch.softmax(sl, dim=1)[:, 1]  # (1, H, W, D)
                total = total + 0.1 * scar_prob.mean()
                n_neg += 1

        total = total / max(n_pos + n_neg, 1)
        return {"scar_loss": total, "total_loss": total}


class CTLoss(nn.Module):
    """Loss for CT semi-supervised training (CPS or Mean Teacher).

    Parameters
    ----------
    cfg : CFG
        ``loss_weights``: sup_dice, sup_ce, cps (or mt_consist).
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        w = cfg.loss_weights
        class_w = w.get("ce_class_weight", None)
        if class_w is not None:
            class_w = torch.tensor(class_w, dtype=torch.float32)
        self.supervised_loss = DiceCELoss(
            dice_weight=w.get("sup_dice", 0.5), ce_weight=w.get("sup_ce", 0.5), ce_class_weight=class_w
        )
        self.consistency_fn = nn.MSELoss()  # Mean Teacher: MSE between softmax outputs
        self.boundary_loss = HausdorffERLoss(alpha=2.0, erosions=5) if w.get("sup_boundary", 0.0) > 0 else None
        self.w_boundary = w.get("sup_boundary", 0.0)
        self.w_cps = w.get("cps", 1.0)
        self.w_mt = w.get("mt_consist", 1.0)

    def forward(
        self,
        logits1: torch.Tensor,
        logits2: Optional[torch.Tensor] = None,
        logits_t: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        labeled_mask: Optional[torch.Tensor] = None,
        cps_weight: float = 1.0,
    ) -> dict:
        sup_loss = logits1.sum() * 0.0
        boundary_loss = logits1.sum() * 0.0
        if target is not None and labeled_mask is not None and labeled_mask.any():
            if logits2 is not None:
                # CPS: supervised loss on both models
                l1 = logits1[labeled_mask]
                l2 = logits2[labeled_mask]
                tgt = target[labeled_mask].long()
                sup_loss = 0.5 * (self.supervised_loss(l1, tgt) + self.supervised_loss(l2, tgt))
                if self.boundary_loss is not None:
                    boundary_loss = 0.5 * (self.boundary_loss(l1, tgt) + self.boundary_loss(l2, tgt))
            else:
                # Mean Teacher: supervised loss on student only
                sup_loss = self.supervised_loss(logits1[labeled_mask], target[labeled_mask].long())
                if self.boundary_loss is not None:
                    boundary_loss = self.boundary_loss(logits1[labeled_mask], target[labeled_mask].long())

        # Consistency loss
        consist_loss = logits1.sum() * 0.0
        if logits_t is not None:
            # Mean Teacher: MSE(student_softmax, teacher_softmax)
            consist_loss = (
                self.w_mt * cps_weight * self.consistency_fn(torch.softmax(logits1, dim=1), torch.softmax(logits_t, dim=1))
            )
        elif logits2 is not None:
            # CPS: cross-pseudo-label CE
            with torch.no_grad():
                pseudo1 = logits1.argmax(dim=1)
                pseudo2 = logits2.argmax(dim=1)
            cps1 = nn.functional.cross_entropy(logits1, pseudo2)
            cps2 = nn.functional.cross_entropy(logits2, pseudo1)
            consist_loss = self.w_cps * cps_weight * 0.5 * (cps1 + cps2)

        total = sup_loss + consist_loss
        if self.boundary_loss is not None:
            total = total + self.w_boundary * boundary_loss
        return {"sup_loss": sup_loss, "consist_loss": consist_loss, "total_loss": total}
