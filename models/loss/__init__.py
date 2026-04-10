"""
Loss function module for CARE2026.
"""

from typing import Optional

import torch
import torch.nn as nn

from .boundary_loss import BoundaryLoss
from .dice_loss import DiceCELoss, FocalTverskyLoss, SoftDiceLoss, TverskyLoss

__all__ = [
    "SoftDiceLoss",
    "DiceCELoss",
    "TverskyLoss",
    "FocalTverskyLoss",
    "BoundaryLoss",
    "MRILoss",
    "CTLoss",
]


class MRILoss(nn.Module):
    """Compound loss for dual-head MRI model.

    - LA cavity head: DiceCELoss
    - Scar head: TverskyLoss + FocalTverskyLoss + optional BoundaryLoss

    Parameters
    ----------
    cfg : CFG
        Training configuration with loss_weights dict containing:
        la_dice, scar_dice, scar_focal, scar_boundary (0 = disabled).
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.la_dice_loss = DiceCELoss(dice_weight=0.5, ce_weight=0.5)
        self.scar_tversky = TverskyLoss(alpha=0.3, beta=0.7)
        self.scar_focal = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)
        self.scar_boundary = BoundaryLoss()
        w = cfg.loss_weights
        self.w_la_dice = w.get("la_dice", 1.0)
        self.w_scar_dice = w.get("scar_dice", 2.0)
        self.w_scar_focal = w.get("scar_focal", 0.5)
        self.w_scar_boundary = w.get("scar_boundary", 0.0)

    def forward(
        self,
        la_logits: torch.Tensor,
        scar_logits: torch.Tensor,
        la_target: torch.Tensor,
        scar_target: torch.Tensor,
        has_scar: torch.Tensor,
    ) -> dict:
        """Compute the MRI multi-task loss.

        Parameters
        ----------
        la_logits : torch.Tensor, shape (B, 2, H, W, D)
        scar_logits : torch.Tensor, shape (B, 2, H, W, D)
        la_target : torch.Tensor, shape (B, H, W, D), dtype long
        scar_target : torch.Tensor, shape (B, H, W, D), dtype long
            Zeros for samples without scar annotation.
        has_scar : torch.Tensor, shape (B,), dtype bool

        Returns
        -------
        dict
            Keys: la_loss, scar_loss, total_loss.
        """
        la_loss = self.la_dice_loss(la_logits, la_target.long())
        scar_loss = torch.tensor(0.0, device=la_logits.device)
        if has_scar.any():
            sl = scar_logits[has_scar]
            st = scar_target[has_scar].long()
            scar_loss = self.w_scar_dice * self.scar_tversky(sl, st) + self.w_scar_focal * self.scar_focal(sl, st)
            if self.w_scar_boundary > 0:
                scar_loss = scar_loss + self.w_scar_boundary * self.scar_boundary(sl, st)
        total_loss = self.w_la_dice * la_loss + scar_loss
        return {"la_loss": la_loss, "scar_loss": scar_loss, "total_loss": total_loss}


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
