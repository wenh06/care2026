"""
Loss function module for CARE2026.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .boundary_loss import BoundaryLoss, HausdorffDTLoss, HausdorffERLoss
from .centerline_loss import CenterlineCELoss
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
    # centerline loss
    "CenterlineCELoss",
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

    Supports two supervised loss modes (configurable via ``loss_weights.loss_mode``):

    - ``"focal_tversky"`` (default): FocalTverskyLoss(α=0.7, β=0.3, γ=0.75).
      Penalises false positives to reduce foreground over-prediction.
    - ``"dice_ce"`` (nnUNet): DiceCELoss with configurable weights.
      Matches nnUNet's native training recipe.

    Optional: CE, boundary loss (HausdorffERLoss), clCE.

    Parameters
    ----------
    cfg : CFG
        ``loss_weights`` keys: loss_mode, tversky_alpha/beta/gamma,
        sup_dice, sup_ce, ce_class_weight, sup_boundary, sup_clce,
        clce_classes, cps, mt_consist, deep_supervision.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        w = cfg.loss_weights
        self._loss_mode = w.get("loss_mode", "focal_tversky")
        self._deep_sup = w.get("deep_supervision", False)

        if self._loss_mode == "dice_ce":
            # nnUNet-style: Dice + CE
            dice_w = w.get("sup_dice", 0.5)
            ce_w = w.get("sup_ce", 0.5)
            class_w = w.get("ce_class_weight", None)
            if class_w is not None:
                class_w = torch.tensor(class_w, dtype=torch.float32)
            self.supervised_loss = DiceCELoss(
                dice_weight=dice_w,
                ce_weight=ce_w,
                do_bg=False,
                ce_class_weight=class_w,
            )
            self.w_ce = 0.0  # CE already included in DiceCELoss
            self.ce_loss = None
        else:
            # FocalTversky: α > 0.5 penalises FP more than FN
            self.supervised_loss = FocalTverskyLoss(
                alpha=w.get("tversky_alpha", 0.7),
                beta=w.get("tversky_beta", 0.3),
                gamma=w.get("tversky_gamma", 0.75),
                do_bg=False,
            )
            # Optional CE (separate from FocalTversky)
            self.w_ce = w.get("sup_ce", 0.0)
            if self.w_ce > 0:
                class_w = w.get("ce_class_weight", None)
                if class_w is not None:
                    class_w = torch.tensor(class_w, dtype=torch.float32)
                self.ce_loss = nn.CrossEntropyLoss(weight=class_w)
            else:
                self.ce_loss = None

        self.consistency_fn = nn.MSELoss()
        self.boundary_loss = HausdorffERLoss(alpha=2.0, erosions=5) if w.get("sup_boundary", 0.0) > 0 else None
        self.w_boundary = w.get("sup_boundary", 0.0)
        self.clce_loss = CenterlineCELoss(kernel_size=5, lambda_clce=1.0) if w.get("sup_clce", 0.0) > 0 else None
        self.clce_classes = w.get("clce_classes", None)
        self.w_cps = w.get("cps", 1.0)
        self.w_mt = w.get("mt_consist", 1.0)

    def _compute_clce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute clCE loss, optionally restricted to specific classes."""
        if self.clce_classes is not None:
            # Restrict to selected classes: for each class, build binary logits
            # (bg probability + class probability) and binary target
            total = logits.new_zeros(())
            for cls_id in self.clce_classes:
                bin_logits = torch.stack([logits[:, 0], logits[:, cls_id]], dim=1)  # (B,2,H,W,D)
                bin_target = (target == cls_id).long()
                total = total + self.clce_loss(bin_logits, bin_target)
            return total / len(self.clce_classes)
        return self.clce_loss(logits, target)

    def _sup_loss(self, logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Supervised loss on a single logits tensor (per-level or single)."""
        loss = self.supervised_loss(logits, tgt)
        if self.ce_loss is not None:
            loss = loss + self.w_ce * self.ce_loss(logits, tgt)
        return loss

    def _maybe_deep_sup_loss(self, logits, tgt: torch.Tensor) -> torch.Tensor:
        """If *logits* is a deep supervision list, compute per-level loss (nnUNet style).
        Otherwise, return single-level supervised loss.
        """
        if isinstance(logits, (list, tuple)) and self._deep_sup:
            tgt_spatial = tgt.shape[1:]
            total = logits[0].sum() * 0.0
            n = len(logits)
            for lo in logits:
                if lo.shape[2:] != tgt_spatial:
                    lo = nn.functional.interpolate(lo, size=tgt_spatial, mode="trilinear", align_corners=False)
                total = total + self._sup_loss(lo, tgt)
            return total / n
        return self._sup_loss(logits, tgt)

    def forward(
        self,
        logits1: torch.Tensor,
        logits2: Optional[torch.Tensor] = None,
        logits_t: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        labeled_mask: Optional[torch.Tensor] = None,
        cps_weight: float = 1.0,
        clce_weight: float = 0.0,
    ) -> dict:
        sup_loss = logits1.sum() * 0.0 if not isinstance(logits1, (list, tuple)) else logits1[0].sum() * 0.0
        boundary_loss = sup_loss * 0.0
        clce_loss = sup_loss * 0.0

        if target is not None and labeled_mask is not None and labeled_mask.any():
            tgt_labeled = target[labeled_mask].long()
            if logits2 is not None:
                # CPS: supervised loss on both models (no deep sup for CPS)
                l1 = logits1[labeled_mask] if not isinstance(logits1, (list, tuple)) else logits1[0][labeled_mask]
                l2 = logits2[labeled_mask] if not isinstance(logits2, (list, tuple)) else logits2[0][labeled_mask]
                sup_loss = 0.5 * (self._sup_loss(l1, tgt_labeled) + self._sup_loss(l2, tgt_labeled))
                if self.boundary_loss is not None:
                    boundary_loss = 0.5 * (self.boundary_loss(l1, tgt_labeled) + self.boundary_loss(l2, tgt_labeled))
                if self.clce_loss is not None and clce_weight > 0:
                    clce_loss = 0.5 * (self._compute_clce(l1, tgt_labeled) + self._compute_clce(l2, tgt_labeled))
            else:
                if isinstance(logits1, (list, tuple)) and self._deep_sup:
                    # Apply labeled_mask to each level's logits
                    l1_labeled = [lo[labeled_mask] for lo in logits1]
                    sup_loss = self._maybe_deep_sup_loss(l1_labeled, tgt_labeled)
                else:
                    l1_tensor = logits1[labeled_mask] if not isinstance(logits1, (list, tuple)) else logits1
                    sup_loss = self._sup_loss(l1_tensor, tgt_labeled)
                if self.boundary_loss is not None:
                    l1_bd = logits1[0][labeled_mask] if isinstance(logits1, (list, tuple)) else logits1[labeled_mask]
                    boundary_loss = self.boundary_loss(l1_bd, tgt_labeled)
                if self.clce_loss is not None and clce_weight > 0:
                    l1_cl = logits1[0][labeled_mask] if isinstance(logits1, (list, tuple)) else logits1[labeled_mask]
                    clce_loss = self._compute_clce(l1_cl, tgt_labeled)

        # Consistency loss (on full-resolution output only)
        l1_consist = logits1[0] if isinstance(logits1, (list, tuple)) else logits1
        l2_consist = logits2[0] if isinstance(logits2, (list, tuple)) else logits2 if logits2 is not None else None
        lt_consist = logits_t[0] if isinstance(logits_t, (list, tuple)) else logits_t if logits_t is not None else None

        consist_loss = l1_consist.sum() * 0.0
        if lt_consist is not None:
            consist_loss = (
                self.w_mt * cps_weight * self.consistency_fn(torch.softmax(l1_consist, dim=1), torch.softmax(lt_consist, dim=1))
            )
        elif l2_consist is not None:
            with torch.no_grad():
                pseudo1 = l1_consist.argmax(dim=1)
                pseudo2 = l2_consist.argmax(dim=1)
            cps1 = nn.functional.cross_entropy(l1_consist, pseudo2)
            cps2 = nn.functional.cross_entropy(l2_consist, pseudo1)
            consist_loss = self.w_cps * cps_weight * 0.5 * (cps1 + cps2)

        total = sup_loss + consist_loss
        if self.boundary_loss is not None:
            total = total + self.w_boundary * boundary_loss
        if self.clce_loss is not None and clce_weight > 0:
            total = total + clce_weight * clce_loss
        return {"sup_loss": sup_loss, "consist_loss": consist_loss, "total_loss": total}
