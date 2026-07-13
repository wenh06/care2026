"""Custom nnUNet trainers for CARE2026 experiments.

Usage::

    export nnUNet_extTrainer="$PWD/models"
    nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainerScarWeighted
    nnUNetv2_train 521 3d_fullres 0 -tr nnUNetTrainerScarWeighted

``nnUNet_extTrainer`` points to the directory containing this file.
nnUNet recursively scans all ``.py`` files for the requested class name.
"""

import numpy as np
import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from scipy.ndimage import distance_transform_edt
from torch.nn import CrossEntropyLoss


class nnUNetTrainerScarWeighted(nnUNetTrainer):
    """nnUNet with per-class CE weights for scar segmentation.

    Reads ``dataset_json["labels"]`` to find the scar class index and
    assigns: bg = 0.1, scar = 5.0, other fg classes = 1.0.

    nnUNetTrainer sets ``self.ce_weight = None`` then calls
    ``_build_loss()`` which passes it to ``DC_and_CE_loss``.
    We set ``self.ce_weight`` BEFORE ``super().__init__()``.
    """

    def __init__(
        self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device("cuda")
    ):
        labels = dataset_json.get("labels", {})
        n_classes = len(labels)
        weights = [0.1] * n_classes  # bg default
        for name, idx in labels.items():
            idx = int(idx)
            if idx == 0:
                continue
            if "scar" in name.lower():
                weights[idx] = 5.0
            else:
                weights[idx] = 1.0
        self.ce_weight = torch.tensor(weights, dtype=torch.float32)
        super().__init__(plans, configuration, fold, dataset_json, device=device)


class nnUNetTrainerScarGaussian(nnUNetTrainer):
    """nnUNet with Gaussian spatial weighting for scar segmentation.

    Computes a pixel-wise weight map w(x) = 1 + w₀·exp(−d²/2σ²) where
    d is the distance to the nearest scar voxel.  Works for both binary
    (Dataset 501) and multi-class (Dataset 521) labels.

    The distance transform runs on CPU (scipy) per batch — ~0.1s overhead
    for a 256×256×44 patch, negligible relative to forward/backward pass.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        self._w0 = 5.0
        self._sigma_mm = 2.0
        # Find scar class index from label map
        labels = dataset_json.get("labels", {})
        self._scar_cls = 1  # fallback: last non-bg class
        for name, idx in labels.items():
            if "scar" in name.lower():
                self._scar_cls = int(idx)
                break
        super().__init__(plans, configuration, fold, dataset_json, device=device)

    def _build_loss(self):
        loss = _ScarGaussianLoss(
            scar_class=self._scar_cls,
            w0=self._w0,
            sigma_mm=self._sigma_mm,
            batch_dice=self.configuration_manager.batch_dice,
            ignore_label=self.label_manager.ignore_label,
        )
        if self._do_i_compile():
            loss.dice = torch.compile(loss.dice)
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class _ScarGaussianLoss(torch.nn.Module):
    """Dice + pixel-weighted CE with Gaussian spatial weighting.

    Same weight map formula as ``models.loss.ScarLoss``:
    w(x) = 1 + w₀·exp(−d(x)²/2σ²) where d(x) = distance to nearest scar voxel.
    Adapted for nnUNet interface: ``forward(pred, target)`` instead of
    ``forward(scar_logits, scar_target, has_scar)``.
    Uses ``MemoryEfficientSoftDiceLoss`` (nnUNet) instead of ``DiceCELoss``.
    """

    def __init__(
        self,
        scar_class: int,
        w0: float = 5.0,
        sigma_mm: float = 2.0,
        spacing_xy: float = 0.625,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        batch_dice: bool = False,
        ignore_label: int | None = None,
    ):
        super().__init__()
        self.scar_class = scar_class
        self.w0 = w0
        self.sigma_px = max(1.0, sigma_mm / spacing_xy)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ignore_label = ignore_label
        self.dice = MemoryEfficientSoftDiceLoss(apply_nonlin=torch.nn.Softmax(dim=1), batch_dice=batch_dice, smooth=1e-5)
        self.ce = CrossEntropyLoss(reduction="none", ignore_index=ignore_label if ignore_label is not None else -100)

    def _compute_weight(self, tgt: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian spatial weight map from scar mask (CPU, once per batch)."""
        scar_bin = (tgt == self.scar_class).cpu().numpy().astype("uint8")
        batch_w = []
        for b in range(scar_bin.shape[0]):
            sb = scar_bin[b]
            if sb.sum() == 0:
                batch_w.append(np.ones(sb.shape, dtype=np.float32))
            else:
                d = distance_transform_edt(1 - sb).astype(np.float32)
                w = 1.0 + self.w0 * np.exp(-(d**2) / (2 * self.sigma_px**2))
                batch_w.append(w.astype(np.float32))
        return torch.from_numpy(np.stack(batch_w))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Dice
        dice_loss = self.dice(pred, target)

        # Pixel-weighted CE
        # target may be (B, 1, H, W, D) → squeeze
        tgt = target.long()
        if tgt.ndim == 5 and tgt.shape[1] == 1:
            tgt = tgt[:, 0]

        # Cache weight map: DeepSupervisionWrapper calls forward() once per
        # decoder level with the same target — compute the DT only once.
        tgt_ptr = tgt.data_ptr()
        if tgt_ptr != getattr(self, "_cached_tgt_ptr", None):
            self._cached_weight = self._compute_weight(tgt)
            self._cached_tgt_ptr = tgt_ptr
        w_tensor = self._cached_weight.to(pred.device)

        ce_pixel = self.ce(pred, tgt)  # (B, H, W, D)
        ce_loss = (ce_pixel * w_tensor).mean()

        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


# ---------------------------------------------------------------------------
# CT Boundary-aware Trainer
# ---------------------------------------------------------------------------


class nnUNetTrainerCTBoundary(nnUNetTrainer):
    """nnUNet with boundary + topology loss for CT multi-structure segmentation.

    Adds Hausdorff boundary loss (GPU-friendly erosion-based) and
    centerline CE loss for thin tubular structures (PV).  Designed to
    narrow the gap on PV and LAA where standard DiceCE underperforms.

    Usage::

        export nnUNet_extTrainer="$PWD/models"
        nnUNetv2_train 500 3d_fullres 0 -tr nnUNetTrainerCTBoundary
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        self._hd_weight = 0.1
        self._clce_weight = 0.3
        self._hd_erosions = 10
        super().__init__(plans, configuration, fold, dataset_json, device=device)

    def _build_loss(self):
        loss = _CTBoundaryLoss(
            hd_weight=self._hd_weight,
            clce_weight=self._clce_weight,
            hd_erosions=self._hd_erosions,
            batch_dice=self.configuration_manager.batch_dice,
            ignore_label=self.label_manager.ignore_label,
        )
        if self._do_i_compile():
            loss.clce.dice = torch.compile(loss.clce.dice)
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class _CTBoundaryLoss(torch.nn.Module):
    """DiceCE + HausdorffER + CenterlineCE for CT multi-class segmentation.

    L = L_clce (Dice + λ_clce·skeleton_CE) + L_ce + w_hd·L_hd

    - ``CenterlineCELoss`` handles topology of thin PV through soft-skeleton
      cross-entropy (already includes an internal Dice term).
    - ``HausdorffERLoss`` penalises boundary misalignment for PV and LAA
      using GPU-friendly iterative erosion.
    - Standard CE provides per-pixel supervision.
    """

    def __init__(
        self,
        hd_weight: float = 0.1,
        clce_weight: float = 0.3,
        hd_erosions: int = 10,
        batch_dice: bool = False,
        ignore_label: int | None = None,
    ):
        super().__init__()
        from models.loss.boundary_loss import HausdorffERLoss
        from models.loss.centerline_loss import CenterlineCELoss

        self.hd_weight = hd_weight
        self.ignore_label = ignore_label

        # CenterlineCELoss includes internal Dice → covers Dice + topology
        self.clce = CenterlineCELoss(kernel_size=5, lambda_clce=clce_weight, do_bg=False)
        # Standard CE
        self.ce = CrossEntropyLoss(reduction="mean", ignore_index=ignore_label if ignore_label is not None else -100)
        # GPU-friendly Hausdorff (no scipy → works fully on GPU)
        self.hd = HausdorffERLoss(alpha=2.0, erosions=hd_erosions, do_bg=False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # target may be (B, 1, H, W, D) → squeeze
        tgt = target.long()
        if tgt.ndim == 5 and tgt.shape[1] == 1:
            tgt = tgt[:, 0]

        clce_loss = self.clce(pred, tgt)  # internal: Dice + λ·skeleton_CE
        ce_loss = self.ce(pred, tgt)
        hd_loss = self.hd(pred, tgt)

        return clce_loss + ce_loss + self.hd_weight * hd_loss
