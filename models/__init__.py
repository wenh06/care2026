"""
Model wrappers for CARE2026.

Loss is computed inside forward() when labels are provided (MBAS2024 pattern).
The Trainer reads out_tensors["total_loss"] and calls loss.backward().
"""

import re
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file as _load_sft
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import CitationMixin
from torch_ecg.utils.utils_nn import CkptMixin, SizeMixin

from cfg import (
    CT_TrainCfg,
    CT_TrainCfg_MT_nnUNet,
    CT_TrainCfg_nnUNet,
    CT_TrainCfgV2,
    ModelCfg,
    MRI_Stage1_TrainCfg,
    MRI_Stage2_TrainCfg,
)
from const import CT_NUM_CLASSES
from outputs import CARE2026Outputs

from .loss import CTLoss, ScarLoss, Stage1MRILoss
from .vnet import VNet

__all__ = [
    "CARE2026_MRI_Stage1_Model",
    "CARE2026_MRI_Stage2_Model",
    "CARE2026_CT_Model",
    "CARE2026_CT_ModelV2",
    "CARE2026_CT_nnUNet",
    "CARE2026_CT_MT_nnUNet",
]


def _resolve_nnunet_dir(model_dir: Union[str, Path]) -> Path:
    """Auto-discover nnUNet trainer directory and available folds.

    Accepts any level of the nnUNet results tree::

        results/                                          # results root
        results/Dataset500/                               # dataset level
        results/Dataset500/nnUNetTrainer__.../            # trainer level

    Returns ``(trainer_dir, folds, ckpt_path)``.
    """
    model_dir = Path(model_dir).expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"nnUNet directory not found: {model_dir}")

    # Walk down to find the trainer dir (contains plans.json)
    if not (model_dir / "plans.json").exists():
        # Try nnUNetTrainer__* subdir
        trainers = sorted(model_dir.glob("nnUNetTrainer__*"))
        if trainers:
            model_dir = trainers[0]
        elif not (model_dir / "plans.json").exists():
            # Try one level deeper: Dataset*/nnUNetTrainer__*
            for ds_dir in sorted(model_dir.glob("Dataset*")):
                sub_trainers = sorted(ds_dir.glob("nnUNetTrainer__*"))
                if sub_trainers:
                    model_dir = sub_trainers[0]
                    break
            else:
                raise FileNotFoundError(f"Cannot find nnUNet trainer dir (plans.json) under {model_dir}")

    if not (model_dir / "plans.json").exists():
        raise FileNotFoundError(f"plans.json not found in {model_dir}")

    # Auto-detect available folds
    fold_dirs = sorted(model_dir.glob("fold_*"))
    folds = tuple(int(f.name.split("_")[1]) for f in fold_dirs if f.is_dir())

    # Auto-detect a checkpoint
    ckpt_path = None
    for ckpt_name in ("checkpoint_best.pth", "checkpoint_final.pth", "checkpoint_latest.pth"):
        for search_dir in ([fold_dirs[0]] if fold_dirs else []) + [model_dir]:
            candidate = search_dir / ckpt_name
            if candidate.exists():
                ckpt_path = candidate
                break
        if ckpt_path:
            break

    return model_dir, folds, ckpt_path


class CARE2026_MRI_Stage1_Model(nn.Module, SizeMixin, CkptMixin, CitationMixin):
    """Single-head VNet for coarse LA cavity localisation (Stage 1).

    Operates on the full volume downsampled to MRI_STAGE1_SHAPE (144x144x44).
    Predicts a binary LA mask used to locate the LA centroid and to constrain
    scar predictions at Stage 2.
    """

    __name__ = "CARE2026_MRI_Stage1_Model"

    def __init__(
        self,
        config: Optional[CFG] = None,
        train_config: Optional[CFG] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.__config = deepcopy(ModelCfg)
        if config is not None:
            self.__config.update(deepcopy(config))
        self.__train_config = deepcopy(MRI_Stage1_TrainCfg)
        if train_config is not None:
            self.__train_config.update(deepcopy(train_config))
        # Sync inference-relevant fields into config so they are
        # serialised in model_config metadata and available at load time.
        for key in ("canonical_shape", "patch_shape", "apply_mclahe"):
            if key in self.__train_config and key not in self.__config:
                self.__config[key] = self.__train_config[key]
        self.backbone = VNet(self.config.vnet_stage1)
        self.criterion = Stage1MRILoss(self.train_config)

    def forward(self, img: torch.Tensor, labels: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        img = img.to(device=self.device, dtype=self.dtype)
        la_logits = self.backbone(img)
        output = {"la_logits": la_logits, "la_mask": la_logits.argmax(dim=1)}
        if labels is not None:
            loss_dict = self.criterion(la_logits=la_logits, la_target=labels["la_mask"].to(self.device))
            output.update(loss_dict)
        return output

    @torch.no_grad()
    def inference(self, img: Union[np.ndarray, torch.Tensor]) -> CARE2026Outputs:
        original_mode = self.training
        self.eval()
        input_t = self._prepare_input(img)
        output = self.forward(input_t)
        self.train(original_mode)
        return CARE2026Outputs(task="mri", la_mask=output["la_mask"].cpu().numpy().astype(np.uint8))

    def _prepare_input(self, img: Union[np.ndarray, torch.Tensor, list]) -> torch.Tensor:
        if isinstance(img, (list, tuple)):
            img = torch.stack([i if isinstance(i, torch.Tensor) else torch.from_numpy(i) for i in img])
        elif isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        img = img.to(device=self.device, dtype=self.dtype)
        while img.ndim < 5:
            img = img.unsqueeze(0)
        img = (img - img.mean(dim=(2, 3, 4), keepdim=True)) / (img.std(dim=(2, 3, 4), keepdim=True) + 1e-8)
        return img

    @property
    def config(self) -> CFG:
        return self.__config

    @property
    def train_config(self) -> CFG:
        return self.__train_config

    def save(self, path, **kwargs):
        p = Path(str(path))
        name = re.sub(r"(?<=\d)\.(?=\d)", "_", p.name)
        path = str(p.parent / name)
        return super().save(path=path, **kwargs)


class CARE2026_MRI_Stage2_Model(nn.Module, SizeMixin, CkptMixin, CitationMixin):
    """Single-head VNet for scar segmentation (Stage 2).

    Stage 1 already provides the LA cavity mask.  This model focuses the
    entire network capacity on scar.  Trained on 128x128x44 patches with
    ScarLoss (Gaussian spatial weighting).  At inference the predicted
    scar is constrained to a dilated Stage-1 cavity mask (~2 mm).
    """

    __name__ = "CARE2026_MRI_Stage2_Model"

    def __init__(
        self,
        config: Optional[CFG] = None,
        train_config: Optional[CFG] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.__config = deepcopy(ModelCfg)
        if config is not None:
            self.__config.update(deepcopy(config))
        self.__train_config = deepcopy(MRI_Stage2_TrainCfg)
        if train_config is not None:
            self.__train_config.update(deepcopy(train_config))
        # Sync inference-relevant fields into config (do not overwrite).
        for key in ("canonical_shape", "patch_shape", "train_crop_hw", "apply_mclahe"):
            if key in self.__train_config and key not in self.__config:
                self.__config[key] = self.__train_config[key]
        # Single-head VNet or NestedVNet, scar only (2 classes: bg + scar)
        backbone = str(self.__config.get("backbone") or self.__train_config.get("backbone", "vnet_stage2"))
        _mri_backbone_map = {
            "vnet": "vnet_stage2",
            "nested_vnet": "nested_vnet_stage2",
            "vnet_l": "vnet_stage2_l",
            "nested_vnet_l": "nested_vnet_stage2_l",
            "vnet_2ch_l": "vnet_stage2_2ch_l",
        }
        backbone = _mri_backbone_map.get(backbone, backbone)
        self.__config["backbone"] = backbone
        if backbone.startswith("nested"):
            from .nested_vnet import NestedVNet

            self.backbone = NestedVNet(self.config.get(backbone, self.config.nested_vnet_stage2))
            self._is_nested = True
        else:
            self.backbone = VNet(self.config.get(backbone, self.config.vnet_stage2))
            self._is_nested = False
        self.criterion = ScarLoss(self.train_config)

    def forward(self, img: torch.Tensor, labels: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        img = img.to(device=self.device, dtype=self.dtype)
        scar_logits = self.backbone(img)
        # For NestedVNet with deep supervision: upsample each level to the
        # full-resolution space, then average for loss computation.
        if self._is_nested and isinstance(scar_logits, (list, tuple)):
            full_logits = []
            tgt_spatial = scar_logits[-1].shape[2:]
            for lo in scar_logits:
                if lo.shape[2:] != tgt_spatial:
                    lo = nn.functional.interpolate(lo, size=tgt_spatial, mode="trilinear", align_corners=False)
                full_logits.append(lo)
            logits_for_loss = sum(full_logits) / len(full_logits)
            logits_for_mask = scar_logits[-1]
        else:
            logits_for_loss = scar_logits
            logits_for_mask = scar_logits
        output = {"scar_logits": scar_logits, "scar_mask": logits_for_mask.argmax(dim=1)}
        if labels is not None:
            loss_dict = self.criterion(
                scar_logits=logits_for_loss,
                scar_target=labels["scar_mask"].to(self.device),
                has_scar=labels["has_scar"].to(self.device),
            )
            output.update(loss_dict)
        return output

    @torch.no_grad()
    def inference(self, img: Union[np.ndarray, torch.Tensor]) -> CARE2026Outputs:
        original_mode = self.training
        self.eval()
        input_t = self._prepare_input(img)
        output = self.forward(input_t)
        self.train(original_mode)
        scar_mask = output["scar_mask"]
        if isinstance(scar_mask, (list, tuple)):
            scar_mask = scar_mask[-1]
        return CARE2026Outputs(task="mri", scar_mask=scar_mask.cpu().numpy().astype(np.uint8))

    def _prepare_input(self, img: Union[np.ndarray, torch.Tensor, list]) -> torch.Tensor:
        if isinstance(img, (list, tuple)):
            img = torch.stack([i if isinstance(i, torch.Tensor) else torch.from_numpy(i) for i in img])
        elif isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        img = img.to(device=self.device, dtype=self.dtype)
        while img.ndim < 5:
            img = img.unsqueeze(0)
        img = (img - img.mean(dim=(2, 3, 4), keepdim=True)) / (img.std(dim=(2, 3, 4), keepdim=True) + 1e-8)
        return img

    @property
    def config(self) -> CFG:
        return self.__config

    @property
    def train_config(self) -> CFG:
        return self.__train_config

    def save(self, path, **kwargs):
        p = Path(str(path))
        name = re.sub(r"(?<=\d)\.(?=\d)", "_", p.name)
        path = str(p.parent / name)
        return super().save(path=path, **kwargs)


class CARE2026_CT_Model(nn.Module, SizeMixin, CkptMixin, CitationMixin):
    """Semi-supervised CT model (Task 3).

    Supports two modes configured via ``CT_TrainCfg.semi_supervised_mode``:

    - **"cps"**: Cross Pseudo Supervision — two independent VNets
      cross-supervise each other via pseudo-labels.
    - **"mean_teacher"**: Mean Teacher (Tarvainen & Valpola, NeurIPS
      2017) — a single student VNet is supervised by labeled data and
      by the consistency between its predictions and those of an EMA
      teacher model on unlabeled data.
    """

    __name__ = "CARE2026_CT_Model"

    def __init__(self, config: Optional[CFG] = None, train_config: Optional[CFG] = None, **kwargs: Any) -> None:
        super().__init__()
        self.__config = deepcopy(ModelCfg)
        if config is not None:
            self.__config.update(deepcopy(config))
        self.__train_config = deepcopy(CT_TrainCfg)
        if train_config is not None:
            self.__train_config.update(deepcopy(train_config))

        # Sync inference-relevant fields into config (do not overwrite).
        for key in ("patch_size", "normalization", "semi_supervised_mode"):
            if key in self.__train_config and key not in self.__config:
                self.__config[key] = self.__train_config[key]

        # Read mode from config first (populated from safetensors metadata at
        # load time), falling back to train_config.  Store in config so that
        # CkptMixin.save() serialises it in model_config metadata, which
        # CkptMixin.from_checkpoint() passes to __init__ as ``config``.
        self.mode = self.__config.get("semi_supervised_mode") or self.__train_config.get("semi_supervised_mode", "cps")
        self.__config["semi_supervised_mode"] = self.mode
        if self.mode not in ("cps", "mean_teacher", "supervised"):
            raise ValueError(f"Unknown semi_supervised_mode: {self.mode}")

        backbone = str(self.__config.get("backbone") or self.__train_config.get("backbone", "vnet_ct"))
        _ct_backbone_map = {"vnet": "vnet_ct", "nested_vnet": "nested_vnet_ct"}
        backbone = _ct_backbone_map.get(backbone, backbone)
        self.__config["backbone"] = backbone  # persist in model_config
        if backbone.startswith("nested"):
            from .nested_vnet import NestedVNet

            model_cfg = self.config.get(backbone, self.config.nested_vnet_ct)
            self._make_model = lambda: NestedVNet(model_cfg)
        else:
            model_cfg = self.config.get(backbone, self.config.vnet_ct)
            self._make_model = lambda: VNet(model_cfg)
        self._backbone = backbone
        self._is_nested = backbone.startswith("nested")

        self.model1 = self._make_model()
        # Load pretrained encoder weights (shape-matched; strict=False)
        pretrained = self.__train_config.get("pretrained_encoder", None)
        if pretrained:
            pretrained_sd = _load_sft(pretrained, device="cpu")
            if any(k.startswith("model1.") for k in pretrained_sd):
                pretrained_sd = {k[len("model1.") :]: v for k, v in pretrained_sd.items() if k.startswith("model1.")}
            self.model1.load_state_dict(pretrained_sd, strict=False)
        if self.mode == "cps":
            self.model2 = self._make_model()
        elif self.mode == "mean_teacher":
            self.teacher = self._make_model()
            self.teacher.load_state_dict(self.model1.state_dict())
            for p in self.teacher.parameters():
                p.requires_grad = False
        self.criterion = CTLoss(self.train_config)
        self._mt_decay = float(self.__train_config.get("mt_ema_decay", 0.99))

    def _update_teacher(self) -> None:
        """EMA update: θ_t ← α·θ_t + (1−α)·θ_s (applied in train mode)."""
        if not hasattr(self, "teacher"):
            return
        alpha = self._mt_decay
        with torch.no_grad():
            for tp, sp in zip(self.teacher.parameters(), self.model1.parameters()):
                tp.data.mul_(alpha).add_(sp.data, alpha=1.0 - alpha)

    def _ds_last(self, logits):
        """Return the last (full-resolution) output from a NestedVNet list."""
        return logits[-1] if isinstance(logits, (list, tuple)) else logits

    def forward(
        self,
        img: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        cps_weight: float = 1.0,
        clce_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        img = img.to(device=self.device, dtype=self.dtype)
        if self.mode == "cps":
            logits1, logits2 = self.model1(img), self.model2(img)
            seg_mask = self._ds_last((self._ds_last(logits1) + self._ds_last(logits2)) / 2).argmax(dim=1)
            output = {"logits1": logits1, "logits2": logits2, "seg_mask": seg_mask}
        elif self.mode == "mean_teacher":
            logits_s = self.model1(img)
            seg_mask = self._ds_last(logits_s).argmax(dim=1)
            output = {"logits1": logits_s, "seg_mask": seg_mask}
            if self.training:
                with torch.no_grad():
                    logits_t = self.teacher(img)
                    output["logits_t"] = logits_t
        else:
            # supervised: single model
            logits_s = self.model1(img)
            seg_mask = self._ds_last(logits_s).argmax(dim=1)
            output = {"logits1": logits_s, "seg_mask": seg_mask}

        if labels is not None:
            target = labels.get("ct_mask")
            labeled_mask = labels.get("labeled")
            if target is not None:
                target = target.to(self.device)
            if labeled_mask is not None:
                labeled_mask = labeled_mask.to(self.device)

            logits1 = output.get("logits1")
            logits2 = output.get("logits2")
            logits_t = output.get("logits_t") if self.mode == "mean_teacher" else None

            if self._is_nested:
                # Deep supervision: upsample each level to target spatial size,
                # then average loss across all levels.
                total_loss = logits1[0].sum() * 0.0
                sup_losses, consist_losses = [], []
                tgt_spatial = target.shape[1:]  # (H, W, D) — target has no channel dim
                for level, (l1,) in enumerate(zip(logits1)):
                    if l1.shape[2:] != tgt_spatial:
                        l1 = nn.functional.interpolate(l1, size=tgt_spatial, mode="trilinear", align_corners=False)
                    l2_lev = logits2[level] if logits2 is not None else None
                    if l2_lev is not None and l2_lev.shape[2:] != tgt_spatial:
                        l2_lev = nn.functional.interpolate(l2_lev, size=tgt_spatial, mode="trilinear", align_corners=False)
                    lt_lev = logits_t[level] if logits_t is not None else None
                    if lt_lev is not None and lt_lev.shape[2:] != tgt_spatial:
                        lt_lev = nn.functional.interpolate(lt_lev, size=tgt_spatial, mode="trilinear", align_corners=False)
                    ld = self.criterion(
                        logits1=l1,
                        logits2=l2_lev,
                        logits_t=lt_lev,
                        target=target,
                        labeled_mask=labeled_mask,
                        cps_weight=cps_weight,
                        clce_weight=clce_weight,
                    )
                    sup_losses.append(ld["sup_loss"])
                    consist_losses.append(ld["consist_loss"])
                    total_loss = total_loss + ld["total_loss"]
                loss_dict = {
                    "sup_loss": sum(sup_losses) / len(sup_losses),
                    "consist_loss": sum(consist_losses) / len(consist_losses),
                    "total_loss": total_loss / len(logits1),
                }
            else:
                loss_dict = self.criterion(
                    logits1=logits1,
                    logits2=logits2,
                    logits_t=logits_t,
                    target=target,
                    labeled_mask=labeled_mask,
                    cps_weight=cps_weight,
                    clce_weight=clce_weight,
                )
            output.update(loss_dict)
        return output

    @torch.no_grad()
    def inference(self, img: Union[np.ndarray, torch.Tensor]) -> CARE2026Outputs:
        original_mode = self.training
        self.eval()
        input_t = self._prepare_input(img)
        output = self.forward(input_t)
        self.train(original_mode)
        return CARE2026Outputs(task="ct", ct_mask=output["seg_mask"].cpu().numpy().astype(np.uint8))

    def _prepare_input(self, img: Union[np.ndarray, torch.Tensor, list]) -> torch.Tensor:
        if isinstance(img, (list, tuple)):
            img = torch.stack([i if isinstance(i, torch.Tensor) else torch.from_numpy(i) for i in img])
        elif isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        img = img.to(device=self.device, dtype=self.dtype)
        while img.ndim < 5:
            img = img.unsqueeze(0)
        return img

    @property
    def config(self) -> CFG:
        return self.__config

    @property
    def train_config(self) -> CFG:
        return self.__train_config

    def save(self, path, **kwargs):
        p = Path(str(path))
        name = re.sub(r"(?<=\d)\.(?=\d)", "_", p.name)
        path = str(p.parent / name)
        # Strip teacher/model2 to halve checkpoint size, and mark config
        # as supervised so from_checkpoint(strict=True) doesn't expect them.
        saved_mode = self.__config.get("semi_supervised_mode")
        restored = {}
        for attr in ("teacher", "model2"):
            if hasattr(self, attr):
                restored[attr] = getattr(self, attr)
                delattr(self, attr)
        if restored:
            self.__config["semi_supervised_mode"] = "supervised"
        try:
            return super().save(path=path, **kwargs)
        finally:
            for attr, m in restored.items():
                setattr(self, attr, m)
            if saved_mode is not None:
                self.__config["semi_supervised_mode"] = saved_mode


class CARE2026_CT_ModelV2(nn.Module, SizeMixin, CkptMixin, CitationMixin):
    """CT segmentation model V2 (Task 3) — supervised-first design.

    Fixes three root causes of poor CT performance vs. the original
    :class:`CARE2026_CT_Model`:

    1. **InstanceNorm** replaces BatchNorm — batch-size independent,
       critical when training 3-D models with batch_size ≤ 4.
    2. **Mish activation** replaces ReLU — smoother gradients, better
       convergence on small medical-imaging datasets.
    3. **AdamW** replaces SGD — adaptive per-parameter learning rates
       handle the noisy gradients from small-batch 3-D training.

    Supports three modes via ``semi_supervised_mode``:
      - ``"supervised"`` (default) — single VNet, labelled data only.
      - ``"cps"`` — Cross Pseudo Supervision with two parallel VNets.
      - ``"mean_teacher"`` — Mean Teacher with EMA teacher.

    The recommendation is to start with ``"supervised"`` on the 50
    labelled CTs, establish a strong baseline, then optionally enable
    semi-supervised modes.
    """

    __name__ = "CARE2026_CT_ModelV2"

    def __init__(self, config: Optional[CFG] = None, train_config: Optional[CFG] = None, **kwargs: Any) -> None:
        super().__init__()
        self.__config = deepcopy(ModelCfg)
        if config is not None:
            self.__config.update(deepcopy(config))
        self.__train_config = deepcopy(CT_TrainCfgV2)
        if train_config is not None:
            self.__train_config.update(deepcopy(train_config))

        # Sync inference-relevant fields into config (serialised in metadata).
        for key in ("patch_size", "normalization", "semi_supervised_mode"):
            if key in self.__train_config and key not in self.__config:
                self.__config[key] = self.__train_config[key]

        self.mode = self.__config.get("semi_supervised_mode") or self.__train_config.get("semi_supervised_mode", "supervised")
        self.__config["semi_supervised_mode"] = self.mode
        if self.mode not in ("supervised", "cps", "mean_teacher"):
            raise ValueError(f"Unknown semi_supervised_mode: {self.mode}")

        # Resolve backbone: "vnet_ct_v2" (default) or "nested_vnet_ct_v2"
        backbone = str(self.__config.get("backbone", self.__train_config.get("backbone", "vnet_ct_v2")))
        _ct_backbone_map_v2 = {"vnet": "vnet_ct_v2", "nested_vnet": "nested_vnet_ct_v2"}
        backbone = _ct_backbone_map_v2.get(backbone, backbone)
        self.__config["backbone"] = backbone
        if backbone.startswith("nested"):
            from .nested_vnet import NestedVNet

            model_cfg = (
                self.config.nested_vnet_ct_v2 if hasattr(self.config, "nested_vnet_ct_v2") else self.config.nested_vnet_ct
            )
            self._make_model = lambda: NestedVNet(model_cfg)
        else:
            model_cfg = self.config.vnet_ct_v2 if hasattr(self.config, "vnet_ct_v2") else self.config.vnet_ct
            self._make_model = lambda: VNet(model_cfg)
        self._backbone = backbone
        self._is_nested = backbone.startswith("nested")

        self.model1 = self._make_model()
        pretrained = self.__train_config.get("pretrained_encoder", None)
        if pretrained:
            pretrained_sd = _load_sft(pretrained, device="cpu")
            if any(k.startswith("model1.") for k in pretrained_sd):
                pretrained_sd = {k[len("model1.") :]: v for k, v in pretrained_sd.items() if k.startswith("model1.")}
            self.model1.load_state_dict(pretrained_sd, strict=False)
        if self.mode == "cps":
            self.model2 = self._make_model()
        elif self.mode == "mean_teacher":
            self.teacher = self._make_model()
            self.teacher.load_state_dict(self.model1.state_dict())
            for p in self.teacher.parameters():
                p.requires_grad = False
        self.criterion = CTLoss(self.train_config)
        self._mt_decay = float(self.__train_config.get("mt_ema_decay", 0.99))

    # ------------------------------------------------------------------
    # Teacher EMA update (Mean Teacher mode only)
    # ------------------------------------------------------------------

    def _update_teacher(self) -> None:
        """EMA update: θ_t ← α·θ_t + (1−α)·θ_s."""
        if not hasattr(self, "teacher"):
            return
        alpha = self._mt_decay
        with torch.no_grad():
            for tp, sp in zip(self.teacher.parameters(), self.model1.parameters()):
                tp.data.mul_(alpha).add_(sp.data, alpha=1.0 - alpha)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _ds_last(self, logits):
        """Return the last (full-resolution) output from a NestedVNet list."""
        return logits[-1] if isinstance(logits, (list, tuple)) else logits

    def forward(
        self,
        img: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        cps_weight: float = 1.0,
        clce_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        img = img.to(device=self.device, dtype=self.dtype)

        if self.mode == "cps":
            logits1, logits2 = self.model1(img), self.model2(img)
            seg_mask = self._ds_last((self._ds_last(logits1) + self._ds_last(logits2)) / 2).argmax(dim=1)
            output = {"logits1": logits1, "logits2": logits2, "seg_mask": seg_mask}
        elif self.mode == "mean_teacher":
            logits_s = self.model1(img)
            seg_mask = self._ds_last(logits_s).argmax(dim=1)
            output = {"logits1": logits_s, "seg_mask": seg_mask}
            if self.training:
                with torch.no_grad():
                    logits_t = self.teacher(img)
                    output["logits_t"] = logits_t
        else:
            # Supervised-only: single VNet, no teacher, no CPS
            logits_s = self.model1(img)
            seg_mask = self._ds_last(logits_s).argmax(dim=1)
            output = {"logits1": logits_s, "seg_mask": seg_mask}

        if labels is not None:
            target = labels.get("ct_mask")
            labeled_mask = labels.get("labeled")
            if target is not None:
                target = target.to(self.device)
            if labeled_mask is not None:
                labeled_mask = labeled_mask.to(self.device)

            logits1 = output.get("logits1")
            logits2 = output.get("logits2")
            logits_t = output.get("logits_t") if self.mode == "mean_teacher" else None

            if self._is_nested:
                # Deep supervision: average loss across decoder levels
                tgt_spatial = target.shape[1:]  # (H, W, D)
                total_loss = logits1[0].sum() * 0.0
                sup_losses, consist_losses = [], []
                num_levels = len(logits1)
                for level in range(num_levels):
                    l1 = logits1[level]
                    if l1.shape[2:] != tgt_spatial:
                        l1 = nn.functional.interpolate(l1, size=tgt_spatial, mode="trilinear", align_corners=False)
                    l2_lev = logits2[level] if logits2 is not None else None
                    if l2_lev is not None and l2_lev.shape[2:] != tgt_spatial:
                        l2_lev = nn.functional.interpolate(l2_lev, size=tgt_spatial, mode="trilinear", align_corners=False)
                    lt_lev = logits_t[level] if logits_t is not None else None
                    if lt_lev is not None and lt_lev.shape[2:] != tgt_spatial:
                        lt_lev = nn.functional.interpolate(lt_lev, size=tgt_spatial, mode="trilinear", align_corners=False)
                    ld = self.criterion(
                        logits1=l1,
                        logits2=l2_lev,
                        logits_t=lt_lev,
                        target=target,
                        labeled_mask=labeled_mask,
                        cps_weight=cps_weight,
                    )
                    sup_losses.append(ld["sup_loss"])
                    consist_losses.append(ld["consist_loss"])
                    total_loss = total_loss + ld["total_loss"]
                loss_dict = {
                    "sup_loss": sum(sup_losses) / num_levels,
                    "consist_loss": sum(consist_losses) / num_levels,
                    "total_loss": total_loss / num_levels,
                }
            else:
                loss_dict = self.criterion(
                    logits1=logits1,
                    logits2=logits2,
                    logits_t=logits_t,
                    target=target,
                    labeled_mask=labeled_mask,
                    cps_weight=cps_weight,
                    clce_weight=clce_weight,
                )
            output.update(loss_dict)
        return output

    @torch.no_grad()
    def inference(self, img: Union[np.ndarray, torch.Tensor]) -> CARE2026Outputs:
        original_state = self.training
        self.eval()
        input_t = self._prepare_input(img)
        output = self.forward(input_t)
        self.train(original_state)
        return CARE2026Outputs(task="ct", ct_mask=output["seg_mask"].cpu().numpy().astype(np.uint8))

    def _prepare_input(self, img: Union[np.ndarray, torch.Tensor, list]) -> torch.Tensor:
        if isinstance(img, (list, tuple)):
            img = torch.stack([i if isinstance(i, torch.Tensor) else torch.from_numpy(i) for i in img])
        elif isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        img = img.to(device=self.device, dtype=self.dtype)
        while img.ndim < 5:
            img = img.unsqueeze(0)
        return img

    @property
    def config(self) -> CFG:
        return self.__config

    @property
    def train_config(self) -> CFG:
        return self.__train_config

    def save(self, path, **kwargs):
        p = Path(str(path))
        name = re.sub(r"(?<=\d)\.(?=\d)", "_", p.name)
        path = str(p.parent / name)
        return super().save(path=path, **kwargs)


class CARE2026_CT_nnUNet(nn.Module, SizeMixin, CkptMixin, CitationMixin):
    """CT model using nnUNet's full inference pipeline (Task 3).

    Wraps ``nnUNetPredictor`` for preprocessing, normalization, sliding
    window, and resampling — exactly matching the nnUNet training recipe.
    """

    __name__ = "CARE2026_CT_nnUNet"

    def __init__(
        self,
        config: Optional[CFG] = None,
        train_config: Optional[CFG] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.__config = deepcopy(ModelCfg)
        if config is not None:
            self.__config.update(deepcopy(config))
        self.__train_config = deepcopy(CT_TrainCfg_nnUNet)
        if train_config is not None:
            self.__train_config.update(deepcopy(train_config))

        for key in ("patch_shape", "normalization", "target_spacing", "arch_config"):
            if key in self.__train_config and key not in self.__config:
                self.__config[key] = deepcopy(self.__train_config[key])
        self.__config["backbone"] = "nnunet"

        # ── Resolve nnUNet directory ─────────────────────────────────
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        model_dir = self.__train_config.get("nnunet_model_dir", None)
        if model_dir is None:
            raise ValueError("CT_TrainCfg_nnUNet.nnunet_model_dir must be set.")

        trainer_dir, auto_folds, auto_ckpt = _resolve_nnunet_dir(model_dir)
        folds = self.__train_config.get("nnunet_folds") or auto_folds or (0,)
        if isinstance(folds, str):
            folds = tuple(int(f.strip()) for f in folds.split(","))
        if isinstance(folds, list):
            folds = tuple(folds)
        ckpt_name = self.__train_config.get("nnunet_checkpoint") or auto_ckpt.name if auto_ckpt else "checkpoint_best.pth"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__config["nnunet_trainer_dir"] = str(trainer_dir)
        self.__config["nnunet_folds"] = folds

        self._predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        self._predictor.initialize_from_trained_model_folder(
            str(trainer_dir),
            use_folds=folds,
            checkpoint_name=ckpt_name,
        )
        # Expose the network as backbone for compatibility
        self.backbone = self._predictor.network

        self.criterion = CTLoss(self.train_config)

    # ------------------------------------------------------------------
    # Forward (patch-based, for training compatibility)
    # ------------------------------------------------------------------

    def forward(
        self,
        img: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        cps_weight: float = 1.0,
        clce_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        img = img.to(device=self.device, dtype=self.dtype)
        logits = self.backbone(img)
        if isinstance(logits, (list, tuple)):
            seg_mask = logits[0].argmax(dim=1)
        else:
            seg_mask = logits.argmax(dim=1)
        output: Dict[str, torch.Tensor] = {"logits1": logits, "seg_mask": seg_mask}

        if labels is not None:
            target = labels.get("ct_mask")
            if target is not None:
                target = target.to(self.device)
            if isinstance(logits, (list, tuple)):
                tgt_spatial = target.shape[1:]
                total_loss = logits[0].sum() * 0.0
                num_levels = len(logits)
                sup_losses = []
                for level in range(num_levels):
                    l1 = logits[level]
                    if l1.shape[2:] != tgt_spatial:
                        l1 = nn.functional.interpolate(l1, size=tgt_spatial, mode="trilinear", align_corners=False)
                    ld = self.criterion(
                        logits1=l1,
                        logits2=None,
                        logits_t=None,
                        target=target,
                        labeled_mask=None,
                        cps_weight=0.0,
                        clce_weight=clce_weight,
                    )
                    sup_losses.append(ld["sup_loss"])
                    total_loss = total_loss + ld["total_loss"]
                output.update({"sup_loss": sum(sup_losses) / num_levels, "total_loss": total_loss / num_levels})
            else:
                loss_dict = self.criterion(
                    logits1=logits,
                    logits2=None,
                    logits_t=None,
                    target=target,
                    labeled_mask=None,
                    cps_weight=0.0,
                    clce_weight=clce_weight,
                )
                output.update(loss_dict)
        return output

    # ------------------------------------------------------------------
    # Full-volume inference (uses nnUNetPredictor for correctness)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, image_npy: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
        """Run full-volume prediction using nnUNet's pipeline.

        Parameters
        ----------
        image_npy : np.ndarray, shape (H, W, D), float32
            Raw CT volume in original voxel space.
        spacing : (float, float, float)
            Voxel spacing in mm (x, y, z).

        Returns
        -------
        np.ndarray, shape (H, W, D), uint8
            Predicted multi-class mask (0-3).
        """
        self._predictor.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # nnUNet's NibabelIOWithReorient transposes nibabel's (x,y,z)
        # to SimpleITK's (z,y,x) ordering.  We must do the same for the
        # raw numpy array and reverse the transpose on the output.
        if image_npy.ndim == 3:
            image_npy = np.transpose(image_npy, (2, 1, 0))  # (x,y,z) → (z,y,x)
        spacing_nnunet = (float(spacing[2]), float(spacing[1]), float(spacing[0]))
        ret = self._predictor.predict_from_list_of_npy_arrays(
            image_or_list_of_images=image_npy[None].astype(np.float32),
            segs_from_prev_stage_or_list_of_segs_from_prev_stage=None,
            properties_or_list_of_properties={"spacing": spacing_nnunet},
            truncated_ofname=None,
            num_processes=1,
            save_probabilities=False,
            num_processes_segmentation_export=1,
        )
        pred = ret[0].astype(np.uint8)
        # Reverse the transpose: (z,y,x) → (x,y,z)
        pred = np.transpose(pred, (2, 1, 0))
        return pred

    # ------------------------------------------------------------------
    # Standard helpers (interface compatibility)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inference(self, img: Union[np.ndarray, torch.Tensor]) -> CARE2026Outputs:
        raise NotImplementedError("Use predict() or predict_ct() for nnUNet models.")

    def _prepare_input(self, img: Union[np.ndarray, torch.Tensor, list]) -> torch.Tensor:
        if isinstance(img, (list, tuple)):
            img = torch.stack([i if isinstance(i, torch.Tensor) else torch.from_numpy(i) for i in img])
        elif isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        return img.to(device=self.device, dtype=self.dtype).unsqueeze(0).unsqueeze(0) if img.ndim < 5 else img

    @property
    def config(self) -> CFG:
        return self.__config

    @property
    def train_config(self) -> CFG:
        return self.__train_config

    def save(self, path, **kwargs):
        p = Path(str(path))
        name = re.sub(r"(?<=\d)\.(?=\d)", "_", p.name)
        path = str(p.parent / name)
        return super().save(path=path, **kwargs)


class CARE2026_CT_MT_nnUNet(nn.Module, SizeMixin, CkptMixin, CitationMixin):
    """Mean Teacher CT model with nnUNet PlainConvUNet backbone."""

    __name__ = "CARE2026_CT_MT_nnUNet"

    def __init__(
        self,
        config: Optional[CFG] = None,
        train_config: Optional[CFG] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.__config = deepcopy(ModelCfg)
        if config is not None:
            self.__config.update(deepcopy(config))
        self.__train_config = deepcopy(CT_TrainCfg_MT_nnUNet)
        if train_config is not None:
            self.__train_config.update(deepcopy(train_config))

        for key in ("patch_shape", "normalization", "target_spacing", "arch_config", "semi_supervised_mode"):
            if key in self.__train_config and key not in self.__config:
                self.__config[key] = deepcopy(self.__train_config[key])
        self.__config["backbone"] = "nnunet_mt"

        self.mode = self.__config.get("semi_supervised_mode") or self.__train_config.get("semi_supervised_mode", "mean_teacher")
        self.__config["semi_supervised_mode"] = self.mode

        from dynamic_network_architectures.architectures.unet import PlainConvUNet

        arch_cfg = deepcopy(self.__config.get("arch_config", self.__train_config.get("arch_config", {})))

        def _make_nnunet():
            return PlainConvUNet(
                input_channels=1,
                n_stages=int(arch_cfg["n_stages"]),
                features_per_stage=list(arch_cfg["features_per_stage"]),
                conv_op=nn.Conv3d,
                kernel_sizes=[list(k) for k in arch_cfg["kernel_sizes"]],
                strides=[list(s) for s in arch_cfg["strides"]],
                n_conv_per_stage=list(arch_cfg["n_conv_per_stage"]),
                n_conv_per_stage_decoder=list(arch_cfg["n_conv_per_stage_decoder"]),
                num_classes=CT_NUM_CLASSES,
                conv_bias=bool(arch_cfg.get("conv_bias", True)),
                norm_op=nn.InstanceNorm3d,
                norm_op_kwargs=dict(arch_cfg.get("norm_op_kwargs", {"eps": 1e-5, "affine": True})),
                dropout_op=None,
                nonlin=nn.LeakyReLU,
                nonlin_kwargs=dict(arch_cfg.get("nonlin_kwargs", {"inplace": True})),
                deep_supervision=True,
            )

        self.model1 = _make_nnunet()

        pretrained = self.__train_config.get("pretrained_encoder", None)
        if pretrained:
            pretrained_path = Path(pretrained)
            if pretrained_path.is_dir():
                _, _, auto_ckpt = _resolve_nnunet_dir(pretrained_path)
                if auto_ckpt:
                    pretrained_path = auto_ckpt
            if pretrained_path.exists():
                ckpt = torch.load(str(pretrained_path), map_location="cpu", weights_only=False)
                weights = ckpt["network_weights"]
                if any(k.startswith("module.") for k in weights):
                    weights = OrderedDict((k[7:], v) for k, v in weights.items())
                self.model1.load_state_dict(weights, strict=True)

        if self.mode == "mean_teacher":
            self.teacher = _make_nnunet()
            self.teacher.load_state_dict(self.model1.state_dict())
            for p in self.teacher.parameters():
                p.requires_grad = False

        self.criterion = CTLoss(self.train_config)
        self._mt_decay = float(self.__train_config.get("mt_ema_decay", 0.99))

    def _update_teacher(self) -> None:
        alpha = self._mt_decay
        with torch.no_grad():
            for tp, sp in zip(self.teacher.parameters(), self.model1.parameters()):
                tp.data.mul_(alpha).add_(sp.data, alpha=1.0 - alpha)

    def _ds_last(self, logits):
        return logits[0] if isinstance(logits, (list, tuple)) else logits

    def forward(
        self,
        img: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        cps_weight: float = 1.0,
        clce_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        img = img.to(device=self.device, dtype=self.dtype)
        logits_s = self.model1(img)
        seg_mask = self._ds_last(logits_s).argmax(dim=1)
        output: Dict[str, torch.Tensor] = {"logits1": logits_s, "seg_mask": seg_mask}

        if self.training and self.mode == "mean_teacher":
            with torch.no_grad():
                output["logits_t"] = self.teacher(img)

        if labels is not None:
            target = labels.get("ct_mask")
            labeled_mask = labels.get("labeled")
            if target is not None:
                target = target.to(self.device)
            if labeled_mask is not None:
                labeled_mask = labeled_mask.to(self.device)
            # Pass raw logits — CTLoss handles deep supervision per-level
            loss_dict = self.criterion(
                logits1=logits_s,
                logits2=None,
                logits_t=output.get("logits_t"),
                target=target,
                labeled_mask=labeled_mask,
                cps_weight=cps_weight,
                clce_weight=clce_weight,
            )
            output.update(loss_dict)
        return output

    @torch.no_grad()
    def inference(self, img: Union[np.ndarray, torch.Tensor]) -> CARE2026Outputs:
        original_state = self.training
        self.eval()
        input_t = self._prepare_input(img)
        output = self.forward(input_t)
        self.train(original_state)
        return CARE2026Outputs(task="ct", ct_mask=output["seg_mask"].cpu().numpy().astype(np.uint8))

    def _prepare_input(self, img: Union[np.ndarray, torch.Tensor, list]) -> torch.Tensor:
        if isinstance(img, (list, tuple)):
            img = torch.stack([i if isinstance(i, torch.Tensor) else torch.from_numpy(i) for i in img])
        elif isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        img = img.to(device=self.device, dtype=self.dtype)
        while img.ndim < 5:
            img = img.unsqueeze(0)
        return img

    @property
    def config(self) -> CFG:
        return self.__config

    @property
    def train_config(self) -> CFG:
        return self.__train_config

    def save(self, path, **kwargs):
        p = Path(str(path))
        name = re.sub(r"(?<=\d)\.(?=\d)", "_", p.name)
        # PlainConvUNet shares tensors encoder↔decoder; force .pth to use
        # torch.save (safetensors errors on shared memory tensors).
        path = str(p.parent / name) + ".pth"
        restored = {}
        if hasattr(self, "teacher"):
            restored["teacher"] = self.teacher
            del self.teacher
            self.__config["semi_supervised_mode"] = "supervised"
        try:
            return super().save(path=path, train_config=self.train_config, **kwargs)
        finally:
            for attr, m in restored.items():
                setattr(self, attr, m)
            self.__config["semi_supervised_mode"] = self.mode
