"""
Model wrappers for CARE2026.

Loss is computed inside forward() when labels are provided (MBAS2024 pattern).
The Trainer reads out_tensors["total_loss"] and calls loss.backward().
"""

import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import CitationMixin
from torch_ecg.utils.utils_nn import CkptMixin, SizeMixin

from cfg import CT_TrainCfg, ModelCfg, MRI_Stage1_TrainCfg, MRI_Stage2_TrainCfg
from outputs import CARE2026Outputs

from .loss import CTLoss, ScarLoss, Stage1MRILoss
from .vnet import VNet

__all__ = [
    "CARE2026_MRI_Stage1_Model",
    "CARE2026_MRI_Stage2_Model",
    "CARE2026_CT_Model",
]


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
        # Single-head VNet, scar only (2 classes: bg + scar)
        self.backbone = VNet(self.config.vnet_stage2)
        self.criterion = ScarLoss(self.train_config)

    def forward(self, img: torch.Tensor, labels: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        img = img.to(device=self.device, dtype=self.dtype)
        scar_logits = self.backbone(img)
        output = {"scar_logits": scar_logits, "scar_mask": scar_logits.argmax(dim=1)}
        if labels is not None:
            loss_dict = self.criterion(
                scar_logits=scar_logits,
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
        return CARE2026Outputs(task="mri", scar_mask=output["scar_mask"].cpu().numpy().astype(np.uint8))

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

        self.mode = self.__train_config.get("semi_supervised_mode", "cps")
        if self.mode not in ("cps", "mean_teacher"):
            raise ValueError(f"Unknown semi_supervised_mode: {self.mode}")

        self.model1 = VNet(self.config.vnet_ct)
        if self.mode == "cps":
            self.model2 = VNet(self.config.vnet_ct)
        else:
            # Mean Teacher: EMA teacher model (no grad)
            self.teacher = VNet(self.config.vnet_ct)
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

    def forward(
        self, img: torch.Tensor, labels: Optional[Dict[str, torch.Tensor]] = None, cps_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        img = img.to(device=self.device, dtype=self.dtype)
        if self.mode == "cps":
            logits1, logits2 = self.model1(img), self.model2(img)
            seg_mask = ((logits1 + logits2) / 2).argmax(dim=1)
            output = {"logits1": logits1, "logits2": logits2, "seg_mask": seg_mask}
        else:
            logits_s = self.model1(img)
            seg_mask = logits_s.argmax(dim=1)
            output = {"logits1": logits_s, "seg_mask": seg_mask}
            if self.training:
                with torch.no_grad():
                    logits_t = self.teacher(img)
                    output["logits_t"] = logits_t

        if labels is not None:
            target = labels.get("ct_mask")
            labeled_mask = labels.get("labeled")
            if target is not None:
                target = target.to(self.device)
            if labeled_mask is not None:
                labeled_mask = labeled_mask.to(self.device)
            loss_dict = self.criterion(
                logits1=output.get("logits1"),
                logits2=output.get("logits2"),
                logits_t=output.get("logits_t") if self.mode == "mean_teacher" else None,
                target=target,
                labeled_mask=labeled_mask,
                cps_weight=cps_weight,
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
        return super().save(path=path, **kwargs)
