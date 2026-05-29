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

from cfg import CT_TrainCfg, ModelCfg, MRI_Stage1_TrainCfg, MRI_Stage2_TrainCfg, MRI_TrainCfg
from outputs import CARE2026Outputs

from .loss import CTLoss, MRILoss, Stage1MRILoss
from .nested_vnet import DualHeadNestedVNet
from .vnet import DualHeadVNet, VNet

__all__ = [
    "CARE2026_MRI_Stage1_Model",
    "CARE2026_MRI_Stage2_Model",
    "CARE2026_MRI_Model",       # alias for Stage2
    "CARE2026_CT_Model",
]


class CARE2026_MRI_Stage1_Model(nn.Module, SizeMixin, CkptMixin, CitationMixin):
    """Wrapper for single-head VNet for coarse LA localisation (Stage 1).

    Operates on the full volume downsampled to MRI_STAGE1_SHAPE (144×144×44).
    Predicts a binary LA mask used only to locate the LA centroid for Stage 2.

    Parameters
    ----------
    config : CFG, optional
        Model config overrides (merged on top of ``ModelCfg.vnet_stage1``).
    train_config : CFG, optional
        Training config for loss weights (merged on top of MRI_Stage1_TrainCfg).
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

        # Single-head VNet (binary LA only): use the vnet_stage1 config
        self.backbone = VNet(self.config.vnet_stage1)
        self.criterion = Stage1MRILoss(self.train_config)

    def forward(
        self,
        img: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        img : torch.Tensor, shape (B, 1, H, W, D)
        labels : dict, optional
            Training labels:
            - "la_mask": (B, H, W, D) long

        Returns
        -------
        dict
            Always contains: la_logits, la_mask.
            When labels provided: la_loss, total_loss.
        """
        img = img.to(device=self.device, dtype=self.dtype)
        la_logits = self.backbone(img)
        la_mask = la_logits.argmax(dim=1)

        output = {"la_logits": la_logits, "la_mask": la_mask}

        if labels is not None:
            la_target = labels["la_mask"].to(self.device)
            loss_dict = self.criterion(la_logits=la_logits, la_target=la_target)
            output.update(loss_dict)

        return output

    @torch.no_grad()
    def inference(self, img: Union[np.ndarray, torch.Tensor]) -> CARE2026Outputs:
        """Run inference on a single or batch of coarse MRI volumes."""
        original_mode = self.training
        self.eval()
        input_t = self._prepare_input(img)
        output = self.forward(input_t)
        self.train(original_mode)
        return CARE2026Outputs(
            task="mri",
            la_mask=output["la_mask"].cpu().numpy().astype(np.uint8),
        )

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

    def save(self, path, **kwargs) -> None:
        """Override CkptMixin.save to prevent decimal suffixes from being treated as file extensions."""
        p = Path(str(path))
        name = re.sub(r"(?<=\d)\.(?=\d)", "_", p.name)
        path = str(p.parent / name)
        super().save(path=path, **kwargs)


class CARE2026_MRI_Stage2_Model(nn.Module, SizeMixin, CkptMixin, CitationMixin):
    """Wrapper for dual-head VNet (or NestedVNet) for MRI Tasks 1 & 2 (Stage 2).

    Loss is computed inside forward() when labels are provided.
    Architecture: shared encoder → la_decoder + scar_decoder.
    Instance Norm for domain generalization (Task 2).

    Parameters
    ----------
    config : CFG, optional
        Model config overrides (merged on top of ModelCfg).
    train_config : CFG, optional
        Training config for loss weights (merged on top of MRI_Stage2_TrainCfg).
    backbone : str, default "vnet"
        ``"vnet"``        — :class:`~.vnet.DualHeadVNet`
        ``"nested_vnet"`` — :class:`~.nested_vnet.DualHeadNestedVNet`
    """

    __name__ = "CARE2026_MRI_Stage2_Model"

    def __init__(
        self,
        config: Optional[CFG] = None,
        train_config: Optional[CFG] = None,
        backbone: str = "vnet",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.__config = deepcopy(ModelCfg)
        if config is not None:
            self.__config.update(deepcopy(config))
        self.__train_config = deepcopy(MRI_Stage2_TrainCfg)
        if train_config is not None:
            self.__train_config.update(deepcopy(train_config))

        if backbone == "nested_vnet":
            self.backbone = DualHeadNestedVNet(self.config.nested_vnet)
        else:
            self.backbone = DualHeadVNet(self.config.vnet)
        self.criterion = MRILoss(self.train_config)

    def forward(
        self,
        img: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        img : torch.Tensor, shape (B, 1, H, W, D)
        labels : dict, optional
            Training labels:
            - "la_mask": (B, H, W, D) long
            - "scar_mask": (B, H, W, D) long (zeros where has_scar=False)
            - "has_scar": (B,) bool

        Returns
        -------
        dict
            Always contains: la_logits, scar_logits, la_mask, scar_mask.
            When labels provided: la_loss, scar_loss, total_loss.
        """
        img = img.to(device=self.device, dtype=self.dtype)
        la_out, scar_out = self.backbone(img)

        # Deep supervision: la_out / scar_out may be lists (coarse→fine).
        # Use the finest resolution for prediction; average loss over all levels.
        if isinstance(la_out, list):
            la_logits = la_out[-1]
            scar_logits = scar_out[-1]  # type: ignore[index]
        else:
            la_logits = la_out
            scar_logits = scar_out  # type: ignore[assignment]

        la_mask = la_logits.argmax(dim=1)
        scar_mask = scar_logits.argmax(dim=1)

        output = {
            "la_logits": la_logits,
            "scar_logits": scar_logits,
            "la_mask": la_mask,
            "scar_mask": scar_mask,
        }

        if labels is not None:
            la_target = labels["la_mask"].to(self.device)
            scar_target = labels["scar_mask"].to(self.device)
            has_scar = labels["has_scar"].to(self.device)

            if isinstance(la_out, list):
                # Average loss across deep-supervision levels
                loss_dicts = [
                    self.criterion(
                        la_logits=la_l,
                        scar_logits=sc_l,
                        la_target=la_target,
                        scar_target=scar_target,
                        has_scar=has_scar,
                    )
                    for la_l, sc_l in zip(la_out, scar_out)  # type: ignore[arg-type]
                ]
                n = len(loss_dicts)
                loss_dict = {k: sum(d[k] for d in loss_dicts) / n for k in loss_dicts[0]}
            else:
                loss_dict = self.criterion(
                    la_logits=la_logits,
                    scar_logits=scar_logits,
                    la_target=la_target,
                    scar_target=scar_target,
                    has_scar=has_scar,
                )
            output.update(loss_dict)

        return output

    @torch.no_grad()
    def inference(self, img: Union[np.ndarray, torch.Tensor]) -> CARE2026Outputs:
        """Run inference on a single or batch of MRI volumes."""
        original_mode = self.training
        self.eval()
        input_t = self._prepare_input(img)
        output = self.forward(input_t)
        self.train(original_mode)
        return CARE2026Outputs(
            task="mri",
            la_mask=output["la_mask"].cpu().numpy().astype(np.uint8),
            scar_mask=output["scar_mask"].cpu().numpy().astype(np.uint8),
        )

    def _prepare_input(self, img: Union[np.ndarray, torch.Tensor, list]) -> torch.Tensor:
        if isinstance(img, (list, tuple)):
            img = torch.stack([i if isinstance(i, torch.Tensor) else torch.from_numpy(i) for i in img])
        elif isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        img = img.to(device=self.device, dtype=self.dtype)
        while img.ndim < 5:
            img = img.unsqueeze(0)
        # sample-wise z-score normalisation
        img = (img - img.mean(dim=(2, 3, 4), keepdim=True)) / (img.std(dim=(2, 3, 4), keepdim=True) + 1e-8)
        return img

    @property
    def config(self) -> CFG:
        return self.__config

    @property
    def train_config(self) -> CFG:
        return self.__train_config

    def save(self, path, **kwargs) -> None:
        """Override CkptMixin.save to prevent decimal suffixes from being treated as file extensions.

        e.g. ``...epochloss_0.17121_metric_0.91`` → saves as
        ``...epochloss_0_17121_metric_0_91.safetensors``
        instead of the broken ``...epochloss_0.safetensors``.
        """
        p = Path(str(path))
        name = re.sub(r"(?<=\d)\.(?=\d)", "_", p.name)
        path = str(p.parent / name)
        super().save(path=path, **kwargs)


# Backward-compatibility alias
CARE2026_MRI_Model = CARE2026_MRI_Stage2_Model


class CARE2026_CT_Model(nn.Module, SizeMixin, CkptMixin, CitationMixin):
    """Wrapper for Cross Pseudo Supervision (CPS) CT model (Task 3).

    Contains two independent UNet3D instances for semi-supervised learning.
    Loss (supervised DiceCE + CPS pseudo-label CE) is computed in forward().

    Parameters
    ----------
    config : CFG, optional
        Model config overrides (merged on top of ModelCfg).
    train_config : CFG, optional
        Training config overrides (merged on top of CT_TrainCfg).
    """

    __name__ = "CARE2026_CT_Model"

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
        self.__train_config = deepcopy(CT_TrainCfg)
        if train_config is not None:
            self.__train_config.update(deepcopy(train_config))

        self.model1 = VNet(self.config.vnet_ct)
        self.model2 = VNet(self.config.vnet_ct)
        self.criterion = CTLoss(self.train_config)

    def forward(
        self,
        img: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        cps_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass (CPS).

        Parameters
        ----------
        img : torch.Tensor, shape (B, 1, H, W, D)
        labels : dict, optional
            - "ct_mask": (B, H, W, D) long
            - "labeled": (B,) bool tensor
        cps_weight : float, default 1.0
            Ramp-up factor (0→1 over first N epochs).

        Returns
        -------
        dict
            Always: logits1, logits2, seg_mask.
            When labels provided: sup_loss, cps_loss, total_loss.
        """
        img = img.to(device=self.device, dtype=self.dtype)
        logits1 = self.model1(img)
        logits2 = self.model2(img)
        seg_mask = ((logits1 + logits2) / 2).argmax(dim=1)

        output = {"logits1": logits1, "logits2": logits2, "seg_mask": seg_mask}

        if labels is not None:
            target = labels.get("ct_mask")
            labeled_mask = labels.get("labeled")
            if target is not None:
                target = target.to(self.device)
            if labeled_mask is not None:
                labeled_mask = labeled_mask.to(self.device)
            loss_dict = self.criterion(
                logits1=logits1,
                logits2=logits2,
                target=target,
                labeled_mask=labeled_mask,
                cps_weight=cps_weight,
            )
            output.update(loss_dict)

        return output

    @torch.no_grad()
    def inference(self, img: Union[np.ndarray, torch.Tensor]) -> CARE2026Outputs:
        """Run inference on a single or batch of CT volumes."""
        original_mode = self.training
        self.eval()
        input_t = self._prepare_input(img)
        output = self.forward(input_t)
        self.train(original_mode)
        return CARE2026Outputs(
            task="ct",
            ct_mask=output["seg_mask"].cpu().numpy().astype(np.uint8),
        )

    def _prepare_input(self, img: Union[np.ndarray, torch.Tensor, list]) -> torch.Tensor:
        if isinstance(img, (list, tuple)):
            img = torch.stack([i if isinstance(i, torch.Tensor) else torch.from_numpy(i) for i in img])
        elif isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        img = img.to(device=self.device, dtype=self.dtype)
        while img.ndim < 5:
            img = img.unsqueeze(0)
        return img  # CT already HU-normalized in Dataset

    @property
    def config(self) -> CFG:
        return self.__config

    @property
    def train_config(self) -> CFG:
        return self.__train_config

    def save(self, path, **kwargs) -> None:
        """Override CkptMixin.save to prevent decimal suffixes from being treated as file extensions."""
        p = Path(str(path))
        name = re.sub(r"(?<=\d)\.(?=\d)", "_", p.name)
        path = str(p.parent / name)
        super().save(path=path, **kwargs)
