"""Custom nnUNet trainers for CARE2026 experiments.

Usage::

    export nnUNet_extTrainer="$PWD/models"
    nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainerScarWeighted
    nnUNetv2_train 521 3d_fullres 0 -tr nnUNetTrainerScarWeighted

``nnUNet_extTrainer`` points to the directory containing this file.
nnUNet recursively scans all ``.py`` files for the requested class name.
"""

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


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
