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

    nnUNetTrainer.__init__ sets ``self.ce_weight = None``, then calls
    ``_build_loss()`` which passes it to ``DC_and_CE_loss``.  We set
    ``self.ce_weight`` **before** ``super().__init__()`` so that
    ``_build_loss`` picks up our per-class weights.

    Must use the **exact same signature** as nnUNetTrainer.__init__
    (no ``**kwargs``) because the parent inspects ``locals()``.
    """

    def __init__(
        self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device("cuda")
    ):
        num_fg = len(dataset_json.get("labels", {})) - 1
        if num_fg > 1:
            self.ce_weight = torch.tensor([0.1, 1.0, 5.0], dtype=torch.float32)
        else:
            self.ce_weight = torch.tensor([0.1, 5.0], dtype=torch.float32)
        super().__init__(plans, configuration, fold, dataset_json, device=device)
