"""Trainer classes for the CARE 2026 Left Atrium challenge."""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader, Dataset
from torch_ecg.cfg import CFG
from torch_ecg.components.trainer import BaseTrainer
from torch_ecg.utils.misc import str2bool
from tqdm.auto import tqdm

from cfg import (
    CT_TrainCfg,
    CT_TrainCfg_MT_nnUNet,
    CT_TrainCfg_nnUNet,
    CT_TrainCfgV2,
    ModelCfg,
    MRI_Stage1_TrainCfg,
    MRI_Stage2_TrainCfg,
)
from dataset import (
    CARE2026_CT_Dataset,
    CARE2026_MRI_Stage1_Dataset,
    CARE2026_MRI_Stage2_Dataset,
    collate_fn_ct,
    collate_fn_mri,
    collate_fn_mri_stage1,
)
from models import (
    CARE2026_CT_Model,
    CARE2026_CT_ModelV2,
    CARE2026_CT_MT_nnUNet,
    CARE2026_CT_nnUNet,
    CARE2026_MRI_Stage1_Model,
    CARE2026_MRI_Stage2_Model,
)

__all__ = [
    "CARE2026_MRI_Stage1_Trainer",
    "CARE2026_MRI_Stage2_Trainer",
    "CARE2026_CT_Trainer",
]


def _binary_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    pred_sum = pred.sum()
    target_sum = target.sum()
    if pred_sum == 0 and target_sum == 0:
        return 1.0
    inter = np.logical_and(pred, target).sum()
    return float((2.0 * inter + eps) / (pred_sum + target_sum + eps))


def _binary_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float((pred == target).mean())


def _binary_sensitivity(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fn = np.logical_and(~pred, target).sum()
    if target.sum() == 0:
        return 1.0
    return float((tp + eps) / (tp + fn + eps))


def _sanitize_ckpt_path(path) -> Path:
    """Convert a checkpoint stem path to the actual .safetensors file path.

    torch_ecg BaseTrainer stores paths like ``...epochloss_0.17121_metric_0.91``
    in ``saved_models``, but CkptMixin.save() treats decimal parts as file
    extensions.  We instead save with dots replaced by underscores (e.g.
    ``...epochloss_0_17121_metric_0_91.safetensors``) via the model's save()
    override; this helper applies the same normalisation so cleanup works.
    """
    p = Path(str(path))
    # Replace every digit.digit sequence in the filename with digit_digit
    name = re.sub(r"(?<=\d)\.(?=\d)", "_", p.name)
    return (p.parent / name).with_suffix(".safetensors")


class _FixedPathDeque(deque):
    """A deque that transparently normalises checkpoint paths on append.

    The base trainer appends the *stem* path (e.g. ``checkpoints/…metric_0.91``)
    to ``saved_models``.  We intercept that to store the *actual* file path
    (``checkpoints/…metric_0_91.safetensors``) so that the base trainer's
    ``os.remove(model_to_remove)`` call succeeds.
    """

    def append(self, item) -> None:
        super().append(_sanitize_ckpt_path(item))


class _BaseCARE2026Trainer(BaseTrainer):
    """Shared training utilities for CARE2026 MRI/CT trainers."""

    __DEBUG__ = True

    def __init__(
        self,
        model: nn.Module,
        dataset_cls: Dataset,
        collate_fn,
        model_config: dict,
        train_config: dict,
        device: Optional[torch.device] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> None:
        tc = CFG(deepcopy(train_config))
        if "learning_rate" not in tc and "lr" in tc:
            tc.learning_rate = tc.lr
        if "checkpoints" not in tc:
            tc.checkpoints = tc.get("model_dir")
        super().__init__(
            model=model,
            dataset_cls=dataset_cls,
            model_config=model_config,
            train_config=tc,
            collate_fn=collate_fn,
            device=device,
            lazy=lazy,
            **kwargs,
        )

        # Replace the base trainer's saved_models deque with our path-normalising version
        # so that keep_checkpoint_max cleanup (os.remove) targets the correct .safetensors file.
        self.saved_models = _FixedPathDeque()

        # AMP (automatic mixed precision)
        use_amp = bool(tc.get("use_amp", True)) and torch.cuda.is_available()
        self._use_amp = use_amp
        self._scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # Gradient accumulation: accumulate over N micro-batches before stepping
        self._accum_steps = max(1, int(tc.get("accumulate_grad_batches", 1)))
        self._accum_counter = 0

    @property
    def batch_dim(self) -> int:
        return 0

    @property
    def extra_required_train_config_fields(self) -> List[str]:
        return ["db_dir"]

    def train(self):
        if self.train_loader is None:
            self._setup_dataloaders()
        return super().train()

    def _setup_scheduler(self) -> None:
        scheduler = str(self.train_config.lr_scheduler).lower()
        if scheduler in {"none", ""}:
            self.scheduler = None
            self.train_config.lr_scheduler = "none"
            return

        # Gradient accumulation means the optimizer (and scheduler) step only
        # every ``accum_steps`` batches.  Compute total scheduler steps correctly.
        batches_per_epoch = max(1, len(self.train_loader))
        steps_per_epoch = max(1, (batches_per_epoch + self._accum_steps - 1) // self._accum_steps)
        total_steps = max(1, self.n_epochs * steps_per_epoch)
        if scheduler in {"cosine", "cosine_annealing", "cosineannealing"}:
            self.train_config.lr_scheduler = "cosine"
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=float(self.train_config.get("lr_min", 0.0)),
            )
            return
        if scheduler == "poly":
            self.train_config.lr_scheduler = "poly"
            power = float(self.train_config.get("lr_poly_power", 0.9))
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: max(0.0, 1.0 - step / total_steps) ** power,
            )
            return
        super()._setup_scheduler()

    def _update_lr(self, eval_res: Optional[dict] = None) -> None:
        scheduler = str(self.train_config.lr_scheduler).lower()
        if scheduler in {"cosine", "poly"}:
            if self.scheduler is not None:
                self.scheduler.step()
            return
        super()._update_lr(eval_res)

    def _setup_criterion(self) -> None:
        # loss is encapsulated in the model wrappers
        pass

    def train_one_epoch(self, pbar: tqdm) -> None:
        self._accum_counter = 0
        self.optimizer.zero_grad()
        for input_tensors in self.train_loader:
            self.global_step += 1
            n_samples = input_tensors["image"].shape[self.batch_dim]
            self._accum_counter += 1

            if self._use_amp:
                with torch.amp.autocast("cuda"):
                    out_tensors = self.run_one_step(input_tensors)
                loss = out_tensors["total_loss"].mean() / self._accum_steps
                self._scaler.scale(loss).backward()
            else:
                out_tensors = self.run_one_step(input_tensors)
                loss = out_tensors["total_loss"].mean() / self._accum_steps
                loss.backward()

            # Scale loss back for logging (undo the accumulation divisor)
            loss_for_log = loss.item() * self._accum_steps
            self.epoch_loss += loss_for_log

            if self._accum_counter % self._accum_steps == 0:
                if self._use_amp:
                    self._scaler.step(self.optimizer)
                    self._scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self._update_lr()
                # Mean Teacher EMA update (no-op for non-MT models)
                if hasattr(self._model, "_update_teacher"):
                    self._model._update_teacher()

            if self.global_step % self.train_config.log_step == 0:
                step_metrics = {"loss": loss_for_log}
                for key in ["la_loss", "scar_loss", "sup_loss", "cps_loss", "consist_loss"]:
                    if key in out_tensors:
                        value = out_tensors[key]
                        if isinstance(value, torch.Tensor):
                            step_metrics[key] = value.mean().item()
                if hasattr(self, "_current_cps_weight"):
                    step_metrics["cps_weight"] = self._current_cps_weight
                if self.scheduler is not None:
                    step_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
                    pbar.set_postfix(loss=loss_for_log, lr=self.optimizer.param_groups[0]["lr"])
                else:
                    pbar.set_postfix(loss=loss_for_log)
                self.log_manager.log_metrics(
                    metrics=step_metrics,
                    step=self.global_step,
                    epoch=self.epoch,
                    part="train",
                )
            pbar.update(n_samples)


class CARE2026_MRI_Stage1_Trainer(_BaseCARE2026Trainer):
    """Trainer for MRI Stage 1 coarse LA localisation."""

    __name__ = "CARE2026_MRI_Stage1_Trainer"

    def __init__(
        self,
        model: nn.Module,
        model_config: dict,
        train_config: dict,
        device: Optional[torch.device] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> None:
        tc = CFG(deepcopy(MRI_Stage1_TrainCfg))
        tc.update(deepcopy(train_config))
        tc.classes = ["la_cavity"]
        tc.monitor = tc.get("monitor", "la_dice")
        super().__init__(
            model=model,
            dataset_cls=CARE2026_MRI_Stage1_Dataset,
            collate_fn=collate_fn_mri_stage1,
            model_config=model_config,
            train_config=tc,
            device=device,
            lazy=lazy,
            **kwargs,
        )

    @property
    def save_prefix(self) -> str:
        model_name = getattr(self._model, "__name__", self._model.__class__.__name__)
        return f"{model_name}-mri1"

    def extra_log_suffix(self) -> str:
        return f"mri1_{self.train_config.optimizer}"

    def _setup_dataloaders(
        self,
        train_dataset=None,
        val_dataset=None,
    ) -> None:
        num_workers = 1 if self.device == torch.device("cpu") else 4
        db_dir = self.train_config.db_dir
        val_r = float(self.train_config.get("val_ratio", 0.1))
        seed = int(self.train_config.get("random_seed", 42))
        if val_r <= 0:
            self.train_config.monitor = None
        if train_dataset is None:
            train_dataset = CARE2026_MRI_Stage1_Dataset(
                db_dir=db_dir,
                config=self.train_config,
                training=True,
                val_ratio=val_r,
                random_seed=seed,
            )
        if val_r <= 0:
            self.val_loader = None
        else:
            if val_dataset is None:
                val_dataset = CARE2026_MRI_Stage1_Dataset(
                    db_dir=db_dir,
                    config=self.train_config,
                    training=False,
                    val_ratio=val_r,
                    random_seed=seed,
                )
            self.val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
                drop_last=False,
                collate_fn=collate_fn_mri_stage1,
            )
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn_mri_stage1,
        )
        self.val_train_loader = self.train_loader if bool(self.train_config.get("debug", True)) else None

    def run_one_step(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(
            img=input_tensors["image"],
            labels={"la_mask": input_tensors["la_mask"].long()},
        )

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        original_state = self.model.training
        self.model.eval()
        la_dices: List[float] = []
        with tqdm(
            total=len(data_loader.dataset),
            desc="Evaluation (Stage1)",
            unit="vol",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=False,
        ) as pbar:
            for input_tensors in data_loader:
                out = self.model(img=input_tensors["image"])
                pred_la = out["la_mask"].detach().cpu().numpy()
                gt_la = input_tensors["la_mask"].numpy().astype(np.uint8)
                for i in range(pred_la.shape[0]):
                    la_dices.append(_binary_dice(pred_la[i], gt_la[i]))
                pbar.update(pred_la.shape[0])
        self.model.train(original_state)
        return {"la_dice": float(np.mean(la_dices)) if la_dices else 0.0}


class CARE2026_CT_Trainer(_BaseCARE2026Trainer):
    """Trainer for CT Task 3 (semi-supervised CPS)."""

    __name__ = "CARE2026_CT_Trainer"

    def __init__(
        self,
        model: nn.Module,
        model_config: dict,
        train_config: dict,
        device: Optional[torch.device] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> None:
        tc = CFG(deepcopy(CT_TrainCfg))
        tc.update(deepcopy(train_config))
        tc.classes = ["background", "la", "pv", "laa"]
        tc.monitor = tc.get("monitor", "ct_mean_dice")
        super().__init__(
            model=model,
            dataset_cls=CARE2026_CT_Dataset,
            collate_fn=collate_fn_ct,
            model_config=model_config,
            train_config=tc,
            device=device,
            lazy=lazy,
            **kwargs,
        )

    @property
    def save_prefix(self) -> str:
        model_name = getattr(self._model, "__name__", self._model.__class__.__name__)
        return f"{model_name}-ct"

    def extra_log_suffix(self) -> str:
        return f"ct_{self.train_config.optimizer}"

    def _setup_dataloaders(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
    ) -> None:
        num_workers = 1 if self.device == torch.device("cpu") else 4
        db_dir = self.train_config.db_dir
        semi_mode = str(self.train_config.get("semi_supervised_mode", "cps"))
        warmup = int(self.train_config.get("mt_warmup_epochs", 0))
        val_r = float(self.train_config.get("val_ratio", 0.1))
        seed = int(self.train_config.get("random_seed", 42))

        # No model selection when val set is empty (val_ratio=0).
        # Base trainer saves the last model at the end of training.
        if val_r <= 0:
            self.train_config.monitor = None

        # Warmup mode: start with labeled-only, switch to all after warmup.
        # Avoids wasting compute on unlabeled data while consistency loss is 0.
        if warmup > 0 and semi_mode in ("cps", "mean_teacher"):
            self._train_loader_labeled = self._make_ct_loader(
                db_dir,
                labeled=True,
                training=True,
                val_ratio=val_r,
                seed=seed,
                num_workers=num_workers,
            )
            self._train_loader_all = self._make_ct_loader(
                db_dir,
                labeled=None,
                training=True,
                val_ratio=val_r,
                seed=seed,
                num_workers=num_workers,
            )
            self.train_loader = self._train_loader_labeled
        else:
            train_labeled = True if semi_mode == "supervised" else None
            if train_dataset is None:
                train_dataset = CARE2026_CT_Dataset(
                    db_dir=db_dir,
                    config=self.train_config,
                    training=True,
                    labeled=train_labeled,
                    val_ratio=val_r,
                    random_seed=seed,
                )
            self.train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=False,
                drop_last=False,
                collate_fn=collate_fn_ct,
            )

        if val_r <= 0:
            # No val set: skip val logging, no model selection
            self.val_loader = None
        else:
            if val_dataset is None:
                val_dataset = CARE2026_CT_Dataset(
                    db_dir=db_dir,
                    config=self.train_config,
                    training=False,
                    labeled=True,
                    val_ratio=val_r,
                    random_seed=seed,
                )
            self.val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
                drop_last=False,
                collate_fn=collate_fn_ct,
            )
        self.val_train_loader = self.train_loader if bool(self.train_config.get("debug", True)) else None

    def _make_ct_loader(self, db_dir, labeled, training, val_ratio, seed, num_workers):
        ds = CARE2026_CT_Dataset(
            db_dir=db_dir,
            config=self.train_config,
            training=training,
            labeled=labeled,
            val_ratio=val_ratio,
            random_seed=seed,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn_ct,
        )

    def train_one_epoch(self, pbar) -> None:
        # Swap to all-data loader when warmup ends
        warmup = int(self.train_config.get("mt_warmup_epochs", 0))
        if warmup > 0 and self.epoch == warmup and hasattr(self, "_train_loader_all"):
            print(f"Warmup finished (epoch {self.epoch}). Switching to all-data loader.")
            self.train_loader = self._train_loader_all
            self.val_train_loader = self.train_loader if bool(self.train_config.get("debug", True)) else None
        super().train_one_epoch(pbar)

    def _get_cps_weight(self) -> float:
        """Consistency weight with optional warm-up.

        Returns 0 during warmup, then ramps up linearly to lambda_max.
        """
        warmup = max(0, int(self.train_config.get("mt_warmup_epochs", 0)))
        if self.epoch < warmup:
            return 0.0
        rampup_epochs = max(1, int(self.train_config.get("mt_rampup_epochs", 30)))
        lambda_max = float(self.train_config.get("cps_lambda_max", 1.0))
        t = (self.epoch - warmup) / rampup_epochs
        return min(1.0, t) * lambda_max

    def _get_clce_weight(self) -> float:
        """clCE weight, gated by start epoch."""
        base = float(self.train_config.loss_weights.get("sup_clce", 0.0))
        if base <= 0:
            return 0.0
        start = int(self.train_config.loss_weights.get("clce_start_epoch", 0))
        return base if self.epoch >= start else 0.0

    def run_one_step(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self._current_cps_weight = self._get_cps_weight()
        self._current_clce_weight = self._get_clce_weight()
        return self.model(
            img=input_tensors["image"],
            labels={
                "ct_mask": input_tensors["mask"].long(),
                "labeled": input_tensors["is_labeled"],
            },
            cps_weight=self._current_cps_weight,
            clce_weight=self._current_clce_weight,
        )

    @staticmethod
    def _count_labeled(ds) -> int:
        return sum(1 for r in ds._records if ds._is_labeled_map.get(r, False))

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        # Full-volume for val set (small, matches official pipeline);
        # patch-based otherwise.
        if data_loader is self.val_loader:
            return self._evaluate_full_volume(data_loader)
        return self._evaluate_patch(data_loader)

    def _evaluate_full_volume(self, data_loader: DataLoader) -> Dict[str, float]:
        """Full-volume sliding-window evaluation (matches official pipeline)."""
        from predict import predict_ct

        original_state = self.model.training
        self.model.eval()
        ds = data_loader.dataset

        per_class_dice = {1: [], 2: [], 3: []}
        records = [r for r in ds._records if ds._is_labeled_map.get(r, False)]

        with tqdm(
            total=len(records),
            desc="Evaluation (full-vol)",
            unit="vol",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=False,
        ) as pbar:
            for rec in records:
                img_path = ds._reader.get_data_path(rec)
                gt = ds._reader.load_ann(rec)
                out = predict_ct(img_path, self.model, device=self.device, use_tta=False)
                pred = out.ct_mask
                for class_idx in [1, 2, 3]:
                    per_class_dice[class_idx].append(_binary_dice(pred == class_idx, gt == class_idx))
                pbar.update(1)

        self.model.train(original_state)
        ct_dice_la = float(np.mean(per_class_dice[1])) if per_class_dice[1] else 0.0
        ct_dice_pv = float(np.mean(per_class_dice[2])) if per_class_dice[2] else 0.0
        ct_dice_laa = float(np.mean(per_class_dice[3])) if per_class_dice[3] else 0.0
        return {
            "ct_dice_la": ct_dice_la,
            "ct_dice_pv": ct_dice_pv,
            "ct_dice_laa": ct_dice_laa,
            "ct_mean_dice": float(np.mean([ct_dice_la, ct_dice_pv, ct_dice_laa])),
        }

    def _evaluate_patch(self, data_loader: DataLoader) -> Dict[str, float]:
        """Patch-based evaluation (fast approximation for train set)."""
        original_state = self.model.training
        self.model.eval()

        per_class_dice = {1: [], 2: [], 3: []}

        with tqdm(
            total=len(data_loader.dataset),
            desc="Evaluation (patch)",
            unit="vol",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=False,
        ) as pbar:
            for input_tensors in data_loader:
                out = self.model(img=input_tensors["image"])
                pred = out["seg_mask"].detach().cpu().numpy()
                target = input_tensors["mask"].numpy()
                is_labeled = input_tensors["is_labeled"].numpy()

                for batch_idx in range(pred.shape[0]):
                    if not is_labeled[batch_idx]:
                        continue
                    for class_idx in [1, 2, 3]:
                        per_class_dice[class_idx].append(
                            _binary_dice(pred[batch_idx] == class_idx, target[batch_idx] == class_idx)
                        )
                pbar.update(pred.shape[0])

        self.model.train(original_state)
        ct_dice_la = float(np.mean(per_class_dice[1])) if per_class_dice[1] else 0.0
        ct_dice_pv = float(np.mean(per_class_dice[2])) if per_class_dice[2] else 0.0
        ct_dice_laa = float(np.mean(per_class_dice[3])) if per_class_dice[3] else 0.0
        return {
            "ct_dice_la": ct_dice_la,
            "ct_dice_pv": ct_dice_pv,
            "ct_dice_laa": ct_dice_laa,
            "ct_mean_dice": float(np.mean([ct_dice_la, ct_dice_pv, ct_dice_laa])),
        }


class CARE2026_MRI_Stage2_Trainer(_BaseCARE2026Trainer):
    """Trainer for MRI Stage 2 scar-only segmentation."""

    __name__ = "CARE2026_MRI_Stage2_Trainer"

    def __init__(
        self,
        model: nn.Module,
        model_config: dict,
        train_config: dict,
        device: Optional[torch.device] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> None:
        tc = CFG(deepcopy(MRI_Stage2_TrainCfg))
        tc.update(deepcopy(train_config))
        tc.classes = ["la_scar"]
        tc.monitor = tc.get("monitor", "scar_dice")
        super().__init__(
            model=model,
            dataset_cls=CARE2026_MRI_Stage2_Dataset,
            collate_fn=collate_fn_mri,
            model_config=model_config,
            train_config=tc,
            device=device,
            lazy=lazy,
            **kwargs,
        )

    @property
    def save_prefix(self) -> str:
        model_name = getattr(self._model, "__name__", self._model.__class__.__name__)
        return f"{model_name}-scar"

    def extra_log_suffix(self) -> str:
        return f"scar_{self.train_config.optimizer}"

    def _setup_dataloaders(self, train_dataset=None, val_dataset=None) -> None:
        num_workers = 1 if self.device == torch.device("cpu") else 4
        db_dir = self.train_config.db_dir
        val_r = float(self.train_config.get("val_ratio", 0.1))
        seed = int(self.train_config.get("random_seed", 42))
        if val_r <= 0:
            self.train_config.monitor = None
        if train_dataset is None:
            train_dataset = CARE2026_MRI_Stage2_Dataset(
                db_dir=db_dir,
                config=self.train_config,
                training=True,
                val_ratio=val_r,
                random_seed=seed,
                no_scar_proportion=float(self.train_config.get("no_scar_proportion", 0.3)),
            )
        if val_r <= 0:
            self.val_loader = None
        else:
            if val_dataset is None:
                val_dataset = CARE2026_MRI_Stage2_Dataset(
                    db_dir=db_dir,
                    config=self.train_config,
                    training=False,
                    val_ratio=val_r,
                    random_seed=seed,
                )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False,
                drop_last=False,
                collate_fn=collate_fn_mri,
            )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn_mri,
        )
        self.val_train_loader = self.train_loader if bool(self.train_config.get("debug", True)) else None

    def run_one_step(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(
            img=input_tensors["image"],
            labels={"scar_mask": input_tensors["scar_mask"].long(), "has_scar": input_tensors["has_scar"]},
        )

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        original_state = self.model.training
        self.model.eval()
        scar_dices, scar_accs, scar_sens = [], [], []
        with tqdm(
            total=len(data_loader.dataset),
            desc="Evaluation (Scar)",
            unit="vol",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=False,
        ) as pbar:
            for input_tensors in data_loader:
                out = self.model(img=input_tensors["image"])
                pred_scar = out["scar_mask"].detach().cpu().numpy()
                gt_scar = input_tensors["scar_mask"].numpy().astype(np.uint8)
                has_scar = input_tensors["has_scar"].numpy().astype(bool)
                for idx in range(pred_scar.shape[0]):
                    if has_scar[idx]:
                        scar_dices.append(_binary_dice(pred_scar[idx], gt_scar[idx]))
                        scar_accs.append(_binary_accuracy(pred_scar[idx], gt_scar[idx]))
                        scar_sens.append(_binary_sensitivity(pred_scar[idx], gt_scar[idx]))
                pbar.update(pred_scar.shape[0])
        self.model.train(original_state)
        return {
            "scar_dice": float(np.mean(scar_dices)) if scar_dices else 0.0,
            "scar_acc": float(np.mean(scar_accs)) if scar_accs else 0.0,
            "scar_sen": float(np.mean(scar_sens)) if scar_sens else 0.0,
        }


def get_args(**kwargs: Any) -> CFG:
    cfg = deepcopy(kwargs)
    parser = argparse.ArgumentParser(
        description="Train CARE2026 models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", choices=["mri", "mri_scar", "ct"], required=True)
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=2,
        help="MRI pipeline stage: 1 = coarse LA localisation, 2 = fine segmentation (default).",
    )
    parser.add_argument("--db-dir", dest="db_dir", required=True)
    parser.add_argument("--backbone", default=None, choices=["vnet", "nested_vnet", "vnet_l", "nested_vnet_l"])
    parser.add_argument("-b", "--batch-size", type=int, default=None, dest="batch_size")
    parser.add_argument("--accum-steps", type=int, default=None, dest="accumulate_grad_batches")
    parser.add_argument("--use-amp", type=str2bool, default=None, dest="use_amp")
    parser.add_argument("--epochs", type=int, default=None, dest="n_epochs")
    parser.add_argument("--debug", type=str2bool, default=True, dest="debug")
    parser.add_argument(
        "--semi-mode",
        type=str,
        default=None,
        choices=["cps", "mean_teacher"],
        dest="semi_supervised_mode",
        help="Semi-supervised mode for CT (Task 3).",
    )
    parser.add_argument("--mclahe", type=str2bool, default=None, dest="apply_mclahe", help="Enable MCLAHE preprocessing")
    parser.add_argument("--optimizer", type=str, default=None, choices=["adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=None, dest="lr")
    parser.add_argument("--lr-scheduler", type=str, default=None, choices=["cosine", "poly", "none"])
    parser.add_argument("--val-ratio", type=float, default=None, dest="val_ratio")
    parser.add_argument("--random-seed", type=int, default=None, dest="random_seed", help="Random seed for reproducibility")
    parser.add_argument(
        "--ct-model",
        type=str,
        default="v1",
        choices=["v1", "v2", "nnunet", "nnunet_mt"],
        dest="ct_model_version",
        help="CT model version: v1, v2, nnunet, nnunet_mt (Mean Teacher + PlainConvUNet).",
    )
    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    cfg.update(args)
    return CFG(cfg)


if __name__ == "__main__":
    # Prevent CUDA memory fragmentation (critical for large 3-D volumes)
    import os as _os

    if "PYTORCH_ALLOC_CONF" not in _os.environ:
        _os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task == "mri":
        stage = int(args.get("stage", 2))
        if stage == 1:
            train_config = CFG(deepcopy(MRI_Stage1_TrainCfg))
            train_config.update(args)
            model_config = deepcopy(ModelCfg)
            model = CARE2026_MRI_Stage1_Model(config=model_config, train_config=train_config)
            trainer_cls = CARE2026_MRI_Stage1_Trainer
        else:
            train_config = CFG(deepcopy(MRI_Stage2_TrainCfg))
            train_config.update(args)
            model_config = deepcopy(ModelCfg)
            model = CARE2026_MRI_Stage2_Model(config=model_config, train_config=train_config)
            trainer_cls = CARE2026_MRI_Stage2_Trainer
    else:
        ct_version = args.get("ct_model_version", "v2")
        if ct_version == "nnunet_mt":
            train_config = CFG(deepcopy(CT_TrainCfg_MT_nnUNet))
            train_config.update(args)
            model_config = deepcopy(ModelCfg)
            model = CARE2026_CT_MT_nnUNet(config=model_config, train_config=train_config)
        elif ct_version == "nnunet":
            train_config = CFG(deepcopy(CT_TrainCfg_nnUNet))
            train_config.update(args)
            model_config = deepcopy(ModelCfg)
            model = CARE2026_CT_nnUNet(config=model_config, train_config=train_config)
        elif ct_version == "v2":
            train_config = CFG(deepcopy(CT_TrainCfgV2))
            train_config.update(args)
            model_config = deepcopy(ModelCfg)
            model = CARE2026_CT_ModelV2(config=model_config, train_config=train_config)
        else:
            train_config = CFG(deepcopy(CT_TrainCfg))
            train_config.update(args)
            model_config = deepcopy(ModelCfg)
            model = CARE2026_CT_Model(config=model_config, train_config=train_config)
        trainer_cls = CARE2026_CT_Trainer

    if torch.cuda.device_count() > 1:
        model = DP(model)
    model = model.to(device=device)

    trainer = trainer_cls(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=device,
        lazy=False,
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        # Save best model, matching normal completion behaviour
        if trainer.best_metric > -np.inf:
            from torch_ecg.utils.misc import get_date_str

            save_suffix = f"metric_{trainer.best_eval_res[trainer.train_config.monitor]:.2f}"
            save_folder = f"BestModel_{trainer.save_prefix}{trainer.best_epoch}_{get_date_str()}_{save_suffix}"
            save_path = Path(train_config.get("model_dir", "checkpoints")) / save_folder
            # Restore best weights and save
            trainer._model.load_state_dict(trainer.best_state_dict)
            trainer._model.save(path=str(save_path), train_config=train_config)
            print(
                f"\nSaved best model (epoch {trainer.best_epoch}, {trainer.train_config.monitor}={trainer.best_metric:.4f}): {save_path}"
            )
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
