"""Trainer classes for the CARE 2026 Left Atrium challenge."""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
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

from cfg import CT_TrainCfg, ModelCfg, MRI_TrainCfg
from dataset import CARE2026_CT_Dataset, CARE2026_MRI_Dataset, collate_fn_ct, collate_fn_mri
from models import CARE2026_CT_Model, CARE2026_MRI_Model

__all__ = [
    "CARE2026_MRI_Trainer",
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

        total_steps = max(1, self.n_epochs * max(1, len(self.train_loader)))
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

            if self.global_step % self.train_config.log_step == 0:
                step_metrics = {"loss": loss_for_log}
                for key in ["la_loss", "scar_loss", "sup_loss", "cps_loss"]:
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


class CARE2026_MRI_Trainer(_BaseCARE2026Trainer):
    """Trainer for MRI Tasks 1 & 2 (LA cavity + scar)."""

    __name__ = "CARE2026_MRI_Trainer"

    def __init__(
        self,
        model: nn.Module,
        model_config: dict,
        train_config: dict,
        device: Optional[torch.device] = None,
        lazy: bool = True,
        **kwargs: Any,
    ) -> None:
        tc = CFG(deepcopy(MRI_TrainCfg))
        tc.update(deepcopy(train_config))
        tc.classes = ["la_cavity", "la_scar"]
        tc.monitor = tc.get("monitor", "la_dice")
        super().__init__(
            model=model,
            dataset_cls=CARE2026_MRI_Dataset,
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
        return f"{model_name}-mri"

    def extra_log_suffix(self) -> str:
        return f"mri_{self.train_config.optimizer}"

    def _setup_dataloaders(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
    ) -> None:
        num_workers = 1 if self.device == torch.device("cpu") else 4
        db_dir = self.train_config.db_dir
        if train_dataset is None:
            train_dataset = CARE2026_MRI_Dataset(
                db_dir=db_dir,
                config=self.train_config,
                training=True,
                val_ratio=float(self.train_config.get("val_ratio", 0.1)),
                random_seed=int(self.train_config.get("random_seed", 42)),
            )
        if val_dataset is None:
            val_dataset = CARE2026_MRI_Dataset(
                db_dir=db_dir,
                config=self.train_config,
                training=False,
                val_ratio=float(self.train_config.get("val_ratio", 0.1)),
                random_seed=int(self.train_config.get("random_seed", 42)),
            )

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn_mri,
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn_mri,
        )
        self.val_train_loader = None

    def run_one_step(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(
            img=input_tensors["image"],
            labels={
                "la_mask": input_tensors["la_mask"].long(),
                "scar_mask": input_tensors["scar_mask"].long(),
                "has_scar": input_tensors["has_scar"],
            },
        )

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        original_state = self.model.training
        self.model.eval()

        la_dices: List[float] = []
        scar_dices: List[float] = []
        scar_accs: List[float] = []
        scar_sens: List[float] = []

        with tqdm(
            total=len(data_loader.dataset),
            desc="Evaluation",
            unit="vol",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=False,
        ) as pbar:
            for input_tensors in data_loader:
                out = self.model(img=input_tensors["image"])
                pred_la = out["la_mask"].detach().cpu().numpy()
                pred_scar = out["scar_mask"].detach().cpu().numpy()
                gt_la = input_tensors["la_mask"].numpy().astype(np.uint8)
                gt_scar = input_tensors["scar_mask"].numpy().astype(np.uint8)
                has_scar = input_tensors["has_scar"].numpy().astype(bool)

                for idx in range(pred_la.shape[0]):
                    la_dices.append(_binary_dice(pred_la[idx], gt_la[idx]))
                    if has_scar[idx]:
                        scar_dices.append(_binary_dice(pred_scar[idx], gt_scar[idx]))
                        scar_accs.append(_binary_accuracy(pred_scar[idx], gt_scar[idx]))
                        scar_sens.append(_binary_sensitivity(pred_scar[idx], gt_scar[idx]))
                pbar.update(pred_la.shape[0])

        self.model.train(original_state)
        return {
            "la_dice": float(np.mean(la_dices)) if la_dices else 0.0,
            "scar_dice": float(np.mean(scar_dices)) if scar_dices else 0.0,
            "scar_acc": float(np.mean(scar_accs)) if scar_accs else 0.0,
            "scar_sen": float(np.mean(scar_sens)) if scar_sens else 0.0,
        }


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
        if train_dataset is None:
            train_dataset = CARE2026_CT_Dataset(
                db_dir=db_dir,
                config=self.train_config,
                training=True,
                labeled=None,
                val_ratio=float(self.train_config.get("val_ratio", 0.1)),
                random_seed=int(self.train_config.get("random_seed", 42)),
            )
        if val_dataset is None:
            val_dataset = CARE2026_CT_Dataset(
                db_dir=db_dir,
                config=self.train_config,
                training=False,
                labeled=True,
                val_ratio=float(self.train_config.get("val_ratio", 0.1)),
                random_seed=int(self.train_config.get("random_seed", 42)),
            )

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn_ct,
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn_ct,
        )
        self.val_train_loader = None

    def _get_cps_weight(self) -> float:
        rampup_epochs = max(1, int(self.train_config.get("cps_rampup_epochs", 30)))
        lambda_max = float(self.train_config.get("cps_lambda_max", 1.0))
        return min(1.0, self.epoch / rampup_epochs) * lambda_max

    def run_one_step(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self._current_cps_weight = self._get_cps_weight()
        return self.model(
            img=input_tensors["image"],
            labels={
                "ct_mask": input_tensors["mask"].long(),
                "labeled": input_tensors["is_labeled"],
            },
            cps_weight=self._current_cps_weight,
        )

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        original_state = self.model.training
        self.model.eval()

        per_class_dice = {1: [], 2: [], 3: []}

        with tqdm(
            total=len(data_loader.dataset),
            desc="Evaluation",
            unit="vol",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=False,
        ) as pbar:
            for input_tensors in data_loader:
                out = self.model(img=input_tensors["image"])
                pred = out["seg_mask"].detach().cpu().numpy()
                target = input_tensors["mask"].numpy()

                for batch_idx in range(pred.shape[0]):
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


def get_args(**kwargs: Any) -> CFG:
    cfg = deepcopy(kwargs)
    parser = argparse.ArgumentParser(
        description="Train CARE2026 models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", choices=["mri", "ct"], required=True)
    parser.add_argument("--db-dir", dest="db_dir", required=True)
    parser.add_argument("--backbone", default="vnet", choices=["vnet", "nested_vnet"])
    parser.add_argument("-b", "--batch-size", type=int, default=None, dest="batch_size")
    parser.add_argument("--accum-steps", type=int, default=None, dest="accumulate_grad_batches")
    parser.add_argument("--use-amp", type=str2bool, default=None, dest="use_amp")
    parser.add_argument("--epochs", type=int, default=None, dest="n_epochs")
    parser.add_argument("--debug", type=str2bool, default=False, dest="debug")
    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    cfg.update(args)
    return CFG(cfg)


if __name__ == "__main__":
    # Prevent CUDA memory fragmentation (critical for large 3-D volumes)
    import os as _os
    _os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task == "mri":
        train_config = CFG(deepcopy(MRI_TrainCfg))
        train_config.update(args)
        model_config = deepcopy(ModelCfg)
        model = CARE2026_MRI_Model(config=model_config, train_config=train_config, backbone=args.backbone)
        trainer_cls = CARE2026_MRI_Trainer
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
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
