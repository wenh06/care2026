"""
Inference pipeline for the CARE 2026 Left Atrium challenge.

Supports all three tasks:
- Task 1: LA scar quantification from LGE-MRI
- Task 2: LA cavity segmentation from LGE-MRI
- Task 3: LA multi-structure segmentation from CT

The pipeline follows a coarse-to-fine strategy for MRI tasks and a
sliding-window approach for CT.

Validation data layout assumed::

    <val_data_root>/task1/val_data/val_N/enhanced.nii.gz
    <val_data_root>/task2/val_data/val_N/enhanced.nii.gz
    <val_data_root>/task3/val_data/val_N/NNNN.nii.gz   (NNNN = zero-padded N)

Submission format::

    <results_dir>/LA scar quantification/val_N/val_N_pred.nii.gz
    <results_dir>/LA cavity segmentation/val_N/val_N_pred.nii.gz
    <results_dir>/LA multi-structure segmentation/val_N/val_N_pred.nii.gz
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import List, Optional, Union

import torch
from tqdm.auto import tqdm

from outputs import package_submission
from predict import predict_ct, predict_mri_two_stage

__all__ = [
    "run_task1_inference",
    "run_task2_inference",
    "run_task3_inference",
    "run_all_tasks",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sorted_val_records(val_data_dir: Path) -> List[str]:
    """Return val record names sorted by numeric suffix."""
    if not val_data_dir.exists():
        return []
    recs = [d.name for d in val_data_dir.iterdir() if d.is_dir() and d.name.startswith("val_")]
    recs.sort(key=lambda r: int(r.split("_")[1]))
    return recs


def _ct_image_name(record_id: str) -> str:
    """Derive the CT image filename from the validation record ID.

    ``val_1`` → ``0001.nii.gz``, ``val_12`` → ``0012.nii.gz``.
    """
    num = int(record_id.split("_")[1])
    return f"{num:04d}.nii.gz"


# ---------------------------------------------------------------------------
# Per-task inference runners
# ---------------------------------------------------------------------------


def run_task1_inference(
    stage1_model: torch.nn.Module,
    stage2_model: torch.nn.Module,
    val_data_root: Union[str, Path],
    results_dir: Union[str, Path],
    device: Optional[torch.device] = None,
    use_tta: bool = True,
) -> None:
    """Run Task 1 (LA scar) inference on the validation set.

    Saves **only the scar mask** (as required by the challenge specification).

    Parameters
    ----------
    stage1_model : CARE2026_MRI_Stage1_Model
        Trained single-head VNet for coarse LA localisation.
    stage2_model : CARE2026_MRI_Stage2_Model
        Trained dual-head VNet for fine LA + scar segmentation.
    val_data_root : path-like
        Root containing ``task1/val_data/val_N/`` sub-directories.
    results_dir : path-like
        Output base directory for predictions.
    device : torch.device, optional
        Inference device; defaults to the stage1_model's device.
    use_tta : bool, default True
        Enable 8-fold flip TTA.
    """
    val_data_root = Path(val_data_root)
    val_dir = val_data_root / "task1" / "val_data"
    records = _sorted_val_records(val_dir)

    if not records:
        warnings.warn(f"No Task 1 validation records found in {val_dir}")
        return

    if device is None:
        device = next(stage1_model.parameters()).device

    stage1_model.eval()
    stage2_model.eval()
    for rec in tqdm(records, desc="Task 1 (LA scar)", unit="vol", dynamic_ncols=True):
        img_path = val_dir / rec / "enhanced.nii.gz"
        if not img_path.exists():
            warnings.warn(f"Image not found: {img_path}")
            continue
        out = predict_mri_two_stage(img_path, stage1_model, stage2_model, device=device, use_tta=use_tta)
        out.save_as_nifti(results_dir, record_id=rec, task_num=1)


def run_task2_inference(
    stage1_model: torch.nn.Module,
    stage2_model: torch.nn.Module,
    val_data_root: Union[str, Path],
    results_dir: Union[str, Path],
    device: Optional[torch.device] = None,
    use_tta: bool = True,
) -> None:
    """Run Task 2 (LA cavity) inference on the validation set.

    Saves the LA cavity mask (binary) for each validation record.

    Parameters
    ----------
    stage1_model : CARE2026_MRI_Stage1_Model
        Trained single-head VNet for coarse LA localisation.
    stage2_model : CARE2026_MRI_Stage2_Model
        Trained dual-head VNet for fine LA + scar segmentation.
    val_data_root : path-like
        Root containing ``task2/val_data/val_N/`` sub-directories.
    results_dir : path-like
        Output base directory for predictions.
    device : torch.device, optional
    use_tta : bool, default True
    """
    val_data_root = Path(val_data_root)
    val_dir = val_data_root / "task2" / "val_data"
    records = _sorted_val_records(val_dir)

    if not records:
        warnings.warn(f"No Task 2 validation records found in {val_dir}")
        return

    if device is None:
        device = next(stage1_model.parameters()).device

    stage1_model.eval()
    stage2_model.eval()
    for rec in tqdm(records, desc="Task 2 (LA cavity)", unit="vol", dynamic_ncols=True):
        img_path = val_dir / rec / "enhanced.nii.gz"
        if not img_path.exists():
            warnings.warn(f"Image not found: {img_path}")
            continue
        out = predict_mri_two_stage(img_path, stage1_model, stage2_model, device=device, use_tta=use_tta)
        out.save_as_nifti(results_dir, record_id=rec, task_num=2)


def run_task3_inference(
    model: torch.nn.Module,
    val_data_root: Union[str, Path],
    results_dir: Union[str, Path],
    device: Optional[torch.device] = None,
    use_tta: bool = True,
) -> None:
    """Run Task 3 (CT multi-structure) inference on the validation set.

    Saves a multi-class mask (values 0–3) for each validation record.

    Parameters
    ----------
    model : CARE2026_CT_Model
        Trained CPS model in eval mode.
    val_data_root : path-like
        Root containing ``task3/val_data/val_N/`` sub-directories.
    results_dir : path-like
        Output base directory for predictions.
    device : torch.device, optional
    use_tta : bool, default True
    """
    val_data_root = Path(val_data_root)
    val_dir = val_data_root / "task3" / "val_data"
    records = _sorted_val_records(val_dir)

    if not records:
        warnings.warn(f"No Task 3 validation records found in {val_dir}")
        return

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    for rec in tqdm(records, desc="Task 3 (CT multi-structure)", unit="vol", dynamic_ncols=True):
        img_name = _ct_image_name(rec)
        img_path = val_dir / rec / img_name
        if not img_path.exists():
            warnings.warn(f"Image not found: {img_path}")
            continue
        out = predict_ct(img_path, model, device=device, use_tta=use_tta)
        out.save_as_nifti(results_dir, record_id=rec, task_num=3)


def run_all_tasks(
    mri_stage1_model: Optional[torch.nn.Module],
    mri_stage2_model: Optional[torch.nn.Module],
    ct_model: Optional[torch.nn.Module],
    val_data_root: Union[str, Path],
    results_dir: Union[str, Path],
    device: Optional[torch.device] = None,
    use_tta: bool = True,
    tasks: Optional[List[int]] = None,
) -> None:
    """Convenience wrapper that runs all specified tasks sequentially.

    Parameters
    ----------
    mri_stage1_model : CARE2026_MRI_Stage1_Model or None
        Stage-1 MRI model for coarse LA localisation.  Pass ``None`` to skip
        Tasks 1 & 2.
    mri_stage2_model : CARE2026_MRI_Stage2_Model or None
        Stage-2 MRI model for fine segmentation.  Pass ``None`` to skip
        Tasks 1 & 2.
    ct_model : CARE2026_CT_Model or None
        CT model for Task 3.  Pass ``None`` to skip.
    val_data_root : path-like
        Validation data root (see module docstring for layout).
    results_dir : path-like
        Output directory for predictions.
    device : torch.device, optional
    use_tta : bool, default True
    tasks : list of int, optional
        Subset of ``[1, 2, 3]`` to run.  Defaults to all three.
    """
    if tasks is None:
        tasks = [1, 2, 3]

    mri_ready = mri_stage1_model is not None and mri_stage2_model is not None

    if 1 in tasks:
        if mri_ready:
            run_task1_inference(mri_stage1_model, mri_stage2_model, val_data_root, results_dir, device=device, use_tta=use_tta)
        else:
            warnings.warn("Task 1 skipped: MRI Stage-1 and/or Stage-2 model not provided.")

    if 2 in tasks:
        if mri_ready:
            run_task2_inference(mri_stage1_model, mri_stage2_model, val_data_root, results_dir, device=device, use_tta=use_tta)
        else:
            warnings.warn("Task 2 skipped: MRI Stage-1 and/or Stage-2 model not provided.")

    if 3 in tasks:
        if ct_model is not None:
            run_task3_inference(ct_model, val_data_root, results_dir, device=device, use_tta=use_tta)
        else:
            warnings.warn("Task 3 skipped: no CT model provided.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import warnings as _warnings

    from torch_ecg.utils.misc import str2bool

    from cfg import BaseCfg
    from models import CARE2026_CT_Model, CARE2026_MRI_Stage1_Model, CARE2026_MRI_Stage2_Model

    parser = argparse.ArgumentParser(description="CARE2026 Left Atrium — end-to-end inference + submission packaging")
    parser.add_argument(
        "--val_data_root",
        type=str,
        default=str(BaseCfg.db_dir or "/input"),
        help="Root directory containing task1/, task2/, task3/ sub-directories.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(BaseCfg.results_dir),
        help="Output directory for prediction NIfTI files.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(BaseCfg.model_dir),
        help="Directory containing model checkpoints.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tta", type=str2bool, default=True)
    parser.add_argument(
        "--tasks",
        type=str,
        default="1,2,3",
        help="Comma-separated tasks to run, e.g. '1,3'.",
    )
    parser.add_argument(
        "--team_name",
        type=str,
        default="REVENGER",
        help="Team name for the submission zip filename.",
    )
    parser.add_argument(
        "--package",
        type=str2bool,
        default=True,
        help="Whether to create a submission zip after inference.",
    )
    args = parser.parse_args()

    if "cuda" in args.device and not torch.cuda.is_available():
        args.device = "cpu"
        _warnings.warn("CUDA not available. Falling back to CPU.")
    device = torch.device(args.device)

    tasks = [int(t.strip()) for t in args.tasks.split(",")]
    model_dir = Path(args.model_dir).expanduser().resolve()

    mri_stage1_model, mri_stage2_model, ct_model = None, None, None

    if 1 in tasks or 2 in tasks:
        # Stage-1: discover latest BestModel_*-mri1*.safetensors
        ckpt1 = model_dir / "mri_stage1_model.safetensors"
        if not ckpt1.exists():
            candidates = sorted(model_dir.glob("BestModel_*-mri1*.safetensors"), key=lambda p: p.stat().st_mtime)
            if candidates:
                ckpt1 = candidates[-1]
        if ckpt1.exists():
            mri_stage1_model = CARE2026_MRI_Stage1_Model.from_checkpoint(str(ckpt1), device=device)[0]
            mri_stage1_model = mri_stage1_model.to(device).eval()
            print(f"Loaded MRI Stage-1 model from: {ckpt1.name}")
        else:
            _warnings.warn(f"MRI Stage-1 checkpoint not found in {model_dir}. Tasks 1 & 2 will be skipped.")

        # Stage-2: discover latest BestModel_*-mri2*.safetensors
        ckpt2 = model_dir / "mri_stage2_model.safetensors"
        if not ckpt2.exists():
            candidates = sorted(model_dir.glob("BestModel_*-mri2*.safetensors"), key=lambda p: p.stat().st_mtime)
            if candidates:
                ckpt2 = candidates[-1]
        if ckpt2.exists():
            mri_stage2_model = CARE2026_MRI_Stage2_Model.from_checkpoint(str(ckpt2), device=device)[0]
            mri_stage2_model = mri_stage2_model.to(device).eval()
            print(f"Loaded MRI Stage-2 model from: {ckpt2.name}")
        else:
            _warnings.warn(f"MRI Stage-2 checkpoint not found in {model_dir}. Tasks 1 & 2 will be skipped.")

    if 3 in tasks:
        ct_ckpt = model_dir / "ct_model.safetensors"
        if not ct_ckpt.exists():
            candidates = sorted(model_dir.glob("BestModel_*-ct*.safetensors"), key=lambda p: p.stat().st_mtime)
            if candidates:
                ct_ckpt = candidates[-1]
        if ct_ckpt.exists():
            ct_model = CARE2026_CT_Model.from_checkpoint(str(ct_ckpt), device=device)[0]
            ct_model = ct_model.to(device).eval()
            print(f"Loaded CT model from: {ct_ckpt.name}")
        else:
            _warnings.warn(f"CT model checkpoint not found in {model_dir}. Task 3 will be skipped.")

    run_all_tasks(
        mri_stage1_model=mri_stage1_model,
        mri_stage2_model=mri_stage2_model,
        ct_model=ct_model,
        val_data_root=args.val_data_root,
        results_dir=args.results_dir,
        device=device,
        use_tta=args.tta,
        tasks=tasks,
    )

    if args.package:
        zip_path = package_submission(
            results_dir=args.results_dir,
            team_name=args.team_name,
        )
        print(f"Submission zip created: {zip_path}")
