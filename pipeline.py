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
from predict import predict_ct, predict_mri

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
    model: torch.nn.Module,
    val_data_root: Union[str, Path],
    results_dir: Union[str, Path],
    device: Optional[torch.device] = None,
    use_tta: bool = True,
) -> None:
    """Run Task 1 (LA scar) inference on the validation set.

    Saves **only the scar mask** (as required by the challenge specification).

    Parameters
    ----------
    model : CARE2026_MRI_Model
        Trained dual-head VNet in eval mode.
    val_data_root : path-like
        Root containing ``task1/val_data/val_N/`` sub-directories.
    results_dir : path-like
        Output base directory for predictions.
    device : torch.device, optional
        Inference device; defaults to the model's device.
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
        device = next(model.parameters()).device

    model.eval()
    for rec in tqdm(records, desc="Task 1 (LA scar)", unit="vol", dynamic_ncols=True):
        img_path = val_dir / rec / "enhanced.nii.gz"
        if not img_path.exists():
            warnings.warn(f"Image not found: {img_path}")
            continue
        out = predict_mri(img_path, model, device=device, use_tta=use_tta)
        out.save_as_nifti(results_dir, record_id=rec, task_num=1)


def run_task2_inference(
    model: torch.nn.Module,
    val_data_root: Union[str, Path],
    results_dir: Union[str, Path],
    device: Optional[torch.device] = None,
    use_tta: bool = True,
) -> None:
    """Run Task 2 (LA cavity) inference on the validation set.

    Saves the LA cavity mask (binary) for each validation record.

    Parameters
    ----------
    model : CARE2026_MRI_Model
        Trained dual-head VNet in eval mode.
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
        device = next(model.parameters()).device

    model.eval()
    for rec in tqdm(records, desc="Task 2 (LA cavity)", unit="vol", dynamic_ncols=True):
        img_path = val_dir / rec / "enhanced.nii.gz"
        if not img_path.exists():
            warnings.warn(f"Image not found: {img_path}")
            continue
        out = predict_mri(img_path, model, device=device, use_tta=use_tta)
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
    mri_model: Optional[torch.nn.Module],
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
    mri_model : CARE2026_MRI_Model or None
        MRI model for Tasks 1 & 2.  Pass ``None`` to skip.
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

    if 1 in tasks:
        if mri_model is not None:
            run_task1_inference(mri_model, val_data_root, results_dir, device=device, use_tta=use_tta)
        else:
            warnings.warn("Task 1 skipped: no MRI model provided.")

    if 2 in tasks:
        if mri_model is not None:
            run_task2_inference(mri_model, val_data_root, results_dir, device=device, use_tta=use_tta)
        else:
            warnings.warn("Task 2 skipped: no MRI model provided.")

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
    from models import CARE2026_CT_Model, CARE2026_MRI_Model

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
        help="Directory containing 'mri_model.pth.tar' and 'ct_model.pth.tar'.",
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

    mri_model, ct_model = None, None

    if 1 in tasks or 2 in tasks:
        mri_ckpt = model_dir / "mri_model.pth.tar"
        if mri_ckpt.exists():
            mri_model = CARE2026_MRI_Model.from_checkpoint(str(mri_ckpt), device=device)[0]
            mri_model = mri_model.to(device).eval()
        else:
            _warnings.warn(f"MRI model checkpoint not found: {mri_ckpt}. Tasks 1 & 2 will be skipped.")

    if 3 in tasks:
        ct_ckpt = model_dir / "ct_model.pth.tar"
        if ct_ckpt.exists():
            ct_model = CARE2026_CT_Model.from_checkpoint(str(ct_ckpt), device=device)[0]
            ct_model = ct_model.to(device).eval()
        else:
            _warnings.warn(f"CT model checkpoint not found: {ct_ckpt}. Task 3 will be skipped.")

    run_all_tasks(
        mri_model=mri_model,
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
