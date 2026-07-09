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
from datetime import datetime
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


def _sorted_records(data_dir: Path) -> List[str]:
    """Return record names sorted by numeric suffix.

    Handles both ``val_N`` and ``test_NN`` (zero-padded) prefixes.
    """
    if not data_dir.exists():
        return []
    recs = [d.name for d in data_dir.iterdir() if d.is_dir() and (d.name.startswith("val_") or d.name.startswith("test_"))]
    recs.sort(key=lambda r: int(r.split("_")[1]))
    return recs


def _ct_image_name(record_id: str) -> str:
    """Derive the CT image filename from the validation record ID.

    ``val_1`` → ``0001.nii.gz``, ``val_12`` → ``0012.nii.gz``.
    """
    num = int(record_id.split("_")[1])
    return f"{num:04d}.nii.gz"


def _print_model_config(model: torch.nn.Module, name: str, ckpt_path: Path) -> None:
    """Log key preprocessing/architecture config from a loaded checkpoint.

    Call **after** ``model.train_config.update(aux_config)`` so the
    train_config reflects what was actually used during training.
    """
    tc = model.train_config
    print(f"[{name}] {ckpt_path.name}")
    print(f"  apply_mclahe : {tc.get('apply_mclahe', False)}")
    print(f"  backbone     : {tc.get('backbone', 'vnet')}")
    print(f"  task / stage : {tc.get('task', '?')} / {tc.get('stage', '?')}")
    print(f"  epochs       : {tc.get('n_epochs', '?')}")
    # Detect norm type from the encoder stem
    try:
        norm_layer = model.backbone.encoder.stem[1]
        norm_type = type(norm_layer).__name__
    except Exception:
        norm_type = "?"
    print(f"  encoder norm : {norm_type}")


# ---------------------------------------------------------------------------
# Per-task inference runners
# ---------------------------------------------------------------------------


def _output_exists(results_dir: Path, record_id: str, task_num: int) -> bool:
    """Check if the prediction file for *record_id* already exists."""
    from outputs import _TASK_DIRNAME

    out_dir = results_dir / _TASK_DIRNAME[task_num] / record_id
    if record_id.startswith("test_"):
        out_name = f"{record_id.split('_')[1]}_pred.nii.gz"
    else:
        out_name = f"{record_id}_pred.nii.gz"
    return (out_dir / out_name).exists()


def _resolve_task_dir(val_data_root: Path, task_num: int) -> Path:
    data_dir = val_data_root / f"task{task_num}"
    if (data_dir / "val_data").exists():
        data_dir = data_dir / "val_data"
    return data_dir


def run_task1_inference(
    stage1_model: torch.nn.Module,
    stage2_model: torch.nn.Module,
    val_data_root: Union[str, Path],
    results_dir: Union[str, Path],
    device: Optional[torch.device] = None,
    use_tta: bool = True,
    s1_threshold: float = 0.5,
    s2_threshold: float = 0.5,
    scar_dilation: Optional[float] = 5.0,
    overwrite: bool = False,
) -> None:
    """Run Task 1 (LA scar) inference."""
    val_data_root = Path(val_data_root)
    results_dir = Path(results_dir)
    data_dir = _resolve_task_dir(val_data_root, 1)
    records = _sorted_records(data_dir)
    if not records:
        warnings.warn(f"No Task 1 records found in {data_dir}")
        return

    if device is None:
        device = next(stage1_model.parameters()).device

    stage1_model.eval()
    stage2_model.eval()
    skipped = 0
    for rec in tqdm(records, desc="Task 1 (LA scar)", unit="vol", dynamic_ncols=True):
        if not overwrite and _output_exists(results_dir, rec, 1):
            print(f"  [SKIP] {rec}")
            skipped += 1
            continue
        img_path = data_dir / rec / "enhanced.nii.gz"
        if not img_path.exists():
            warnings.warn(f"Image not found: {img_path}")
            continue
        out = predict_mri_two_stage(
            img_path,
            stage1_model,
            stage2_model,
            device=device,
            use_tta=use_tta,
            s1_threshold=s1_threshold,
            s2_threshold=s2_threshold,
            scar_dilation=scar_dilation,
        )
        out.save_as_nifti(results_dir, record_id=rec, task_num=1)
    if skipped:
        print(f"Task 1: skipped {skipped} already-completed case(s)")


def run_task2_inference(
    stage1_model: torch.nn.Module,
    val_data_root: Union[str, Path],
    results_dir: Union[str, Path],
    device: Optional[torch.device] = None,
    use_tta: bool = True,
    s1_threshold: float = 0.5,
    overwrite: bool = False,
) -> None:
    """Run Task 2 (LA cavity) inference."""
    val_data_root = Path(val_data_root)
    results_dir = Path(results_dir)
    data_dir = _resolve_task_dir(val_data_root, 2)
    records = _sorted_records(data_dir)
    if not records:
        warnings.warn(f"No Task 2 records found in {data_dir}")
        return

    if device is None:
        device = next(stage1_model.parameters()).device

    stage1_model.eval()
    skipped = 0
    for rec in tqdm(records, desc="Task 2 (LA cavity)", unit="vol", dynamic_ncols=True):
        if not overwrite and _output_exists(results_dir, rec, 2):
            print(f"  [SKIP] {rec}")
            skipped += 1
            continue
        img_path = data_dir / rec / "enhanced.nii.gz"
        if not img_path.exists():
            warnings.warn(f"Image not found: {img_path}")
            continue
        out = predict_mri_two_stage(
            img_path,
            stage1_model,
            stage2_model=None,
            device=device,
            use_tta=use_tta,
            s1_threshold=s1_threshold,
        )
        out.save_as_nifti(results_dir, record_id=rec, task_num=2)
    if skipped:
        print(f"Task 2: skipped {skipped} already-completed case(s)")


def run_task3_inference(
    model: torch.nn.Module,
    val_data_root: Union[str, Path],
    results_dir: Union[str, Path],
    device: Optional[torch.device] = None,
    use_tta: bool = True,
    overwrite: bool = False,
) -> None:
    """Run Task 3 (CT multi-structure) inference."""
    val_data_root = Path(val_data_root)
    results_dir = Path(results_dir)
    data_dir = _resolve_task_dir(val_data_root, 3)
    records = _sorted_records(data_dir)
    if not records:
        warnings.warn(f"No Task 3 records found in {data_dir}")
        return

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    skipped = 0
    for rec in tqdm(records, desc="Task 3 (CT multi-structure)", unit="vol", dynamic_ncols=True):
        if not overwrite and _output_exists(results_dir, rec, 3):
            print(f"  [SKIP] {rec}")
            skipped += 1
            continue
        img_name = _ct_image_name(rec)
        img_path = data_dir / rec / img_name
        if not img_path.exists():
            warnings.warn(f"Image not found: {img_path}")
            continue
        out = predict_ct(img_path, model, device=device, use_tta=use_tta)
        out.save_as_nifti(results_dir, record_id=rec, task_num=3)
    if skipped:
        print(f"Task 3: skipped {skipped} already-completed case(s)")


def run_all_tasks(
    mri_stage1_model: Optional[torch.nn.Module],
    mri_stage2_model: Optional[torch.nn.Module],
    ct_model: Optional[torch.nn.Module],
    val_data_root: Union[str, Path],
    results_dir: Union[str, Path],
    device: Optional[torch.device] = None,
    use_tta: bool = True,
    overwrite: bool = False,
    scar_dilation: Optional[float] = 5.0,
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
            run_task1_inference(
                mri_stage1_model,
                mri_stage2_model,
                val_data_root,
                results_dir,
                device=device,
                use_tta=use_tta,
                overwrite=overwrite,
                scar_dilation=scar_dilation,
            )
        else:
            warnings.warn("Task 1 skipped: MRI Stage-1 and/or Stage-2 model not provided.")

    if 2 in tasks:
        if mri_stage1_model is not None:
            run_task2_inference(
                mri_stage1_model, val_data_root, results_dir, device=device, use_tta=use_tta, overwrite=overwrite
            )
        else:
            warnings.warn("Task 2 skipped: MRI Stage-1 and/or Stage-2 model not provided.")

    if 3 in tasks:
        if ct_model is not None:
            run_task3_inference(ct_model, val_data_root, results_dir, device=device, use_tta=use_tta, overwrite=overwrite)
        else:
            warnings.warn("Task 3 skipped: no CT model provided.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_model(name: str, model_path: str, device: torch.device, mri_mclahe: Optional[bool] = None):
    """Load a model from *model_path*, auto-detecting VNet (file) vs nnUNet (directory)."""
    from cfg import CT_TrainCfg_nnUNet as _nnunet_cfg
    from models import (
        CARE2026_CT_Model,
        CARE2026_CT_nnUNet,
        CARE2026_MRI_nnUNet,
        CARE2026_MRI_Stage1_Model,
        CARE2026_MRI_Stage2_Model,
    )

    path = Path(model_path).expanduser().resolve()
    is_nnunet = path.is_dir() and (path / "plans.json").exists()

    if name == "ct":
        if is_nnunet:
            _tc = dict(_nnunet_cfg)
            _tc["nnunet_model_dir"] = str(path)
            model = CARE2026_CT_nnUNet(train_config=_tc)
            print(f"[CT] nnUNet model from {path}")
        else:
            model, aux = CARE2026_CT_Model.from_checkpoint(str(path), device=device)
            model.train_config.update(aux)
            _print_model_config(model, "CT", path)
        return model.to(device).eval()

    if name == "mri_stage1":
        if is_nnunet:
            kwargs = {"train_config": {"nnunet_model_dir": str(path)}}
            if mri_mclahe is not None:
                kwargs["apply_mclahe"] = mri_mclahe
            model = CARE2026_MRI_nnUNet(**kwargs)
            print(f"[MRI Stage-1] nnUNet model from {path}")
        else:
            model, aux = CARE2026_MRI_Stage1_Model.from_checkpoint(str(path), device=device)
            model.train_config.update(aux)
            _print_model_config(model, "MRI Stage-1", path)
        return model.to(device).eval()

    if name == "mri_stage2":
        if is_nnunet:
            kwargs = {"train_config": {"nnunet_model_dir": str(path)}}
            if mri_mclahe is not None:
                kwargs["apply_mclahe"] = mri_mclahe
            model = CARE2026_MRI_nnUNet(**kwargs)
            print(f"[MRI Stage-2] nnUNet model from {path}")
        else:
            model, aux = CARE2026_MRI_Stage2_Model.from_checkpoint(str(path), device=device)
            model.train_config.update(aux)
            _print_model_config(model, "MRI Stage-2", path)
        return model.to(device).eval()

    raise ValueError(f"Unknown model name: {name}")


if __name__ == "__main__":
    from torch_ecg.utils.misc import str2bool

    from cfg import PredictCfg

    parser = argparse.ArgumentParser(description="CARE2026 Left Atrium — end-to-end inference + submission packaging")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/input",
        help="Validation data root (contains task1/, task2/, task3/ sub-directories).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/output",
        help="Output directory for prediction NIfTI files.",
    )
    parser.add_argument(
        "--val_data_root",
        type=str,
        default=None,
        dest="input_dir_override",
        help="Deprecated alias for --input_dir.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        dest="output_dir_override",
        help="Deprecated alias for --output_dir.",
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
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Sub-directory name appended to --output_dir (default: auto-generated timestamp).",
    )
    parser.add_argument("--s1_threshold", type=float, default=0.5, help="Stage-1 LA cavity probability threshold.")
    parser.add_argument(
        "--overwrite",
        type=str2bool,
        default=False,
        help="Overwrite existing prediction files instead of skipping.",
    )
    parser.add_argument(
        "--s2_threshold", type=float, default=0.5, help="Stage-2 scar probability threshold (VNet only; nnUNet uses argmax)."
    )
    parser.add_argument(
        "--scar-dilation",
        type=lambda x: None if x.lower() in ("none", "null") else (float(x) if float(x) > 0 else (_ for _ in ()).throw(ValueError("scar_dilation must be > 0 or 'none'"))),
        default=5.0,
        help="Scar constraint dilation in mm (>0), or 'none' to disable.",
    )
    parser.add_argument("--ct_threshold", type=float, default=0.5, help="CT multi-class probability threshold.")
    parser.add_argument(
        "--ct-model",
        type=str,
        default=PredictCfg.ct_model,
        help="Path to CT model (.safetensors file or nnUNet directory).",
    )
    parser.add_argument(
        "--mri-stage1-model",
        type=str,
        default=PredictCfg.mri_stage1_model,
        help="Path to MRI Stage-1 model (.safetensors file or nnUNet directory).",
    )
    parser.add_argument(
        "--mri-stage2-model",
        type=str,
        default=PredictCfg.mri_stage2_model,
        help="Path to MRI Stage-2 model (.safetensors file or nnUNet directory).",
    )
    parser.add_argument(
        "--mri-mclahe",
        type=str2bool,
        default=None,
        help="Force MCLAHE on/off for MRI nnUNet models (default: auto-detect from VNet config, False for nnUNet).",
    )
    args = parser.parse_args()

    # Resolve --input_dir / --output_dir, with backward compatibility for
    # the deprecated --val_data_root / --results_dir flags.
    input_dir = args.input_dir_override if args.input_dir_override is not None else args.input_dir
    output_dir = args.output_dir_override if args.output_dir_override is not None else args.output_dir

    # Append run name sub-directory (default: timestamped)
    run_name = args.run_name if args.run_name else datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = str(Path(output_dir) / run_name)

    if "cuda" in args.device and not torch.cuda.is_available():
        args.device = "cpu"
        warnings.warn("CUDA not available. Falling back to CPU.")
    device = torch.device(args.device)

    tasks = [int(t.strip()) for t in args.tasks.split(",")]

    print("Inference config:")
    print(f"  input_dir  : {input_dir}")
    print(f"  output_dir : {output_dir}")
    print(f"  tasks      : {tasks}")
    print(f"  device     : {args.device}")
    print(f"  TTA        : {args.tta}")
    print(f"  run_name   : {run_name}")
    print()

    mri_stage1_model, mri_stage2_model, ct_model = None, None, None

    if 1 in tasks or 2 in tasks:
        try:
            mri_stage1_model = _load_model("mri_stage1", args.mri_stage1_model, device, args.mri_mclahe)
        except FileNotFoundError as e:
            warnings.warn(f"MRI Stage-1: {e}. Tasks 1 & 2 will be skipped.")
        try:
            mri_stage2_model = _load_model("mri_stage2", args.mri_stage2_model, device, args.mri_mclahe)
        except FileNotFoundError as e:
            warnings.warn(f"MRI Stage-2: {e}. Tasks 1 & 2 will be skipped.")

    if 3 in tasks:
        try:
            ct_model = _load_model("ct", args.ct_model, device)
        except FileNotFoundError as e:
            warnings.warn(f"CT: {e}. Task 3 will be skipped.")

    run_all_tasks(
        mri_stage1_model=mri_stage1_model,
        mri_stage2_model=mri_stage2_model,
        ct_model=ct_model,
        val_data_root=input_dir,
        results_dir=output_dir,
        device=device,
        use_tta=args.tta,
        overwrite=args.overwrite,
        scar_dilation=args.scar_dilation,
        tasks=tasks,
    )

    if args.package:
        zip_path = package_submission(
            results_dir=output_dir,
            team_name=args.team_name,
        )
        print(f"Submission zip created: {zip_path}")
