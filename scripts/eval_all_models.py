"""Evaluate models (nnUNet + VNet) on labeled training data.

Each path auto-detects VNet (.safetensors file) vs nnUNet (directory with plans.json).
MCLAHE is auto-read from model config (nnUNet: dataset.json; VNet: checkpoint metadata).

Usage::

    # Task 1 — paired S1 (cavity) + S2 (scar), positionally matched
    python scripts/eval_all_models.py --db-dir ... \\
        --t1-s1 checkpoints/mri_stage1_model.safetensors \\
        --t1-s2 checkpoints/mri_stage2_model.safetensors

    # Multiple Task 1 combos
    python scripts/eval_all_models.py --db-dir ... \\
        --t1-s1 path/to/S1_A --t1-s2 path/to/S2_A \\
        --t1-s1 path/to/S1_B --t1-s2 path/to/S2_B

    # Task 2 — cavity models (repeatable)
    python scripts/eval_all_models.py --db-dir ... \\
        --t2 path/to/cavity_A --t2 path/to/cavity_B

    # Task 3 — CT models (repeatable)
    python scripts/eval_all_models.py --db-dir ... \\
        --t3 path/to/ct_A --t3 path/to/ct_B

    # Full
    python scripts/eval_all_models.py --db-dir ... \\
        --t1-s1 .../Dataset502_*/... --t1-s2 .../Dataset501_*/... \\
        --t2 .../Dataset502_*/... \\
        --t3 .../Dataset500_*/... \\
        --output results.txt
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"batchgenerators")
warnings.filterwarnings("ignore", category=UserWarning, module=r"google")

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import _load_model  # auto-detect VNet vs nnUNet
from predict import predict_ct, predict_mri_two_stage, predict_mri_two_stage_hybrid, predict_mri_two_stage_legacy

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _binary_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> Tuple[float, float, float]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    denom = pred_b.sum() + gt_b.sum()
    dice = (2 * inter + eps) / (denom + eps) if denom > 0 else 1.0
    acc = float((pred_b == gt_b).mean())
    tp = inter
    fn = np.logical_and(~pred_b, gt_b).sum()
    sens = (tp + eps) / (tp + fn + eps) if (tp + fn) > 0 else 1.0
    return float(dice), float(acc), float(sens)


def _resample_if_needed(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.shape == gt.shape:
        return pred
    t = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, size=gt.shape, mode="nearest").squeeze().numpy().astype(np.uint8)


def _short_label(model_path: str) -> str:
    """Derive a short display label from a model path."""
    p = Path(model_path)
    # nnUNet: extract dataset ID from parent dir name
    for parent in p.parts:
        if parent.startswith("Dataset") and "_" in parent:
            return parent.split("_")[0]  # e.g. "Dataset501" or "Dataset501_CARE2026MRI_Scar"
    # VNet: use stem
    name = p.stem if p.suffix == ".safetensors" else p.name
    if len(name) > 30:
        name = name[:27] + "..."
    return name


def _dedup_args(lst: List[str], label: str) -> List[str]:
    """Warn on duplicates and return deduplicated list."""
    seen = set()
    result = []
    for item in lst:
        key = str(Path(item).resolve())
        if key in seen:
            warnings.warn(f"Duplicate {label} path: {item}")
        else:
            seen.add(key)
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on labeled training data")
    parser.add_argument("--db-dir", required=True, help="CARE2026 data root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks", type=str, default="1,2,3")
    parser.add_argument("--output", type=str, default=None, help="Write results to file")
    # Task 1: paired S1 + S2
    parser.add_argument("--t1-s1", type=str, action="append", default=[], dest="t1_s1", help="Stage 1 path (repeatable)")
    parser.add_argument("--t1-s2", type=str, action="append", default=[], dest="t1_s2", help="Stage 2 path (repeatable)")
    # Task 2: cavity models
    parser.add_argument("--t2", type=str, action="append", default=[], dest="t2", help="Cavity model path (repeatable)")
    # Task 3: CT models
    parser.add_argument("--t3", type=str, action="append", default=[], dest="t3", help="CT model path (repeatable)")
    parser.add_argument("--tta", action="store_true", default=False, help="Enable 8-fold flip TTA (default: off)")
    parser.add_argument(
        "--mri-pipeline",
        type=str,
        choices=["native", "hybrid", "legacy"],
        default="native",
        help="MRI inference pipeline (default: native)",
    )
    args = parser.parse_args()

    # Dedup
    args.t1_s1 = _dedup_args(args.t1_s1, "--t1-s1")
    args.t1_s2 = _dedup_args(args.t1_s2, "--t1-s2")
    args.t2 = _dedup_args(args.t2, "--t2")
    args.t3 = _dedup_args(args.t3, "--t3")

    # Validate Task 1 pairing
    n_t1 = len(args.t1_s1)
    if n_t1 != len(args.t1_s2):
        raise ValueError(f"--t1-s1 ({n_t1} entries) and --t1-s2 ({len(args.t1_s2)} entries) must have same length")

    db_dir = Path(args.db_dir)
    tasks = [int(t.strip()) for t in args.tasks.split(",")]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    _predict_fn = {
        "native": predict_mri_two_stage,
        "hybrid": predict_mri_two_stage_hybrid,
        "legacy": predict_mri_two_stage_legacy,
    }[args.mri_pipeline]

    print("=" * 65)
    print("  Model Evaluation on Labeled Training Data")
    print("=" * 65)
    print(f"  Data root: {db_dir}")
    print(f"  Tasks    : {tasks}")
    print(f"  Task 1   : {n_t1} combo(s)")
    print(f"  Task 2   : {len(args.t2)} model(s)")
    print(f"  Task 3   : {len(args.t3)} model(s)")
    print()

    all_results: Dict[str, Dict[str, Any]] = {}

    # ==================================================================
    # Task 1 — LA Scar
    # ==================================================================
    if 1 in tasks and n_t1 > 0:
        task1_dir = db_dir / "LA scar quantification（MRI）" / "train_data"
        records = sorted(
            [d for d in task1_dir.iterdir() if d.is_dir() and d.name.startswith("train_")],
            key=lambda p: int(p.name.split("_")[1]),
        )
        task1_results: Dict[str, Dict] = {}

        for i in range(n_t1):
            s1_path = args.t1_s1[i]
            s2_path = args.t1_s2[i]
            label = f"S1={_short_label(s1_path)} + S2={_short_label(s2_path)}"

            s1 = _load_model("mri_stage1", s1_path, device)
            s2 = _load_model("mri_stage2", s2_path, device)
            # Auto-read MCLAHE
            s1_mc = bool((getattr(s1, "config", {}) or {}).get("apply_mclahe", False))
            s2_mc = bool((getattr(s2, "config", {}) or {}).get("apply_mclahe", False))
            use_mclahe = s1_mc or s2_mc

            dice_vals, acc_vals, sen_vals = [], [], []
            for rec_dir in tqdm(records, desc=f"  T1 [{label}]", unit="case"):
                img_path = rec_dir / "enhanced.nii.gz"
                gt_path = rec_dir / "scarSegImgM.nii.gz"
                if not img_path.exists() or not gt_path.exists():
                    continue
                gt = (nib.load(str(gt_path)).get_fdata() > 0).astype(np.uint8)
                if gt.sum() == 0:
                    continue
                out = _predict_fn(img_path, s1, s2, use_tta=args.tta, apply_mclahe=use_mclahe, s2_threshold=0.5)
                pred = _resample_if_needed(out.scar_mask, gt)
                d, a, s = _binary_metrics(pred, gt)
                dice_vals.append(d)
                acc_vals.append(a)
                sen_vals.append(s)
            task1_results[label] = {
                "G-DSC": np.mean(dice_vals),
                "ACC": np.mean(acc_vals),
                "SEN": np.mean(sen_vals),
            }
        all_results[f"Task 1 — LA Scar ({len(records)} cases)"] = task1_results

    # ==================================================================
    # Task 2 — LA Cavity
    # ==================================================================
    if 2 in tasks and args.t2:
        task1_dir = db_dir / "LA scar quantification（MRI）" / "train_data"
        task2_dir = db_dir / "LA cavity segmentation（MRI）" / "train_data"
        all_cases: List[Path] = []
        for d in sorted(task1_dir.iterdir(), key=lambda p: int(p.name.split("_")[1])):
            if d.is_dir() and (d / "atriumSegImgMO.nii.gz").exists():
                all_cases.append(d)
        for d in sorted(task2_dir.iterdir(), key=lambda p: int(p.name.split("_")[1])):
            if d.is_dir() and (d / "atriumSegImgMO.nii.gz").exists():
                all_cases.append(d)

        task2_results: Dict[str, Dict] = {}
        for model_path in args.t2:
            label = _short_label(model_path)
            model = _load_model("mri_stage1", model_path, device)
            use_mclahe = bool((getattr(model, "config", {}) or {}).get("apply_mclahe", False))
            dice_vals = []
            for rec_dir in tqdm(all_cases, desc=f"  T2 [{label}]", unit="case"):
                img_path = rec_dir / "enhanced.nii.gz"
                gt_path = rec_dir / "atriumSegImgMO.nii.gz"
                if not img_path.exists() or not gt_path.exists():
                    continue
                gt = (nib.load(str(gt_path)).get_fdata() > 0).astype(np.uint8)
                if gt.sum() == 0:
                    continue
                img = nib.load(str(img_path)).get_fdata().astype(np.float32)
                if use_mclahe:
                    from utils.mclahe import mclahe as _mc

                    img = _mc(img)
                if hasattr(model, "predict"):
                    zooms = tuple(float(s) for s in nib.load(str(img_path)).header.get_zooms()[:3])
                    pred = model.predict(img, zooms, use_tta=args.tta)
                    pred = (pred > 0).astype(np.uint8)
                else:
                    out = _predict_fn(img_path, model, stage2_model=None, use_tta=args.tta)
                    pred = out.la_mask
                pred = _resample_if_needed(pred, gt)
                d, _, _ = _binary_metrics(pred, gt)
                dice_vals.append(d)
            task2_results[label] = {"DSC": np.mean(dice_vals)}
        all_results[f"Task 2 — LA Cavity ({len(all_cases)} cases)"] = task2_results

    # ==================================================================
    # Task 3 — CT
    # ==================================================================
    if 3 in tasks and args.t3:
        ct_dir = db_dir / "cardiac anatomy segmentation（CT）" / "train_data"
        records = sorted(
            [
                d
                for d in ct_dir.iterdir()
                if d.is_dir()
                and d.name.startswith("train_")
                and (d / f"{str(int(d.name.split('_')[1])).zfill(4)}.nii.gz").exists()
                and (d / f"label_{str(int(d.name.split('_')[1])).zfill(4)}.nii.gz").exists()
            ],
            key=lambda p: int(p.name.split("_")[1]),
        )
        task3_results: Dict[str, Dict] = {}
        for model_path in args.t3:
            label = _short_label(model_path)
            model = _load_model("ct", model_path, device)
            per_class = {1: [], 2: [], 3: []}
            labeled = 0
            for rec_dir in tqdm(records, desc=f"  T3 [{label}]", unit="case"):
                num_str = str(int(rec_dir.name.split("_")[1])).zfill(4)
                img_path = rec_dir / f"{num_str}.nii.gz"
                lbl_path = rec_dir / f"label_{num_str}.nii.gz"
                gt = nib.load(str(lbl_path)).get_fdata().astype(np.uint8)
                out = predict_ct(img_path, model, use_tta=args.tta)
                pred = _resample_if_needed(out.ct_mask, gt)
                labeled += 1
                for cls_id in [1, 2, 3]:
                    d, _, _ = _binary_metrics(pred == cls_id, gt == cls_id)
                    per_class[cls_id].append(d)
            la = np.mean(per_class[1]) if per_class[1] else 0
            pv = np.mean(per_class[2]) if per_class[2] else 0
            laa = np.mean(per_class[3]) if per_class[3] else 0
            task3_results[label] = {
                "LA": la,
                "PV": pv,
                "LAA": laa,
                "Mean": float(np.mean([la, pv, laa])),
            }
        all_results[f"Task 3 — CT ({len(records)} total)"] = task3_results

    # ==================================================================
    # Print summary
    # ==================================================================
    if not all_results:
        print("No models to evaluate.")
        return

    lines = []
    lines.append("=" * 65)
    lines.append("  SUMMARY")
    lines.append("=" * 65)

    for task_key, entries in all_results.items():
        lines.append(f"\n--- {task_key} ---")
        # Determine column width
        max_label_len = max(len(k) for k in entries)
        for label, metrics in entries.items():
            parts = "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            lines.append(f"  {label:<{max_label_len}}  {parts}")

    text = "\n".join(lines)
    print(text)
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(text)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
