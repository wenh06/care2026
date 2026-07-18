"""Evaluate nnUNet models on labeled training data (all tasks).

Usage::

    python scripts/eval_all_nnunet.py --db-dir <CARE2026_data_root>
    python scripts/eval_all_nnunet.py --db-dir ... --output results.txt
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

# Suppress noisy third-party warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"batchgenerators\.")
warnings.filterwarnings("ignore", category=UserWarning, module=r"google\.protobuf")

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import CARE2026_CT_nnUNet, CARE2026_MRI_nnUNet
from predict import predict_ct, predict_mri_two_stage, predict_mri_two_stage_hybrid, predict_mri_two_stage_legacy


def _get_trainer_dir(ds_id: int, results_root: Path) -> Path:
    candidates = sorted(results_root.glob(f"Dataset{ds_id}_*/nnUNetTrainer__*"))
    if not candidates:
        raise FileNotFoundError(f"No trainer dir for Dataset {ds_id} in {results_root}")
    return candidates[0]


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


def _print_summary(results: Dict, out_path: Path = None):
    """Print all results in a clean table; optionally write to file."""
    lines = []
    lines.append("=" * 65)
    lines.append("  SUMMARY")
    lines.append("=" * 65)

    for task_key, entries in results.items():
        lines.append(f"\n--- {task_key} ---")
        for label, metrics in entries.items():
            parts = "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            lines.append(f"  {label}:  {parts}")

    text = "\n".join(lines)
    print(text)
    if out_path is not None:
        out_path.write_text(text)


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", required=True, help="CARE2026 data root")
    parser.add_argument("--nnunet-results", default="tmp/nnUNet_results", help="nnUNet results directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks", type=str, default="1,2,3", help="Comma-separated tasks")
    parser.add_argument("--output", type=str, default=None, help="Write results to file (default: stdout only)")
    parser.add_argument(
        "--mri-pipeline",
        type=str,
        choices=["native", "hybrid", "legacy"],
        default="native",
        help="MRI inference pipeline (default: native)",
    )
    parser.add_argument(
        "--use",
        type=str,
        choices=["auto", "best", "final", "latest"],
        default="auto",
        help="Which checkpoint to load (default: auto-detect best→final→latest).",
    )
    args = parser.parse_args()

    db_dir = Path(args.db_dir)
    results_root = Path(args.nnunet_results)
    tasks = [int(t.strip()) for t in args.tasks.split(",")]

    nnunet_checkpoint = None if args.use == "auto" else f"checkpoint_{args.use}.pth"

    _predict_fn = {
        "native": predict_mri_two_stage,
        "hybrid": predict_mri_two_stage_hybrid,
        "legacy": predict_mri_two_stage_legacy,
    }[args.mri_pipeline]

    all_results: Dict[str, Dict] = {}

    # ==================================================================
    # Task 1 — LA Scar (two-stage: cavity → crop → scar)
    # ==================================================================
    if 1 in tasks:
        task1_dir = db_dir / "LA scar quantification（MRI）" / "train_data"
        records = sorted(
            [d for d in task1_dir.iterdir() if d.is_dir() and d.name.startswith("train_")],
            key=lambda p: int(p.name.split("_")[1]),
        )
        task1_results = {}

        combos = [
            ("502+501", 502, 501, False, "no CLAHE"),
            ("512+511", 512, 511, True, "CLAHE"),
        ]
        for label, ds_s1, ds_s2, use_mclahe, desc in combos:
            s1_dir = _get_trainer_dir(ds_s1, results_root)
            s2_dir = _get_trainer_dir(ds_s2, results_root)
            _tc = {"nnunet_model_dir": str(s1_dir)}
            if nnunet_checkpoint:
                _tc["nnunet_checkpoint"] = nnunet_checkpoint
            s1 = CARE2026_MRI_nnUNet(train_config=_tc, apply_mclahe=use_mclahe)
            _tc = {"nnunet_model_dir": str(s2_dir)}
            if nnunet_checkpoint:
                _tc["nnunet_checkpoint"] = nnunet_checkpoint
            s2 = CARE2026_MRI_nnUNet(train_config=_tc, apply_mclahe=use_mclahe)
            dice_vals, acc_vals, sen_vals = [], [], []
            for rec_dir in tqdm(records, desc=f"  T1 {label}", unit="case"):
                img_path = rec_dir / "enhanced.nii.gz"
                gt_path = rec_dir / "scarSegImgM.nii.gz"
                if not img_path.exists() or not gt_path.exists():
                    continue
                gt = (nib.load(str(gt_path)).get_fdata() > 0).astype(np.uint8)
                if gt.sum() == 0:
                    continue
                out = _predict_fn(img_path, s1, s2, use_tta=False, apply_mclahe=use_mclahe, s2_threshold=0.5)
                pred = _resample_if_needed(out.scar_mask, gt)
                d, a, s = _binary_metrics(pred, gt)
                dice_vals.append(d)
                acc_vals.append(a)
                sen_vals.append(s)
            task1_results[f"{label} ({desc})"] = {
                "G-DSC": np.mean(dice_vals),
                "ACC": np.mean(acc_vals),
                "SEN": np.mean(sen_vals),
            }
        all_results[f"Task 1 — LA Scar ({len(records)} cases)"] = task1_results

    # ==================================================================
    # Task 2 — LA Cavity
    # ==================================================================
    if 2 in tasks:
        task1_dir = db_dir / "LA scar quantification（MRI）" / "train_data"
        task2_dir = db_dir / "LA cavity segmentation（MRI）" / "train_data"
        all_cases = []
        for d in sorted(task1_dir.iterdir(), key=lambda p: int(p.name.split("_")[1])):
            if d.is_dir() and (d / "atriumSegImgMO.nii.gz").exists():
                all_cases.append(d)
        for d in sorted(task2_dir.iterdir(), key=lambda p: int(p.name.split("_")[1])):
            if d.is_dir() and (d / "atriumSegImgMO.nii.gz").exists():
                all_cases.append(d)

        task2_results = {}
        for ds_id, label in [(502, "no CLAHE"), (512, "CLAHE")]:
            trainer_dir = _get_trainer_dir(ds_id, results_root)
            use_mclahe = ds_id >= 510
            _tc = {"nnunet_model_dir": str(trainer_dir)}
            if nnunet_checkpoint:
                _tc["nnunet_checkpoint"] = nnunet_checkpoint
            model = CARE2026_MRI_nnUNet(train_config=_tc, apply_mclahe=use_mclahe)
            dice_vals = []
            dice_vals = []
            for rec_dir in tqdm(all_cases, desc=f"  T2 {ds_id}", unit="case"):
                img_path = rec_dir / "enhanced.nii.gz"
                gt_path = rec_dir / "atriumSegImgMO.nii.gz"
                if not img_path.exists() or not gt_path.exists():
                    continue
                gt = (nib.load(str(gt_path)).get_fdata() > 0).astype(np.uint8)
                if gt.sum() == 0:
                    continue
                nii = nib.load(str(img_path))
                img = nii.get_fdata().astype(np.float32)
                zooms = tuple(float(s) for s in nii.header.get_zooms()[:3])
                if use_mclahe:
                    from utils.mclahe import mclahe as _mc

                    img = _mc(img)
                pred = model.predict(img, zooms)
                pred = (pred > 0).astype(np.uint8)
                pred = _resample_if_needed(pred, gt)
                d, _, _ = _binary_metrics(pred, gt)
                dice_vals.append(d)
            task2_results[f"Dataset {ds_id} ({label})"] = {"DSC": np.mean(dice_vals)}
        all_results[f"Task 2 — LA Cavity ({len(all_cases)} cases)"] = task2_results

    # ==================================================================
    # Task 3 — CT
    # ==================================================================
    if 3 in tasks:
        ct_dir = db_dir / "cardiac anatomy segmentation（CT）" / "train_data"
        records = sorted(
            [d for d in ct_dir.iterdir() if d.is_dir() and d.name.startswith("train_")],
            key=lambda p: int(p.name.split("_")[1]),
        )
        trainer_dir = _get_trainer_dir(500, results_root)
        _tc = {"nnunet_model_dir": str(trainer_dir)}
        if nnunet_checkpoint:
            _tc["nnunet_checkpoint"] = nnunet_checkpoint
        ct_model = CARE2026_CT_nnUNet(train_config=_tc)
        per_class = {1: [], 2: [], 3: []}
        labeled = 0
        for rec_dir in tqdm(records, desc="  T3 500", unit="case"):
            num_str = str(int(rec_dir.name.split("_")[1])).zfill(4)
            img_path = rec_dir / f"{num_str}.nii.gz"
            lbl_path = rec_dir / f"label_{num_str}.nii.gz"
            if not img_path.exists() or not lbl_path.exists():
                continue
            gt = nib.load(str(lbl_path)).get_fdata().astype(np.uint8)
            out = predict_ct(img_path, ct_model, use_tta=False)
            pred = _resample_if_needed(out.ct_mask, gt)
            labeled += 1
            for cls_id in [1, 2, 3]:
                d, _, _ = _binary_metrics(pred == cls_id, gt == cls_id)
                per_class[cls_id].append(d)
        la = np.mean(per_class[1]) if per_class[1] else 0
        pv = np.mean(per_class[2]) if per_class[2] else 0
        laa = np.mean(per_class[3]) if per_class[3] else 0
        task3_results = {
            f"Dataset 500 (5-fold ensemble, {labeled} labeled)": {
                "LA": la,
                "PV": pv,
                "LAA": laa,
                "Mean": float(np.mean([la, pv, laa])),
            }
        }
        all_results[f"Task 3 — CT ({labeled} labeled)"] = task3_results

    # ==================================================================
    # Print unified summary
    # ==================================================================
    out_path = Path(args.output) if args.output else None
    _print_summary(all_results, out_path)
    if out_path:
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
