"""Evaluate nnUNet models on labeled training data (all tasks).

Usage::

    python scripts/eval_all_nnunet.py --db-dir <CARE2026_data_root>
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import CARE2026_CT_nnUNet, CARE2026_MRI_nnUNet
from predict import predict_ct, predict_mri_two_stage


def _get_trainer_dir(ds_id: int, results_root: Path) -> Path:
    candidates = sorted(results_root.glob(f"Dataset{ds_id}_*/nnUNetTrainer__*"))
    if not candidates:
        raise FileNotFoundError(f"No trainer dir for Dataset {ds_id} in {results_root}")
    return candidates[0]


def _binary_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> Tuple[float, float, float]:
    """Return dice, accuracy, sensitivity for binary masks."""
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


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", required=True, help="CARE2026 data root")
    parser.add_argument(
        "--nnunet-results",
        default="tmp/nnUNet_results",
        help="nnUNet results directory",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tasks", type=str, default="1,2,3", help="Comma-separated tasks")
    args = parser.parse_args()

    db_dir = Path(args.db_dir)
    results_root = Path(args.nnunet_results)
    tasks = [int(t.strip()) for t in args.tasks.split(",")]

    # ==================================================================
    # Task 1 — LA Scar (two-stage: cavity → crop → scar)
    # ==================================================================
    if 1 in tasks:
        task1_dir = db_dir / "LA scar quantification（MRI）" / "train_data"
        records = sorted(
            [d for d in task1_dir.iterdir() if d.is_dir() and d.name.startswith("train_")],
            key=lambda p: int(p.name.split("_")[1]),
        )
        print("=" * 65)
        print(f"Task 1 — LA Scar ({len(records)} labeled cases)")
        print("=" * 65)

        combos = [
            ("502+501", 502, 501, False, "no CLAHE"),
            ("512+511", 512, 511, True, "CLAHE"),
        ]

        for label, ds_s1, ds_s2, use_mclahe, desc in combos:
            s1_dir = _get_trainer_dir(ds_s1, results_root)
            s2_dir = _get_trainer_dir(ds_s2, results_root)
            s1 = CARE2026_MRI_nnUNet(
                train_config={"nnunet_model_dir": str(s1_dir)},
                apply_mclahe=use_mclahe,
            )
            s2 = CARE2026_MRI_nnUNet(
                train_config={"nnunet_model_dir": str(s2_dir)},
                apply_mclahe=use_mclahe,
            )

            dice_vals, acc_vals, sen_vals = [], [], []
            for rec_dir in tqdm(records, desc=f"  {label}", unit="case"):
                img_path = rec_dir / "enhanced.nii.gz"
                gt_path = rec_dir / "scarSegImgM.nii.gz"
                if not img_path.exists() or not gt_path.exists():
                    continue
                gt = (nib.load(str(gt_path)).get_fdata() > 0).astype(np.uint8)
                if gt.sum() == 0:
                    continue

                out = predict_mri_two_stage(
                    img_path,
                    s1,
                    s2,
                    use_tta=False,
                    apply_mclahe=use_mclahe,
                    s2_threshold=0.5,
                )
                pred = out.scar_mask
                if pred.shape != gt.shape:
                    t = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                    pred = F.interpolate(t, size=gt.shape, mode="nearest").squeeze().numpy().astype(np.uint8)

                d, a, s = _binary_metrics(pred, gt)
                dice_vals.append(d)
                acc_vals.append(a)
                sen_vals.append(s)

            print(f"\n  {label} ({desc}):")
            print(f"    G-DSC : {np.mean(dice_vals):.4f}")
            print(f"    ACC   : {np.mean(acc_vals):.4f}")
            print(f"    SEN   : {np.mean(sen_vals):.4f}")

    # ==================================================================
    # Task 2 — LA Cavity (Stage 1 output)
    # ==================================================================
    if 2 in tasks:
        task1_dir = db_dir / "LA scar quantification（MRI）" / "train_data"
        task2_dir = db_dir / "LA cavity segmentation（MRI）" / "train_data"

        # Collect all cavity-labeled cases
        all_cases = []
        for d in sorted(task1_dir.iterdir(), key=lambda p: int(p.name.split("_")[1])):
            if d.is_dir() and (d / "atriumSegImgMO.nii.gz").exists():
                all_cases.append(("T1_" + d.name, d))
        for d in sorted(task2_dir.iterdir(), key=lambda p: int(p.name.split("_")[1])):
            if d.is_dir() and (d / "atriumSegImgMO.nii.gz").exists():
                all_cases.append(("T2_" + d.name, d))

        print(f"\n{'=' * 65}")
        print(f"Task 2 — LA Cavity ({len(all_cases)} labeled cases)")
        print("=" * 65)

        for ds_id, label in [(502, "no CLAHE"), (512, "CLAHE")]:
            trainer_dir = _get_trainer_dir(ds_id, results_root)
            use_mclahe = ds_id >= 510
            model = CARE2026_MRI_nnUNet(
                train_config={"nnunet_model_dir": str(trainer_dir)},
                apply_mclahe=use_mclahe,
            )

            dice_vals = []
            for case_label, rec_dir in tqdm(all_cases, desc=f"  Dataset {ds_id}", unit="case"):
                img_path = rec_dir / "enhanced.nii.gz"
                gt_path = rec_dir / "atriumSegImgMO.nii.gz"
                if not img_path.exists() or not gt_path.exists():
                    continue
                gt = (nib.load(str(gt_path)).get_fdata() > 0).astype(np.uint8)
                if gt.sum() == 0:
                    continue

                # Single-stage nnUNet: call predict on full image
                spacer = (0.625, 0.625, 2.5)
                img = nib.load(str(img_path)).get_fdata().astype(np.float32)

                # MCLAHE if needed
                if use_mclahe:
                    from utils.mclahe import mclahe as _mc

                    img = _mc(img)

                pred = model.predict(img, spacer)
                pred = (pred > 0).astype(np.uint8)

                if pred.shape != gt.shape:
                    t = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                    pred = F.interpolate(t, size=gt.shape, mode="nearest").squeeze().numpy().astype(np.uint8)

                d, _, _ = _binary_metrics(pred, gt)
                dice_vals.append(d)

            print(f"\n  Dataset {ds_id} ({label}):")
            print(f"    DSC : {np.mean(dice_vals):.4f}")

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
        ct_model = CARE2026_CT_nnUNet(train_config={"nnunet_model_dir": str(trainer_dir)})

        print(f"\n{'=' * 65}")
        print(f"Task 3 — CT ({len(records)} total, 50 labeled)")
        print("=" * 65)

        per_class_dice = {1: [], 2: [], 3: []}
        labeled_count = 0
        for rec_dir in tqdm(records, desc="  CT", unit="case"):
            rec_num = int(rec_dir.name.split("_")[1])
            num_str = str(rec_num).zfill(4)
            img_path = rec_dir / f"{num_str}.nii.gz"
            lbl_path = rec_dir / f"label_{num_str}.nii.gz"
            if not img_path.exists():
                continue
            if not lbl_path.exists():
                continue

            gt = nib.load(str(lbl_path)).get_fdata().astype(np.uint8)
            out = predict_ct(img_path, ct_model, use_tta=False)
            pred = out.ct_mask
            if pred.shape != gt.shape:
                t = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                pred = F.interpolate(t, size=gt.shape, mode="nearest").squeeze().numpy().astype(np.uint8)

            labeled_count += 1
            for cls_id in [1, 2, 3]:
                d, _, _ = _binary_metrics(pred == cls_id, gt == cls_id)
                per_class_dice[cls_id].append(d)

        la = np.mean(per_class_dice[1]) if per_class_dice[1] else 0
        pv = np.mean(per_class_dice[2]) if per_class_dice[2] else 0
        laa = np.mean(per_class_dice[3]) if per_class_dice[3] else 0
        mean_dice = float(np.mean([la, pv, laa]))
        print(f"\n  Dataset 500 (5-fold ensemble, {labeled_count} labeled cases):")
        print(f"    LA  : {la:.4f}")
        print(f"    PV  : {pv:.4f}")
        print(f"    LAA : {laa:.4f}")
        print(f"    Mean: {mean_dice:.4f}")


if __name__ == "__main__":
    main()
