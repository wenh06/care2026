"""Convert CARE2026 LGE-MRI data to nnUNet v2 format.

Creates nnUNet datasets for Task 1 (scar) and Task 2 (cavity).

Common datasets
--------------
- **Dataset 502** — LA cavity segmentation (Task 2): 190 full-volume cases,
  binary LA cavity label.
- **Dataset 501** — LA scar segmentation (Task 1): 60 cases centroid-cropped
  to 256×256×44, binary scar label.
- **Dataset 511/512** — Same as 501/502 but with MCLAHE (``--mclahe``).

Multi-class (cavity + scar joint label)
---------------------------------------
- **Dataset 521** — 60 cropped cases, 2-class label (cavity=1, scar=2).
- **Dataset 531** — Same as 521 but with MCLAHE.
- **Dataset 600** — 60 full-volume cases, 2-class label.

Usage::

    # Standard (binary scar, no CLAHE)
    python scripts/prep_nnunet_mri.py --db-dir ... --task 1           # → 501
    python scripts/prep_nnunet_mri.py --db-dir ... --task 2           # → 502

    # With CLAHE
    python scripts/prep_nnunet_mri.py --db-dir ... --task 1 --mclahe  # → 511

    # Multi-class (cavity + scar) — must pass explicit --dataset-id-task1
    python scripts/prep_nnunet_mri.py --db-dir ... --task 1 --multi-class --dataset-id-task1 521
    python scripts/prep_nnunet_mri.py --db-dir ... --task 1 --multi-class --mclahe --dataset-id-task1 531
    python scripts/prep_nnunet_mri.py --db-dir ... --task 1 --multi-class --no-crop --dataset-id-task1 600
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm.auto import tqdm

from const import MRI_STAGE2_CROP_SHAPE
from utils.centroid_crop import centroid_crop_3d
from utils.mclahe import mclahe as _mclahe

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (data, affine) from a NIfTI file."""
    img = nib.load(str(path))
    return img.get_fdata(), img.affine


def _save_nifti(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    """Save a 3-D array as a NIfTI file, preserving the input dtype."""
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))


def _scar_weight_map(
    scar_mask: np.ndarray,
    w0: float = 5.0,
    sigma_mm: float = 2.0,
    spacing_xy: float = 0.625,
) -> np.ndarray:
    """Gaussian spatial weight map centred on scar voxels.

    w(x) = 1 + w₀ · exp(−d(x)² / 2σ²)
    where d(x) = Euclidean distance to nearest scar voxel.

    Matches ``ScarLoss`` in ``models/loss/__init__.py``.
    """
    from scipy.ndimage import distance_transform_edt

    if scar_mask.sum() == 0:
        return np.ones_like(scar_mask, dtype=np.float32)
    sigma_px = max(1.0, sigma_mm / spacing_xy)
    d = distance_transform_edt(1 - scar_mask).astype(np.float32)
    return (1.0 + w0 * np.exp(-(d**2) / (2 * sigma_px**2))).astype(np.float32)


def _centroid_crop(
    image: np.ndarray,
    la_mask: np.ndarray,
    scar_mask: np.ndarray,
    crop_shape: Tuple[int, int, int] = MRI_STAGE2_CROP_SHAPE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop *image* and *scar_mask* around the LA cavity centroid.

    Parameters
    ----------
    image : np.ndarray, shape (H, W, D)
        LGE-MRI volume (should already be in canonical 576×576×44 space).
    la_mask : np.ndarray, shape (H, W, D), uint8
        Binary LA cavity mask (GT).
    scar_mask : np.ndarray, shape (H, W, D), uint8
        Binary scar mask (GT).
    crop_shape : (cH, cW, cD)

    Returns
    -------
    cropped_image : np.ndarray of shape *crop_shape*
    cropped_scar : np.ndarray of shape *crop_shape*
    """
    H, W, D = image.shape

    # Compute centroid from cavity mask
    fg = np.where(la_mask > 0)
    if fg[0].size == 0:
        cx, cy, cz = H // 2, W // 2, D // 2
    else:
        cx, cy, cz = int(round(fg[0].mean())), int(round(fg[1].mean())), int(round(fg[2].mean()))

    img_crop, _, _ = centroid_crop_3d(image, (cx, cy, cz), crop_shape)
    scar_crop, _, _ = centroid_crop_3d(scar_mask, (cx, cy, cz), crop_shape)
    return img_crop, scar_crop


def _process_crop_case(
    img_src: Path,
    la_src: Path,
    scar_src: Path,
    img_dir: Path,
    lbl_dir: Path,
    case_id: str,
    crop_shape: Tuple[int, int, int],
    apply_mclahe: bool = False,
    multi_class: bool = False,
    save_weight_map: bool = False,
) -> None:
    """Load, centroid-crop, and save one case for Task 1.

    MCLAHE is applied to the full canonical image **before** cropping,
    matching ``dataset.py:_load_all`` (Stage 2) behaviour.

    When ``multi_class`` is True, the label is 2-class (1=cavity, 2=scar);
    scar overwrites cavity where both are present.

    *scar_src* may be None for no-scar cases (hard negatives); in that case
    the label is saved as an all-zero mask of *crop_shape*.
    """
    image, affine = _load_nifti(img_src)
    la_data, _ = _load_nifti(la_src)
    la_bin = (la_data > 0).astype(np.uint8)

    if apply_mclahe:
        image = _mclahe(image)

    if scar_src is not None:
        scar_data, _ = _load_nifti(scar_src)
        scar_bin = (scar_data > 0).astype(np.uint8)
        if multi_class:
            # 2-class label: cavity=1, scar=2 (scar takes priority)
            label = la_bin.astype(np.uint8) + scar_bin.astype(np.uint8) * 2
            # Fix overlap: scar=2 wherever scar is present
            label[scar_bin > 0] = 2
            cropped_img, cropped_label = _centroid_crop(image, la_bin, label, crop_shape)
        else:
            cropped_img, cropped_label = _centroid_crop(image, la_bin, scar_bin, crop_shape)
    else:
        if multi_class:
            # No-scar case with multi-class: cavity-only label
            label = la_bin.astype(np.uint8)
            zero = np.zeros_like(image, dtype=np.uint8)
            cropped_img, _ = _centroid_crop(image, la_bin, zero, crop_shape)
            cropped_label = np.zeros(crop_shape, dtype=np.uint8)
        else:
            zero_scar = np.zeros_like(image, dtype=np.uint8)
            cropped_img, _ = _centroid_crop(image, la_bin, zero_scar, crop_shape)
            cropped_label = np.zeros(crop_shape, dtype=np.uint8)

    _save_nifti(cropped_img, affine, img_dir / f"{case_id}_0000.nii.gz")
    _save_nifti(cropped_label, affine, lbl_dir / f"{case_id}.nii.gz")
    if save_weight_map:
        if scar_src is not None:
            scar_data, _ = _load_nifti(scar_src)
            scar_bin = (scar_data > 0).astype(np.uint8)
            _, cropped_scar = _centroid_crop(np.zeros_like(image, dtype=np.uint8), la_bin, scar_bin, crop_shape)
            wm = _scar_weight_map(cropped_scar)
        else:
            wm = np.ones(crop_shape, dtype=np.float32)
        np.save(str(lbl_dir / f"{case_id}_weight.npy"), wm)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Prepare LGE-MRI data for nnUNet v2")
    parser.add_argument("--db-dir", required=True, help="Root of the CARE2026 dataset")
    parser.add_argument("--task", type=int, choices=[1, 2], default=None, help="Which task to prepare (default: both)")
    parser.add_argument("--dataset-id-task1", type=int, default=501, help="nnUNet dataset ID for Task 1 (scar)")
    parser.add_argument("--dataset-id-task2", type=int, default=502, help="nnUNet dataset ID for Task 2 (LA cavity)")
    parser.add_argument("--output", default=None, help="nnUNet_raw directory (default: $nnUNet_raw or tmp/nnUNet_raw)")
    parser.add_argument(
        "--no-crop",
        action="store_true",
        default=False,
        help="Do NOT crop Task 1 data (train on full volume instead)",
    )
    parser.add_argument(
        "--no-scar-proportion",
        type=float,
        default=0.0,
        help="Fraction of no-scar (Task 2) cases to include in Task 1 as hard negatives (default 0)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for no-scar case sampling")
    parser.add_argument(
        "--multi-class",
        action="store_true",
        default=False,
        help="Create 2-class labels (1=cavity, 2=scar) instead of binary scar-only labels",
    )
    parser.add_argument(
        "--weight-map",
        action="store_true",
        default=False,
        help="Save Gaussian spatial weight map alongside label (for custom loss)",
    )
    parser.add_argument(
        "--mclahe",
        action="store_true",
        default=False,
        help="Apply MCLAHE contrast enhancement to images before saving",
    )
    args = parser.parse_args()

    # MCLAHE variant uses +10 dataset ID to keep non-MCLAHE data intact
    if args.mclahe:
        if args.dataset_id_task1 == 501:
            args.dataset_id_task1 = 511
        if args.dataset_id_task2 == 502:
            args.dataset_id_task2 = 512

    db_dir = Path(args.db_dir)
    task1_dir = db_dir / "LA scar quantification（MRI）" / "train_data"
    task2_dir = db_dir / "LA cavity segmentation（MRI）" / "train_data"

    nnunet_raw = Path(args.output) if args.output else Path(os.environ.get("nnUNet_raw", "tmp/nnUNet_raw"))

    crop_shape = tuple(MRI_STAGE2_CROP_SHAPE)  # (256, 256, 44)

    # ------------------------------------------------------------------
    # Dataset 502 — LA cavity segmentation (Task 2): 190 full-volume cases
    # ------------------------------------------------------------------
    if args.task is None or args.task == 2:
        dataset_name = f"Dataset{args.dataset_id_task2:03d}_CARE2026MRI_Cavity"
        out_dir = nnunet_raw / dataset_name
        img_dir = out_dir / "imagesTr"
        lbl_dir = out_dir / "labelsTr"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        case_idx = 0
        # Task 1 directory cases (train_1..train_60): have cavity + scar labels
        t1_cases = sorted(task1_dir.iterdir(), key=lambda p: int(p.name.split("_")[1]))
        for d in tqdm(t1_cases, desc="Task2 images (T1 dir)", unit="case"):
            if not d.is_dir():
                continue
            img_src = d / "enhanced.nii.gz"
            la_src = d / "atriumSegImgMO.nii.gz"
            if not img_src.exists() or not la_src.exists():
                continue
            case_id = f"CARE{case_idx:04d}"
            if args.mclahe:
                image, affine = _load_nifti(img_src)
                image = _mclahe(image)
                _save_nifti(image, affine, img_dir / f"{case_id}_0000.nii.gz")
            else:
                os.symlink(img_src.resolve(), img_dir / f"{case_id}_0000.nii.gz")
            la_data, la_affine = _load_nifti(la_src)
            la_bin = (la_data > 0).astype(np.uint8)
            _save_nifti(la_bin, la_affine, lbl_dir / f"{case_id}.nii.gz")
            case_idx += 1

        n_task1_cases = case_idx
        # Task 2 directory cases (train_1..train_130): cavity labels only
        t2_cases_all = sorted(task2_dir.iterdir(), key=lambda p: int(p.name.split("_")[1]))
        for d in tqdm(t2_cases_all, desc="Task2 images (T2 dir)", unit="case"):
            if not d.is_dir():
                continue
            img_src = d / "enhanced.nii.gz"
            la_src = d / "atriumSegImgMO.nii.gz"
            if not img_src.exists() or not la_src.exists():
                continue
            case_id = f"CARE{case_idx:04d}"
            if args.mclahe:
                image, affine = _load_nifti(img_src)
                image = _mclahe(image)
                _save_nifti(image, affine, img_dir / f"{case_id}_0000.nii.gz")
            else:
                os.symlink(img_src.resolve(), img_dir / f"{case_id}_0000.nii.gz")
            la_data, la_affine = _load_nifti(la_src)
            la_bin = (la_data > 0).astype(np.uint8)
            _save_nifti(la_bin, la_affine, lbl_dir / f"{case_id}.nii.gz")
            case_idx += 1

        n_total = case_idx
        dataset_json = {
            "channel_names": {"0": "LGE-MRI"},
            "labels": {"background": 0, "LA_cavity": 1},
            "numTraining": n_total,
            "file_ending": ".nii.gz",
            "overwrite_image_reader_writer": "NibabelIOWithReorient",
        }
        with open(out_dir / "dataset.json", "w") as f:
            json.dump(dataset_json, f, indent=2)

        print(f"[Task 2] Created {out_dir}")
        print(f"  imagesTr: {len(list(img_dir.glob('*_0000.nii.gz')))} files")
        print(f"  labelsTr: {len(list(lbl_dir.glob('*.nii.gz')))} files")
        print(f"  ({n_task1_cases} from Task 1 dir, {n_total - n_task1_cases} from Task 2 dir)")
        if args.mclahe:
            print("  MCLAHE: enabled")
        print(f"\nNext: nnUNetv2_plan_and_preprocess -d {args.dataset_id_task2:03d} --verify_dataset_integrity -c 3d_fullres")
        print(f"Train single fold: nnUNetv2_train {args.dataset_id_task2:03d} 3d_fullres 0")
        print(f"Train all folds  : for f in 0 1 2 3 4; do nnUNetv2_train {args.dataset_id_task2:03d} 3d_fullres $f; done")

    # ------------------------------------------------------------------
    # Dataset 501 — LA scar segmentation (Task 1): 60 cropped cases
    # ------------------------------------------------------------------
    if args.task is None or args.task == 1:
        dataset_name = f"Dataset{args.dataset_id_task1:03d}_CARE2026MRI_Scar"
        out_dir = nnunet_raw / dataset_name
        img_dir = out_dir / "imagesTr"
        lbl_dir = out_dir / "labelsTr"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        case_idx = 0

        # Scar-positive cases from Task 1 directory
        scar_cases = sorted(task1_dir.iterdir(), key=lambda p: int(p.name.split("_")[1]))
        for d in tqdm(scar_cases, desc="Task1 scar cases", unit="case"):
            if not d.is_dir():
                continue
            img_src = d / "enhanced.nii.gz"
            la_src = d / "atriumSegImgMO.nii.gz"
            scar_src = d / "scarSegImgM.nii.gz"
            if not img_src.exists() or not la_src.exists() or not scar_src.exists():
                continue

            case_id = f"CARE{case_idx:04d}"

            if args.no_crop:
                image, affine = _load_nifti(img_src)
                if args.mclahe:
                    image = _mclahe(image)
                _save_nifti(image, affine, img_dir / f"{case_id}_0000.nii.gz")
                if args.multi_class:
                    la_data, _ = _load_nifti(la_src)
                    scar_data, _ = _load_nifti(scar_src)
                    label = (la_data > 0).astype(np.uint8) + (scar_data > 0).astype(np.uint8) * 2
                    label[(scar_data > 0)] = 2
                    _save_nifti(label, affine, lbl_dir / f"{case_id}.nii.gz")
                    if args.weight_map:
                        scar_bin = (scar_data > 0).astype(np.uint8)
                        np.save(str(lbl_dir / f"{case_id}_weight.npy"), _scar_weight_map(scar_bin))
                else:
                    scar_data, scar_affine = _load_nifti(scar_src)
                    scar_bin = (scar_data > 0).astype(np.uint8)
                    _save_nifti(scar_bin, scar_affine, lbl_dir / f"{case_id}.nii.gz")
                    if args.weight_map:
                        np.save(str(lbl_dir / f"{case_id}_weight.npy"), _scar_weight_map(scar_bin))
            else:
                _process_crop_case(
                    img_src,
                    la_src,
                    scar_src,
                    img_dir,
                    lbl_dir,
                    case_id,
                    crop_shape,
                    apply_mclahe=args.mclahe,
                    multi_class=args.multi_class,
                    save_weight_map=args.weight_map,
                )

            case_idx += 1

        n_scar_cases = case_idx

        # --- No-scar hard negatives (sampled from Task 2 directory) ---
        no_scar_cases: list = []
        if args.no_scar_proportion > 0:
            t2_cases = sorted(
                [d for d in task2_dir.iterdir() if d.is_dir() and (d / "atriumSegImgMO.nii.gz").exists()],
                key=lambda p: int(p.name.split("_")[1]),
            )
            if t2_cases:
                rng = np.random.default_rng(args.seed)
                n_sample = max(1, round(len(t2_cases) * args.no_scar_proportion))
                sampled = rng.choice(t2_cases, size=n_sample, replace=False)
                for d in tqdm(sampled, desc="Task1 no-scar cases", unit="case"):
                    img_src = d / "enhanced.nii.gz"
                    la_src = d / "atriumSegImgMO.nii.gz"
                    case_id = f"CARE{case_idx:04d}"

                    if args.no_crop:
                        image, affine = _load_nifti(img_src)
                        if args.mclahe:
                            image = _mclahe(image)
                        _save_nifti(image, affine, img_dir / f"{case_id}_0000.nii.gz")
                        all_zero = np.zeros(image.shape, dtype=np.uint8)
                        _save_nifti(all_zero, affine, lbl_dir / f"{case_id}.nii.gz")
                    else:
                        _process_crop_case(
                            img_src,
                            la_src,
                            None,
                            img_dir,
                            lbl_dir,
                            case_id,
                            crop_shape,
                            apply_mclahe=args.mclahe,
                            multi_class=args.multi_class,
                            save_weight_map=args.weight_map,
                        )

                    no_scar_cases.append(case_id)
                    case_idx += 1

        if args.multi_class:
            labels = {"background": 0, "LA_cavity": 1, "LA_scar": 2}
        else:
            labels = {"background": 0, "LA_scar": 1}
        dataset_json = {
            "channel_names": {"0": "LGE-MRI"},
            "labels": labels,
            "numTraining": case_idx,
            "file_ending": ".nii.gz",
            "overwrite_image_reader_writer": "NibabelIOWithReorient",
        }
        with open(out_dir / "dataset.json", "w") as f:
            json.dump(dataset_json, f, indent=2)

        print(f"\n[Task 1] Created {out_dir}")
        print(f"  imagesTr: {len(list(img_dir.glob('*_0000.nii.gz')))} files")
        print(f"  labelsTr: {len(list(lbl_dir.glob('*.nii.gz')))} files")
        print(f"  Crop shape: {crop_shape}" if not args.no_crop else "  Full volume (no crop)")
        print(f"  Scar cases: {n_scar_cases}")
        if args.mclahe:
            print("  MCLAHE: enabled (applied to full image before crop)")
        if no_scar_cases:
            print(f"  No-scar cases: {len(no_scar_cases)} (sampled {args.no_scar_proportion:.0%} from Task 2 dir)")
        print(f"\nNext: nnUNetv2_plan_and_preprocess -d {args.dataset_id_task1:03d} --verify_dataset_integrity -c 3d_fullres")
        print(f"Train single fold: nnUNetv2_train {args.dataset_id_task1:03d} 3d_fullres 0")
        print(f"Train all folds  : for f in 0 1 2 3 4; do nnUNetv2_train {args.dataset_id_task1:03d} 3d_fullres $f; done")


if __name__ == "__main__":
    main()
