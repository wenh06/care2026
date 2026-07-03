"""Convert CARE2026 LGE-MRI data to nnUNet v2 format.

Creates two nnUNet datasets:

- **Dataset 502** — LA cavity segmentation (Task 2): 190 full-volume cases
  (60 from Task 1 dir + 130 from Task 2 dir), binary LA cavity label.

- **Dataset 501** — LA scar segmentation (Task 1): 60 cases **cropped** to a
  fixed-size region centred on the LA cavity centroid.  Uses the same centroid
  crop logic as ``dataset.py:_centroid_crop`` with ``MRI_STAGE2_CROP_SHAPE``
  (256 × 256 × 44) from ``const.py``.  All cropped cases have identical shape,
  which is ideal for nnUNet's self-configuring pipeline.

Usage::

    # Both datasets
    python scripts/prep_nnunet_mri.py --db-dir /Data1/wenh06/CARE2026-LeftAtrium

    # Single dataset
    python scripts/prep_nnunet_mri.py --db-dir ... --task 1
    python scripts/prep_nnunet_mri.py --db-dir ... --task 2

    # With no-scar hard negatives (30% of Task 2 cases)
    python scripts/prep_nnunet_mri.py --db-dir ... --no-scar-proportion 0.3

    # Custom output dir (default: $nnUNet_raw or tmp/nnUNet_raw)
    python scripts/prep_nnunet_mri.py --db-dir ... --output /path/to/nnUNet_raw
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

from const import MRI_STAGE2_CROP_SHAPE
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


def _centroid_crop(
    image: np.ndarray,
    la_mask: np.ndarray,
    scar_mask: np.ndarray,
    crop_shape: Tuple[int, int, int] = MRI_STAGE2_CROP_SHAPE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop *image* and *scar_mask* to a fixed region centred on the LA cavity centroid.

    Matches ``dataset.py:_centroid_crop`` — resample to canonical, compute centroid
    from GT cavity mask, then crop *crop_shape* around it with zero-padding at borders.

    Parameters
    ----------
    image : np.ndarray, shape (H, W, D)
        LGE-MRI volume (should already be in canonical 576×576×44 space).
    la_mask : np.ndarray, shape (H, W, D), uint8
        Binary LA cavity mask (GT).
    scar_mask : np.ndarray, shape (H, W, D), uint8
        Binary scar mask (GT).
    crop_shape : (cH, cW, cD)
        Target crop size, default ``MRI_STAGE2_CROP_SHAPE`` = (256, 256, 44).

    Returns
    -------
    cropped_image : np.ndarray of shape *crop_shape*
    cropped_scar : np.ndarray of shape *crop_shape*
    """
    H, W, D = image.shape
    cH, cW, cD = crop_shape

    # Compute centroid from cavity mask
    fg = np.where(la_mask > 0)
    if fg[0].size == 0:
        # Fallback: image centre
        cx, cy, cz = H // 2, W // 2, D // 2
    else:
        cx, cy, cz = int(round(fg[0].mean())), int(round(fg[1].mean())), int(round(fg[2].mean()))

    def _clamp(center, size, max_dim):
        start = center - size // 2
        return int(np.clip(start, 0, max(max_dim - size, 0)))

    x0 = _clamp(cx, cH, H)
    y0 = _clamp(cy, cW, W)
    z0 = _clamp(cz, cD, D)

    img_crop = image[x0 : x0 + cH, y0 : y0 + cW, z0 : z0 + cD]
    scar_crop = scar_mask[x0 : x0 + cH, y0 : y0 + cW, z0 : z0 + cD]

    # Zero-pad if near border
    pad_x = max(0, cH - img_crop.shape[0])
    pad_y = max(0, cW - img_crop.shape[1])
    pad_z = max(0, cD - img_crop.shape[2])
    if pad_x > 0 or pad_y > 0 or pad_z > 0:
        img_crop = np.pad(img_crop, [(0, pad_x), (0, pad_y), (0, pad_z)])
        scar_crop = np.pad(scar_crop, [(0, pad_x), (0, pad_y), (0, pad_z)])

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
) -> None:
    """Load, centroid-crop, and save one case for Task 1.

    MCLAHE is applied to the full canonical image **before** cropping,
    matching ``dataset.py:_load_all`` (Stage 2) behaviour.

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
        cropped_img, cropped_scar = _centroid_crop(image, la_bin, scar_bin, crop_shape)
    else:
        zero_scar = np.zeros_like(image, dtype=np.uint8)
        cropped_img, _ = _centroid_crop(image, la_bin, zero_scar, crop_shape)
        cropped_scar = np.zeros(crop_shape, dtype=np.uint8)

    _save_nifti(cropped_img, affine, img_dir / f"{case_id}_0000.nii.gz")
    _save_nifti(cropped_scar, affine, lbl_dir / f"{case_id}.nii.gz")


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
        "--mclahe",
        action="store_true",
        default=False,
        help="Apply MCLAHE contrast enhancement to images before saving",
    )
    args = parser.parse_args()

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
        for d in sorted(task1_dir.iterdir(), key=lambda p: int(p.name.split("_")[1])):
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
        for d in sorted(task2_dir.iterdir(), key=lambda p: int(p.name.split("_")[1])):
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
        for d in sorted(task1_dir.iterdir(), key=lambda p: int(p.name.split("_")[1])):
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
                scar_data, scar_affine = _load_nifti(scar_src)
                scar_bin = (scar_data > 0).astype(np.uint8)
                _save_nifti(scar_bin, scar_affine, lbl_dir / f"{case_id}.nii.gz")
            else:
                _process_crop_case(img_src, la_src, scar_src, img_dir, lbl_dir, case_id, crop_shape, apply_mclahe=args.mclahe)

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
                for d in sampled:
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
                            img_src, la_src, None, img_dir, lbl_dir, case_id, crop_shape, apply_mclahe=args.mclahe
                        )

                    no_scar_cases.append(case_id)
                    case_idx += 1

        dataset_json = {
            "channel_names": {"0": "LGE-MRI"},
            "labels": {"background": 0, "LA_scar": 1},
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
