"""Self-training pipeline: nnUNet 5-fold ensemble → pseudo-labels → retrain.

1. 5-fold ensemble inference on 100 unlabeled CTs → pseudo-labels
2. Convert to nnUNet v2 format with 150 labeled cases (50 GT + 100 pseudo)
3. Print nnUNet training command for AutoDL

Usage:
    python scripts/self_train_nnunet.py \
        --db-dir /Data1/wenh06/CARE2026-LeftAtrium \
        --nnunet-dir checkpoints/nnUNet_results/Dataset500_CARE2026CT/nnUNetTrainer__nnUNetPlans__3d_fullres \
        --dataset-id 501
"""

import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", required=True, help="CARE2026 data root")
    parser.add_argument(
        "--nnunet-dir",
        required=True,
        help="nnUNet training output (contains fold_0/..fold_4/, plans.json, dataset.json)",
    )
    parser.add_argument("--dataset-id", type=int, default=501, help="New nnUNet dataset ID for 150-case training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    args = parser.parse_args()

    db_dir = Path(args.db_dir)
    ct_dir = db_dir / "cardiac anatomy segmentation（CT）" / "train_data"
    nnunet_raw = Path(os.environ.get("nnUNet_raw", "tmp/nnUNet_raw"))

    # ── Step 1: Collect labeled and unlabeled cases ────────────────────
    labeled_cases = []
    unlabeled_cases = []
    for d in sorted(ct_dir.iterdir()):
        if not d.is_dir():
            continue
        rec_num = d.name.split("_")[1]
        num_str = str(int(rec_num)).zfill(4)
        img_src = d / f"{num_str}.nii.gz"
        lbl_src = d / f"label_{num_str}.nii.gz"
        if not img_src.exists():
            continue
        if lbl_src.exists():
            labeled_cases.append((d, rec_num, num_str, img_src, lbl_src))
        else:
            unlabeled_cases.append((d, rec_num, num_str, img_src))

    print(f"Labeled: {len(labeled_cases)}  Unlabeled: {len(unlabeled_cases)}")

    # ── Step 2: 5-fold ensemble inference on unlabeled cases ──────────
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=args.device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        args.nnunet_dir,
        use_folds=(0, 1, 2, 3, 4),  # 5-fold ensemble
        checkpoint_name=args.checkpoint,
    )

    print(f"Generating pseudo-labels for {len(unlabeled_cases)} unlabeled cases...")
    for d, rec_num, num_str, img_src in tqdm(unlabeled_cases, desc="Pseudo-labeling", unit="case"):
        nii = nib.load(str(img_src))
        img = nii.get_fdata().astype(np.float32)
        zooms = tuple(nii.header.get_zooms()[:3])
        spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))

        # Axis transpose for nnUNet: (x,y,z) → (z,y,x)
        img_t = np.transpose(img, (2, 1, 0))
        ret = predictor.predict_from_list_of_npy_arrays(
            image_or_list_of_images=img_t[None].astype(np.float32),
            segs_from_prev_stage_or_list_of_segs_from_prev_stage=None,
            properties_or_list_of_properties={"spacing": spacing},
            truncated_ofname=None,
            num_processes=1,
            save_probabilities=False,
            num_processes_segmentation_export=1,
        )
        pred = ret[0].astype(np.uint8)
        pred = np.transpose(pred, (2, 1, 0))  # back to (x,y,z)

        # Save pseudo-label
        lbl_path = d / f"label_{num_str}.nii.gz"
        nii_out = nib.Nifti1Image(pred, affine=nii.affine, header=nii.header)
        nib.save(nii_out, str(lbl_path))

    print("Pseudo-labels saved.")

    # ── Step 3: Build nnUNet dataset with 150 training cases ─────────
    dataset_name = f"Dataset{args.dataset_id:03d}_CARE2026CT_ST"
    out_dir = nnunet_raw / dataset_name
    img_dir = out_dir / "imagesTr"
    lbl_dir = out_dir / "labelsTr"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_linked = 0
    for d, rec_num, num_str, img_src, lbl_src in labeled_cases:
        os.symlink(img_src.resolve(), img_dir / f"CARE{num_str}_0000.nii.gz")
        os.symlink(lbl_src.resolve(), lbl_dir / f"CARE{num_str}.nii.gz")
        n_linked += 1
    for d, rec_num, num_str, img_src in unlabeled_cases:
        lbl_src = d / f"label_{num_str}.nii.gz"  # just generated
        os.symlink(img_src.resolve(), img_dir / f"CARE{num_str}_0000.nii.gz")
        os.symlink(lbl_src.resolve(), lbl_dir / f"CARE{num_str}.nii.gz")
        n_linked += 1

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "left_atrium": 1, "pulmonary_veins": 2, "left_atrial_appendage": 3},
        "numTraining": n_linked,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
    }
    with open(out_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    nnunet_preprocessed = os.environ.get("nnUNet_preprocessed", "tmp/nnUNet_preprocessed")
    nnunet_results = os.environ.get("nnUNet_results", "checkpoints/nnUNet_results")
    print(f"\nCreated {out_dir} ({n_linked} training cases)")
    print("\nNext steps on AutoDL:")
    print(f"  export nnUNet_raw='{nnunet_raw}'")
    print(f"  export nnUNet_preprocessed='{nnunet_preprocessed}'")
    print(f"  export nnUNet_results='{nnunet_results}'")
    print(f"  nnUNetv2_plan_and_preprocess -d {args.dataset_id:03d} --verify_dataset_integrity")
    print(f"  nnUNetv2_train {args.dataset_id:03d} 3d_fullres 0  # repeat for folds 1-4")


if __name__ == "__main__":
    main()
