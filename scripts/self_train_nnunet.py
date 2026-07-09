"""Self-training: nnUNet N-fold ensemble → pseudo-labels → retrain.

1. N-fold ensemble inference on 100 unlabeled CTs → pseudo-labels
2. Convert to nnUNet v2 format with 150 labeled cases (50 GT + 100 pseudo)
3. Print nnUNet training commands
"""

import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ENV_RAW = "nnUNet_raw"
ENV_PREPROC = "nnUNet_preprocessed"
ENV_RES = "nnUNet_results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", required=True, help="CARE2026 data root")
    parser.add_argument(
        "--nnunet-dir",
        required=True,
        help="nnUNet training output (contains fold_0/..fold_4/, plans.json, dataset.json)",
    )
    parser.add_argument("--nnunet-raw", default=None, help="nnUNet_raw dir (default: $nnUNet_raw)")
    parser.add_argument("--nnunet-preprocessed", default=None, help="For printed commands (default: $nnUNet_preprocessed)")
    parser.add_argument("--nnunet-results", default=None, help="For printed commands (default: $nnUNet_results)")
    parser.add_argument(
        "--folds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated folds to ensemble, e.g. '0' for single-fold pseudo-labels",
    )
    parser.add_argument("--dataset-id", type=int, default=503, help="New nnUNet dataset ID for 150-case training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    args = parser.parse_args()

    # Resolve paths: CLI arg > env var > error
    for key, env, label in [
        ("nnunet_raw", ENV_RAW, "nnUNet_raw"),
        ("nnunet_preprocessed", ENV_PREPROC, "nnUNet_preprocessed"),
        ("nnunet_results", ENV_RES, "nnUNet_results"),
    ]:
        val = getattr(args, key)
        if val is None:
            val = os.environ.get(env)
        if val is None:
            raise RuntimeError(f"--{key} not set and ${env} not defined.  export {env}=... or pass --{key}")
        setattr(args, key, val)
    if args.dataset_id in (500, 501, 502, 511, 512, 521):
        raise ValueError(f"--dataset-id {args.dataset_id} conflicts with existing datasets.  Use 503 or higher.")

    db_dir = Path(args.db_dir)
    ct_dir = db_dir / "cardiac anatomy segmentation（CT）" / "train_data"
    nnunet_raw = Path(args.nnunet_raw)

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
    use_folds = tuple(int(f.strip()) for f in args.folds.split(","))
    predictor.initialize_from_trained_model_folder(
        args.nnunet_dir,
        use_folds=use_folds,
        checkpoint_name=args.checkpoint,
    )

    # Save pseudo-labels to a separate dir, not touching original data
    pseudo_dir = nnunet_raw.parent / "pseudo_labels"
    pseudo_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating pseudo-labels for {len(unlabeled_cases)} unlabeled cases...")
    for d, rec_num, num_str, img_src in tqdm(unlabeled_cases, desc="Pseudo-labeling", unit="case"):
        nii = nib.load(str(img_src))
        img = nii.get_fdata().astype(np.float32)
        zooms = tuple(nii.header.get_zooms()[:3])
        spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))

        # Axis transpose for nnUNet: (x,y,z) -> (z,y,x)
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

        # Save pseudo-label in separate directory
        nii_out = nib.Nifti1Image(pred, affine=nii.affine, header=nii.header)
        nib.save(nii_out, str(pseudo_dir / f"label_{num_str}.nii.gz"))

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
        lbl_src = pseudo_dir / f"label_{num_str}.nii.gz"  # just generated
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

    # ── Step 4: Print commands ───────────────────────────────────────
    raw = args.nnunet_raw
    preproc = args.nnunet_preprocessed
    results = args.nnunet_results
    print(f"\nCreated {out_dir} ({n_linked} training cases)")
    print("\nRun the following after setting env vars:")
    print(f"  export nnUNet_raw='{raw}'")
    print(f"  export nnUNet_preprocessed='{preproc}'")
    print(f"  export nnUNet_results='{results}'")
    print(f"  nnUNetv2_plan_and_preprocess -d {args.dataset_id:03d} --verify_dataset_integrity -c 3d_fullres")
    print(f"  nnUNetv2_train {args.dataset_id:03d} 3d_fullres 0")
    print(f"  for f in 0 1 2 3 4; do nnUNetv2_train {args.dataset_id:03d} 3d_fullres $f; done")


if __name__ == "__main__":
    main()
