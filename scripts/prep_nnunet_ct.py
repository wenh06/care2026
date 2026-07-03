"""Convert CARE2026 CT data to nnUNet v2 format.

Creates symlinks to avoid copying large files.

Output:
    nnUNet_raw/Dataset500_CARE2026CT/
      imagesTr/  0050_0000.nii.gz  (modality suffix _0000 for CT)
      labelsTr/  0050.nii.gz
      dataset.json

Usage:
    python scripts/prep_nnunet_ct.py --db-dir /Data1/wenh06/CARE2026-LeftAtrium
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", required=True)
    parser.add_argument("--dataset-id", type=int, default=500, help="nnUNet dataset ID")
    parser.add_argument("--output", default=None, help="nnUNet_raw dir (default: $nnUNet_raw)")
    args = parser.parse_args()

    db_dir = Path(args.db_dir)
    ct_dir = db_dir / "cardiac anatomy segmentation（CT）" / "train_data"

    nnunet_raw = Path(args.output) if args.output else Path(os.environ.get("nnUNet_raw", "tmp/nnUNet_raw"))
    dataset_name = f"Dataset{args.dataset_id:03d}_CARE2026CT"
    out_dir = nnunet_raw / dataset_name
    img_dir = out_dir / "imagesTr"
    lbl_dir = out_dir / "labelsTr"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_labeled = 0
    for d in sorted(ct_dir.iterdir()):
        if not d.is_dir():
            continue
        rec_num = int(d.name.split("_")[1])
        num_str = str(rec_num).zfill(4)
        img_src = d / f"{num_str}.nii.gz"
        lbl_src = d / f"label_{num_str}.nii.gz"
        if not img_src.exists():
            continue
        if lbl_src.exists():
            # Labeled → imagesTr + labelsTr
            os.symlink(img_src.resolve(), img_dir / f"CARE{num_str}_0000.nii.gz")
            os.symlink(lbl_src.resolve(), lbl_dir / f"CARE{num_str}.nii.gz")
            n_labeled += 1

    # dataset.json
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "left_atrium": 1, "pulmonary_veins": 2, "left_atrial_appendage": 3},
        "numTraining": n_labeled,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
    }
    with open(out_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Created {out_dir}")
    print(f"  imagesTr: {len(list(img_dir.glob('*.nii.gz')))} files")
    print(f"  labelsTr: {len(list(lbl_dir.glob('*.nii.gz')))} files")
    print(f"\nNext: nnUNetv2_plan_and_preprocess -d {args.dataset_id:03d} --verify_dataset_integrity -c 3d_fullres")
    print(f"Train single fold: nnUNetv2_train {args.dataset_id:03d} 3d_fullres 0")
    print(f"Train all folds  : for f in 0 1 2 3 4; do nnUNetv2_train {args.dataset_id:03d} 3d_fullres $f; done")


if __name__ == "__main__":
    main()
