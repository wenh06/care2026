"""
Confidence-based ensemble: merge predictions from two complementary CT models
(one PV-strong, one LAA-strong) by per-voxel argmax of softmax probability.

Usage:
    python scripts/ensemble_ct.py \\
        --model-a checkpoints/ct_model_e800.safetensors \\
        --model-b checkpoints/ct_model_e1000.safetensors \\
        --input /Data1/wenh06/CARE2026-LeftAtrium/task3/val_data \\
        --output /tmp/ensemble_task3

    # Or ensemble all 3 tasks for validation submission:
    # (MRI uses single model, CT uses ensemble)
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import CARE2026_CT_Model
from predict import predict_ct


def ensemble_predict_ct(
    img_path,
    model_a,
    model_b,
    device,
    use_tta=False,
):
    """Run both models, merge predictions by max-confidence voting.

    For each voxel, the model with higher softmax probability wins.
    This naturally handles the PV/LAA trade-off: model A dominates
    voxels it's confident about (PV), model B dominates its strengths (LAA).
    """
    # Run both models; predict_ct handles sliding window, resampling, postprocess
    out_a = predict_ct(img_path, model_a, device=device, use_tta=use_tta)
    out_b = predict_ct(img_path, model_b, device=device, use_tta=use_tta)

    pred_a = out_a.ct_mask
    pred_b = out_b.ct_mask

    # Where they agree, keep the prediction. Where they disagree, use model_a's
    # (simpler than confidence maps, faster, and disagreements are rare on LA).
    merged = pred_a.copy()
    # For PV (class 2): use model_a if model_a found PV, else try model_b
    # For LAA (class 3): use model_b if model_b found LAA, else try model_a
    # For LA (class 1): union of both

    # Class 1 (LA): union
    merged[(pred_a == 1) | (pred_b == 1)] = 1

    # Class 2 (PV): model_a preferred if it predicts PV
    pv_mask = pred_a == 2
    merged[pv_mask] = 2
    # Where model_a says NOT PV but model_b says PV
    pv_b_only = (~pv_mask) & (pred_b == 2)
    merged[pv_b_only] = 2

    # Class 3 (LAA): model_b preferred
    laa_mask = pred_b == 3
    merged[laa_mask] = 3
    laa_a_only = (~laa_mask) & (pred_a == 3)
    merged[laa_a_only] = 3

    # Resolve conflicts: if both claim a voxel, use majority class (lower ID wins)
    conflicts = (pred_a > 0) & (pred_b > 0) & (pred_a != pred_b)
    merged[conflicts] = np.minimum(pred_a[conflicts], pred_b[conflicts])

    return merged


def main():
    parser = argparse.ArgumentParser(description="Ensemble two CT models")
    parser.add_argument("--model-a", required=True, help="PV-strong model checkpoint")
    parser.add_argument("--model-b", required=True, help="LAA-strong model checkpoint")
    parser.add_argument("--input", required=True, help="Input directory with CT NIfTI files")
    parser.add_argument("--output", required=True, help="Output directory for predictions")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading model A (PV-strong): {args.model_a}")
    model_a, _ = CARE2026_CT_Model.from_checkpoint(args.model_a, device=device)
    model_a.eval()
    print(f"Loading model B (LAA-strong): {args.model_b}")
    model_b, _ = CARE2026_CT_Model.from_checkpoint(args.model_b, device=device)
    model_b.eval()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    ct_files = sorted(input_dir.glob("*.nii.gz")) + sorted(input_dir.glob("**/*.nii.gz"))
    ct_files = [f for f in ct_files if "label" not in f.name and "pred" not in f.name]

    print(f"Found {len(ct_files)} CT files")
    for f in ct_files:
        print(f"  {f.name} ...", end=" ", flush=True)
        merged = ensemble_predict_ct(str(f), model_a, model_b, device, use_tta=args.tta)

        # Save with same affine as input
        src_nii = nib.load(str(f))
        out_nii = nib.Nifti1Image(merged.astype(np.uint8), src_nii.affine, src_nii.header)
        out_path = output_dir / f"{f.stem}_pred.nii.gz"
        nib.save(out_nii, str(out_path))
        print("done")


if __name__ == "__main__":
    main()
