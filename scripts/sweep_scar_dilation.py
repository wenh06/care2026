"""Sweep scar dilation constraint radius on labeled training data.

Usage::

    python scripts/sweep_scar_dilation.py --db-dir <data_root> \\
        --s1 .../Dataset502_*/... --s2 .../Dataset501_*/... \\
        [--output results.txt]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import _load_model
from predict import predict_mri_two_stage


def _binary_metrics(pred, gt, eps=1e-7):
    pb = pred.astype(bool); gb = gt.astype(bool)
    inter = np.logical_and(pb, gb).sum()
    denom = pb.sum() + gb.sum()
    d = (2*inter + eps) / (denom + eps) if denom > 0 else 1.0
    a = float((pb == gb).mean())
    tp = inter; fn = np.logical_and(~pb, gb).sum()
    s = tp / (tp + fn + eps) if (tp + fn) > 0 else 1.0
    return float(d), float(a), float(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", required=True)
    parser.add_argument("--s1", required=True, help="Stage 1 (cavity) model path")
    parser.add_argument("--s2", required=True, help="Stage 2 (scar) model path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dilations", type=str, default="none,2,3,4,5,6,8,10",
                        help="Comma-separated dilation mm values; 'none'=no constraint")
    parser.add_argument("--output", type=str, default=None, help="Write results to file")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    db_dir = Path(args.db_dir)
    task1_dir = db_dir / "LA scar quantification（MRI）" / "train_data"
    records = sorted(
        [d for d in task1_dir.iterdir() if d.is_dir() and d.name.startswith("train_")],
        key=lambda p: int(p.name.split("_")[1]),
    )

    s1 = _load_model("mri_stage1", args.s1, device)
    s2 = _load_model("mri_stage2", args.s2, device)
    s1_mc = bool((getattr(s1, "config", {}) or {}).get("apply_mclahe", False))
    s2_mc = bool((getattr(s2, "config", {}) or {}).get("apply_mclahe", False))
    use_mclahe = s1_mc or s2_mc

    dilations = [None if v.strip().lower() == "none" else float(v.strip()) for v in args.dilations.split(",")]
    results: Dict[str, Dict] = {}

    for dm in dilations:
        dice_vals, acc_vals, sen_vals = [], [], []
        for rec_dir in tqdm(records, desc=f"  d={dm}", unit="case", leave=False):
            img_path = rec_dir / "enhanced.nii.gz"
            gt_path = rec_dir / "scarSegImgM.nii.gz"
            if not img_path.exists() or not gt_path.exists():
                continue
            gt = (nib.load(str(gt_path)).get_fdata() > 0).astype(np.uint8)
            if gt.sum() == 0:
                continue
            out = predict_mri_two_stage(
                img_path, s1, s2, use_tta=False, apply_mclahe=use_mclahe,
                s2_threshold=0.5, scar_dilation=dm,
            )
            pred = out.scar_mask
            if pred.shape != gt.shape:
                t = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                pred = F.interpolate(t, size=gt.shape, mode="nearest").squeeze().numpy().astype(np.uint8)
            d, a, s = _binary_metrics(pred, gt)
            dice_vals.append(d); acc_vals.append(a); sen_vals.append(s)
        label = "none" if dm is None else f"{dm}mm"
        results[label] = {"G-DSC": np.mean(dice_vals), "ACC": np.mean(acc_vals), "SEN": np.mean(sen_vals)}

    # Print unified summary
    lines = []
    lines.append("=" * 45)
    lines.append(f"  Scar Dilation Sweep ({len(records)} cases)")
    lines.append("=" * 45)
    lines.append(f"{'Dilation':>10s}  {'G-DSC':>8s}  {'ACC':>8s}  {'SEN':>8s}")
    lines.append("-" * 42)
    for label in results:
        m = results[label]
        lines.append(f"{label:>10s}  {m['G-DSC']:8.4f}  {m['ACC']:8.4f}  {m['SEN']:8.4f}")
    text = "\n".join(lines)
    print(text)
    if args.output:
        Path(args.output).write_text(text)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
