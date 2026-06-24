"""
Quick CT model inference test — load checkpoint, run on a few labelled
records, report per-class Dice scores.

Usage:
    python scripts/test_ct_inference.py \\
        --checkpoint checkpoints/ct_model.safetensors \\
        --db-dir /Data1/wenh06/CARE2026-LeftAtrium \\
        --num-records 3
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from data_reader import CARE2026_CT
from models import CARE2026_CT_Model
from predict import predict_ct


def simple_dice(pred, gt, label):
    p = pred == label
    g = gt == label
    intersection = (p & g).sum()
    union = p.sum() + g.sum()
    if union == 0:
        return 1.0
    return 2.0 * intersection / union


def main():
    parser = argparse.ArgumentParser(description="Quick CT model inference test")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.safetensors)")
    parser.add_argument("--db-dir", required=True, help="Path to CARE2026 dataset root")
    parser.add_argument("--num-records", type=int, default=3, help="Number of labelled records to test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("Loading model...")
    device = torch.device(args.device)
    model, train_cfg = CARE2026_CT_Model.from_checkpoint(args.checkpoint, device=device)
    model.eval()
    print("Model loaded successfully.")
    print("Model Mode:", model.mode)
    print("Model config:", model.config)
    print("Model train config:", model.train_config)

    reader = CARE2026_CT(db_dir=args.db_dir, verbose=0)
    labeled_recs = reader.labeled_records
    print(f"Number of labeled records: {len(labeled_recs)}")

    for i in range(min(args.num_records, len(labeled_recs))):
        rec = labeled_recs[i]
        print(f"\nEvaluating {rec}...")
        img_path = reader.get_data_path(rec)
        gt_mask = reader.load_ann(rec)

        print("Running inference...")
        out = predict_ct(img_path, model, device=device, use_tta=False)
        pred_mask = out.ct_mask

        print(f"Ground truth shapes: {gt_mask.shape}, Predictions shape: {pred_mask.shape}")
        print(f"Ground truth unique values: {np.unique(gt_mask)}")
        print(f"Predictions unique values: {np.unique(pred_mask)}")

        for label, name in {1: "LA", 2: "PV", 3: "LAA"}.items():
            dice = simple_dice(pred_mask, gt_mask, label)
            gt_count = (gt_mask == label).sum()
            pred_count = (pred_mask == label).sum()
            print(f"  {name} (label {label}): Dice = {dice:.4f} | GT voxels = {gt_count} | Pred voxels = {pred_count}")


if __name__ == "__main__":
    main()
