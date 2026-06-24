"""
Comprehensive CT model diagnostic: load checkpoint, run sliding-window
inference on all labelled training records, compare against ground truth.

Usage:
    python scripts/diagnose_ct.py \\
        --checkpoint checkpoints/ct_model.safetensors \\
        --db-dir /Data1/wenh06/CARE2026-LeftAtrium \\
        --max-records 50
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from const import CT_PATCH_SIZE
from data_reader import CARE2026_CT
from models import CARE2026_CT_Model
from predict import predict_ct


def dice_score(pred, gt, label):
    """Per-class Dice coefficient."""
    p = pred == label
    g = gt == label
    intersection = (p & g).sum()
    union = p.sum() + g.sum()
    if union == 0:
        return 1.0
    return 2.0 * intersection / union


def iou_score(pred, gt, label):
    """Per-class IoU / Jaccard."""
    p = pred == label
    g = gt == label
    intersection = (p & g).sum()
    union = p.sum() + g.sum() - intersection
    if union == 0:
        return 1.0
    return intersection / union


def sensitivity(pred, gt, label):
    """Per-class sensitivity / recall."""
    p = pred == label
    g = gt == label
    tp = (p & g).sum()
    fn = ((~p) & g).sum()
    denom = tp + fn
    if denom == 0:
        return 1.0
    return tp / denom


def precision(pred, gt, label):
    """Per-class precision."""
    p = pred == label
    g = gt == label
    tp = (p & g).sum()
    fp = (p & (~g)).sum()
    denom = tp + fp
    if denom == 0:
        return 1.0
    return tp / denom


CLASS_NAMES = {0: "BG", 1: "LA", 2: "PV", 3: "LAA"}


def main():
    parser = argparse.ArgumentParser(description="CT model comprehensive diagnostic")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.safetensors)")
    parser.add_argument("--db-dir", required=True, help="Path to CARE2026 dataset root")
    parser.add_argument("--max-records", type=int, default=50, help="Max labelled records to evaluate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-tta", action="store_true", default=False, help="Disable test-time augmentation")
    args = parser.parse_args()

    device = torch.device(args.device)
    use_tta = not args.no_tta
    print(f"Device: {device}, TTA: {use_tta}")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"\nLoading model from {args.checkpoint}...")
    model, train_cfg = CARE2026_CT_Model.from_checkpoint(args.checkpoint, device=device)
    model.eval()

    print(f"  Mode: {model.mode}")
    print(f"  Normalization: {train_cfg.get('normalization', 'N/A')}")

    for k in ("backbone", "patch_size", "semi_supervised_mode", "n_epochs", "optimizer", "lr"):
        print(f"  train_cfg.{k}: {train_cfg.get(k, 'N/A')}")
    print(f"  model num_classes: {model.config.vnet_ct.get('num_classes', 'N/A')}")
    print(f"  model activation: {model.config.vnet_ct.get('activation', 'N/A')}")
    print(f"  model norm: {model.config.vnet_ct.get('norm', 'N/A')}")
    print(f"  model params: {sum(p.numel() for p in model.parameters()):,}")

    # Check BN running stats (first layer)
    for name, param in model.named_parameters():
        if "running_mean" in name:
            rmean = param.data.cpu().numpy()
            print(f"  {name}: mean={rmean.mean():.4f}, std={rmean.std():.4f}, min={rmean.min():.4f}, max={rmean.max():.4f}")
        if "running_var" in name:
            rvar = param.data.cpu().numpy()
            print(f"  {name}: mean={rvar.mean():.4f}, std={rvar.std():.4f}, min={rvar.min():.4f}, max={rvar.max():.4f}")
        break  # Just first one for sanity

    # ------------------------------------------------------------------
    # 2. Load data reader
    # ------------------------------------------------------------------
    reader = CARE2026_CT(db_dir=args.db_dir, verbose=0)
    labeled_recs = reader.labeled_records
    print(f"\nLabeled records: {len(labeled_recs)}")

    recs_to_test = labeled_recs[: args.max_records]
    print(f"Testing on {len(recs_to_test)} records")

    # ------------------------------------------------------------------
    # 3. Run inference + compute metrics
    # ------------------------------------------------------------------
    all_metrics = defaultdict(list)

    for i, rec in enumerate(recs_to_test):
        img_path = reader.get_data_path(rec)
        gt_mask = reader.load_ann(rec)

        print(
            f"\n[{i+1}/{len(recs_to_test)}] {rec} — GT shape={gt_mask.shape}, " f"GT unique={np.unique(gt_mask).tolist()}",
            end="",
            flush=True,
        )

        try:
            out = predict_ct(
                img_path,
                model,
                device=device,
                use_tta=use_tta,
                patch_size=CT_PATCH_SIZE,
                stride=CT_PATCH_SIZE // 2,
            )
            pred_mask = out.ct_mask
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback

            traceback.print_exc()
            continue

        # Check prediction sanity
        pred_unique, pred_counts = np.unique(pred_mask, return_counts=True)
        gt_unique, gt_counts = np.unique(gt_mask, return_counts=True)

        print(f"\n  Pred unique={dict(zip(pred_unique.tolist(), pred_counts.tolist()))}")
        print(f"  GT   unique={dict(zip(gt_unique.tolist(), gt_counts.tolist()))}")

        # Per-class metrics
        for label in [1, 2, 3]:
            name = CLASS_NAMES[label]
            dsc = dice_score(pred_mask, gt_mask, label)
            iou = iou_score(pred_mask, gt_mask, label)
            sen = sensitivity(pred_mask, gt_mask, label)
            prec = precision(pred_mask, gt_mask, label)

            gt_count = (gt_mask == label).sum()
            pred_count = (pred_mask == label).sum()

            all_metrics[f"{name}_dice"].append(dsc)
            all_metrics[f"{name}_iou"].append(iou)
            all_metrics[f"{name}_sen"].append(sen)
            all_metrics[f"{name}_prec"].append(prec)
            all_metrics[f"{name}_gt_voxels"].append(gt_count)
            all_metrics[f"{name}_pred_voxels"].append(pred_count)

            print(f"  {name}: Dice={dsc:.4f} IoU={iou:.4f} Sen={sen:.4f} Prec={prec:.4f} " f"GT={gt_count} Pred={pred_count}")

        all_metrics["mean_dice"].append(
            np.mean(
                [
                    all_metrics["LA_dice"][-1],
                    all_metrics["PV_dice"][-1],
                    all_metrics["LAA_dice"][-1],
                ]
            )
        )

    # ------------------------------------------------------------------
    # 4. Aggregate statistics
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*80}")
    print(f"{'Class':<6} {'Dice':>8} {'IoU':>8} {'Sen':>8} {'Prec':>8} {'GT vox':>10} {'Pred vox':>10}")
    print("-" * 60)

    for label in [1, 2, 3]:
        name = CLASS_NAMES[label]
        dsc = np.array(all_metrics[f"{name}_dice"])
        iou = np.array(all_metrics[f"{name}_iou"])
        sen = np.array(all_metrics[f"{name}_sen"])
        prec = np.array(all_metrics[f"{name}_prec"])
        gt_vox = np.array(all_metrics[f"{name}_gt_voxels"])
        pred_vox = np.array(all_metrics[f"{name}_pred_voxels"])

        print(
            f"{name:<6} {dsc.mean():8.4f} {iou.mean():8.4f} {sen.mean():8.4f} {prec.mean():8.4f} "
            f"{gt_vox.mean():10.0f} {pred_vox.mean():10.0f}"
        )
        print(f"{'':6} ±{dsc.std():7.4f} ±{iou.std():7.4f} ±{sen.std():7.4f} ±{prec.std():7.4f}")
        print(f"{'':6} min={dsc.min():.4f} max={dsc.max():.4f}  " f"(records: {np.argmin(dsc)},{np.argmax(dsc)})")

    mean_dsc = np.array(all_metrics["mean_dice"])
    print(f"\n{'Mean':<6} {mean_dsc.mean():8.4f} ±{mean_dsc.std():.4f}  " f"min={mean_dsc.min():.4f} max={mean_dsc.max():.4f}")

    # ------------------------------------------------------------------
    # 5. Detailed per-record summary (worst and best)
    # ------------------------------------------------------------------
    sorted_idx = np.argsort(mean_dsc)

    print(f"\n{'='*80}")
    print("WORST 5 RECORDS")
    print(f"{'='*80}")
    for rank, idx in enumerate(sorted_idx[:5]):
        rec = recs_to_test[idx]
        print(
            f"  {rank+1}. {rec}: mean_dice={mean_dsc[idx]:.4f}  "
            f"LA={all_metrics['LA_dice'][idx]:.4f}  "
            f"PV={all_metrics['PV_dice'][idx]:.4f}  "
            f"LAA={all_metrics['LAA_dice'][idx]:.4f}"
        )

    print(f"\n{'='*80}")
    print("BEST 5 RECORDS")
    print(f"{'='*80}")
    for rank, idx in enumerate(sorted_idx[-5:][::-1]):
        rec = recs_to_test[idx]
        print(
            f"  {rank+1}. {rec}: mean_dice={mean_dsc[idx]:.4f}  "
            f"LA={all_metrics['LA_dice'][idx]:.4f}  "
            f"PV={all_metrics['PV_dice'][idx]:.4f}  "
            f"LAA={all_metrics['LAA_dice'][idx]:.4f}"
        )

    # ------------------------------------------------------------------
    # 6. Degeneracy checks
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("DEGENERACY CHECKS")
    print(f"{'='*80}")
    for label in [1, 2, 3]:
        name = CLASS_NAMES[label]
        pred_vox = np.array(all_metrics[f"{name}_pred_voxels"])
        gt_vox = np.array(all_metrics[f"{name}_gt_voxels"])
        n_empty_pred = (pred_vox == 0).sum()
        n_empty_gt = (gt_vox == 0).sum()
        print(
            f"  {name}: records with zero pred={n_empty_pred}/{len(pred_vox)}  "
            f"zero GT={n_empty_gt}/{len(gt_vox)}  "
            f"pred/GT ratio mean={np.mean(pred_vox / (gt_vox + 1)):.2f}"
        )


if __name__ == "__main__":
    main()
