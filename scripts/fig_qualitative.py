"""Generate qualitative comparison figure for the paper.

Runs inference on selected Task~1 training cases and creates a figure showing
original LGE-MRI, ground-truth scar, predicted scar, and difference map (FP/FN).

Usage::

    python scripts/fig_qualitative.py --db-dir <CARE2026_data_root> \\
        --nnunet-results tmp/nnUNet_results \\
        --output figures/qualitative.pdf
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import _load_model
from predict import predict_mri_two_stage_hybrid

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


def _binary_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> Tuple[float, float, float]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    denom = pred_b.sum() + gt_b.sum()
    dice = (2 * inter + eps) / (denom + eps) if denom > 0 else 1.0
    return float(dice), float(inter), float(denom)


def _crop_roi(img_slice, gt_slice, pred_slice, margin=20):
    """Crop to the bounding box of GT and predicted scar, plus margin."""
    roi = (gt_slice > 0) | (pred_slice > 0)
    if roi.sum() == 0:
        return img_slice, gt_slice, pred_slice
    rows = np.any(roi, axis=1)
    cols = np.any(roi, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = max(0, rmin - margin)
    rmax = min(img_slice.shape[0], rmax + margin + 1)
    cmin = max(0, cmin - margin)
    cmax = min(img_slice.shape[1], cmax + margin + 1)
    return img_slice[rmin:rmax, cmin:cmax], gt_slice[rmin:rmax, cmin:cmax], pred_slice[rmin:rmax, cmin:cmax]


def _normalize_image(img: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """Percentile-based normalization for display."""
    vmin, vmax = np.percentile(img, [p_low, p_high])
    return np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)


def _draw_mask(ax, mask, color, alpha=0.25):
    """Draw a binary mask as a filled contour overlay (viz.py style)."""
    if mask.max() == 0:
        return
    ax.contourf(mask, levels=[0.5, 1], colors=[color], alpha=alpha, antialiased=True)


def _plot_case(ax_row, img_slice: np.ndarray, scar_gt: np.ndarray, scar_pred: np.ndarray):
    """Plot one case row: [Image+GT, Image+Pred, Diff map]."""
    # --- Column 1: Image + GT ---
    ax = ax_row[0]
    ax.imshow(img_slice, cmap="gray", origin="lower")
    _draw_mask(ax, scar_gt, "#00FF00")
    ax.axis("off")

    # --- Column 2: Image + Pred ---
    ax = ax_row[1]
    ax.imshow(img_slice, cmap="gray", origin="lower")
    _draw_mask(ax, scar_pred, "#FF4444")
    ax.axis("off")

    # --- Column 3: Difference map (TP / FP / FN) ---
    ax = ax_row[2]
    ax.imshow(img_slice, cmap="gray", origin="lower")
    tp = (scar_pred > 0) & (scar_gt > 0)
    fp = (scar_pred > 0) & (scar_gt == 0)
    fn = (scar_pred == 0) & (scar_gt > 0)
    _draw_mask(ax, tp.astype(np.uint8), "#00FF00", alpha=0.30)  # green: correct
    _draw_mask(ax, fp.astype(np.uint8), "#FF4444", alpha=0.30)  # red: FP
    _draw_mask(ax, fn.astype(np.uint8), "#4488FF", alpha=0.30)  # blue: FN
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative comparison figure for the paper.")
    parser.add_argument("--db-dir", required=True, help="CARE2026 data root")
    parser.add_argument("--s1", required=True, help="Phase 1 (cavity) model path")
    parser.add_argument("--s2", required=True, help="Phase 2 (scar) model path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=str, default="figures/qualitative.pdf", help="Output figure path")
    parser.add_argument("--n-cases", type=int, default=3, help="Number of cases to show (default: 3)")
    parser.add_argument(
        "--per-case-csv",
        type=str,
        default=None,
        help="Optional CSV with per-case G-DSC (avoids full inference; only selected cases are inferred).",
    )
    args = parser.parse_args()

    db_dir = Path(args.db_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = args.device

    task1_dir = db_dir / "LA scar quantification（MRI）" / "train_data"

    # --- Determine which cases to infer ---
    if args.per_case_csv:
        # Read pre-computed G-DSC, select worst/median/best
        import csv

        rows = []
        with open(args.per_case_csv, newline="") as f:
            for row in csv.DictReader(f):
                rows.append((row["case"], float(row["G-DSC"])))
        rows.sort(key=lambda x: x[1])
        n = len(rows)
        selected_names = [rows[i][0] for i in [0, n // 2, n - 1]]
        labels = ["Worst", "Median", "Best"]
        print("Selected from CSV:")
        for label, name in zip(labels, selected_names):
            gdsc = [r[1] for r in rows if r[0] == name][0]
            print(f"  {label}: {name} (G-DSC = {gdsc:.4f})")
        target_names = set(selected_names)
    else:
        target_names = None  # infer all, then select

    # --- Load models ---
    s1 = _load_model("mri_stage1", args.s1, device)
    s2 = _load_model("mri_stage2", args.s2, device)
    s1_mc = bool((getattr(s1, "config", {}) or {}).get("apply_mclahe", False))
    s2_mc = bool((getattr(s2, "config", {}) or {}).get("apply_mclahe", False))
    use_mclahe = s1_mc or s2_mc

    # --- Run inference ---
    records = sorted(
        [d for d in task1_dir.iterdir() if d.is_dir() and d.name.startswith("train_")],
        key=lambda p: int(p.name.split("_")[1]),
    )

    print(f"Running inference on {len(records) if target_names is None else len(target_names)} training cases...")
    per_case: List[Dict] = []
    for rec_dir in records:
        if target_names is not None and rec_dir.name not in target_names:
            continue
        img_path = rec_dir / "enhanced.nii.gz"
        gt_path = rec_dir / "scarSegImgM.nii.gz"
        if not img_path.exists() or not gt_path.exists():
            continue
        gt = (nib.load(str(gt_path)).get_fdata() > 0).astype(np.uint8)
        if gt.sum() == 0:
            continue

        out = predict_mri_two_stage_hybrid(img_path, s1, s2, use_tta=False, apply_mclahe=use_mclahe, s2_threshold=0.5)
        pred = out.scar_mask
        if pred.shape != gt.shape:
            import torch
            import torch.nn.functional as F

            t = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            pred = F.interpolate(t, size=gt.shape, mode="nearest").squeeze().numpy().astype(np.uint8)

        dice, _, _ = _binary_metrics(pred, gt)
        img = nib.load(str(img_path)).get_fdata().astype(np.float32)
        best_z = int(np.argmax(gt.sum(axis=(0, 1))))
        per_case.append(
            {
                "name": rec_dir.name,
                "img": img,
                "gt": gt,
                "pred": pred,
                "dice": dice,
                "best_z": best_z,
            }
        )
        print(f"  {rec_dir.name}: G-DSC = {dice:.4f}")

    # Sort by G-DSC and select
    per_case.sort(key=lambda x: x["dice"])
    if target_names is None:
        n = len(per_case)
        indices = [0, n // 2, n - 1]
        selected = [per_case[i] for i in indices]
        labels = ["Worst", "Median", "Best"]
    else:
        selected = sorted(per_case, key=lambda x: x["dice"])
        # labels already set from CSV selection
    print("\nSelected cases for figure:")
    for label, case in zip(labels, selected):
        print(f"  {label}: {case['name']} (G-DSC = {case['dice']:.4f})")

    # Build figure: rows = cases, cols = 3 (Image+GT, Image+Pred, Diff)
    n_cases = len(selected)
    # Pre-crop to determine panel dimensions
    cropped = []
    for case in selected:
        img_norm = _normalize_image(case["img"])
        z = case["best_z"]
        cropped.append(_crop_roi(img_norm[..., z], case["gt"][..., z], case["pred"][..., z]))

    max_h = max(c[0].shape[0] for c in cropped)
    max_w = max(c[0].shape[1] for c in cropped)
    panel_aspect = max_h / max_w if max_w > 0 else 1.0
    panel_w = 2.2  # inches per panel
    panel_h = panel_w * panel_aspect
    fig, axes = plt.subplots(n_cases, 3, figsize=(3 * panel_w, n_cases * panel_h))

    if n_cases == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Image + GT", "Image + Pred", "TP / FP / FN"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10, fontweight="bold")

    for i, (case, (img_s, gt_s, pred_s)) in enumerate(zip(selected, cropped)):
        _plot_case(axes[i], img_s, gt_s, pred_s)
        # Row label
        axes[i, 0].set_ylabel(
            f"{labels[i]}\n({case['name']}, G-DSC={case['dice']:.3f})",
            fontsize=9,
            rotation=0,
            ha="right",
            va="center",
            labelpad=5,
        )

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"\nFigure saved to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
