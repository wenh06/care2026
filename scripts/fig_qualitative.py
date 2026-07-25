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


def _normalize_image(img: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """Percentile-based normalization for display."""
    vmin, vmax = np.percentile(img, [p_low, p_high])
    return np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)


def _plot_case(ax_row, img_slice: np.ndarray, scar_gt: np.ndarray, scar_pred: np.ndarray, slice_idx: int):
    """Plot one case row: [Image+GT, Image+Pred, Diff map]."""
    # --- Column 1: Image + GT outline ---
    ax = ax_row[0]
    ax.imshow(img_slice, cmap="gray", vmin=0, vmax=1)
    gt_mask = np.ma.masked_where(scar_gt == 0, scar_gt)
    ax.imshow(gt_mask, cmap="Greens", alpha=0.7, vmin=0, vmax=1)
    ax.axis("off")

    # --- Column 2: Image + Pred outline ---
    ax = ax_row[1]
    ax.imshow(img_slice, cmap="gray", vmin=0, vmax=1)
    pred_mask = np.ma.masked_where(scar_pred == 0, scar_pred)
    ax.imshow(pred_mask, cmap="Reds", alpha=0.7, vmin=0, vmax=1)
    ax.axis("off")

    # --- Column 3: Difference map ---
    ax = ax_row[2]
    diff = np.zeros((*scar_gt.shape, 3), dtype=np.float32)
    tp = (scar_pred > 0) & (scar_gt > 0)
    fp = (scar_pred > 0) & (scar_gt == 0)
    fn = (scar_pred == 0) & (scar_gt > 0)
    diff[tp] = [0.2, 0.8, 0.2]  # green: correct
    diff[fp] = [0.9, 0.2, 0.2]  # red: FP
    diff[fn] = [0.2, 0.4, 0.9]  # blue: FN
    diff_bg = img_slice[..., None] * 0.3
    diff_out = diff * 0.7 + diff_bg * 0.7
    ax.imshow(diff_out)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative comparison figure for the paper.")
    parser.add_argument("--db-dir", required=True, help="CARE2026 data root")
    parser.add_argument("--s1", required=True, help="Phase 1 (cavity) model path")
    parser.add_argument("--s2", required=True, help="Phase 2 (scar) model path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=str, default="figures/qualitative.pdf", help="Output figure path")
    parser.add_argument("--n-cases", type=int, default=3, help="Number of cases to show (default: 3)")
    args = parser.parse_args()

    db_dir = Path(args.db_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = args.device

    s1 = _load_model("mri_stage1", args.s1, device)
    s2 = _load_model("mri_stage2", args.s2, device)
    s1_mc = bool((getattr(s1, "config", {}) or {}).get("apply_mclahe", False))
    s2_mc = bool((getattr(s2, "config", {}) or {}).get("apply_mclahe", False))
    use_mclahe = s1_mc or s2_mc

    # Run inference on all training cases and collect per-case metrics
    task1_dir = db_dir / "LA scar quantification（MRI）" / "train_data"
    records = sorted(
        [d for d in task1_dir.iterdir() if d.is_dir() and d.name.startswith("train_")],
        key=lambda p: int(p.name.split("_")[1]),
    )

    print(f"Running inference on {len(records)} training cases...")
    per_case: List[Dict] = []
    for rec_dir in records:
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
        # Find representative slices (most scar)
        best_z = int(np.argmax(gt.sum(axis=(1, 2))))
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

    # Sort by G-DSC and select representative cases: best, median, worst
    per_case.sort(key=lambda x: x["dice"])
    n = len(per_case)
    selected_indices = [0, n // 2, n - 1]  # worst, median, best
    selected = [per_case[i] for i in selected_indices]

    labels = ["Worst", "Median", "Best"]
    print("\nSelected cases for figure:")
    for label, case in zip(labels, selected):
        print(f"  {label}: {case['name']} (G-DSC = {case['dice']:.4f})")

    # Build figure: rows = cases, cols = 3 (Image+GT, Image+Pred, Diff)
    n_cases = len(selected)
    fig, axes = plt.subplots(n_cases, 3, figsize=(6.5, 2.2 * n_cases))

    if n_cases == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Image + GT", "Image + Pred", "TP / FP / FN"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10, fontweight="bold")

    for i, case in enumerate(selected):
        img_norm = _normalize_image(case["img"])
        z = case["best_z"]
        _plot_case(
            axes[i],
            img_norm[z],
            case["gt"][z],
            case["pred"][z],
            z,
        )
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
