"""Sweep scar dilation constraint radius on labeled training data.

Runs inference ONCE, then applies different dilation values offline
to the cached raw scar + LA predictions.  Metrics match the paper's
training-set convention (per-case Dice/ACC/SEN averaged over cases,
see eval_all_models.py), so the sweep baseline (dilation=none) is
directly comparable with the reported train G-DSC values.

Usage::

    python scripts/sweep_scar_dilation.py --db-dir <data_root> \\
        --s1 .../Dataset502_*/... --s2 .../Dataset521_*/... \\
        --pipeline hybrid --dilations none,3,5,7 [--output results.txt]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import _load_model
from predict import keep_largest_component, predict_mri_two_stage, predict_mri_two_stage_hybrid


def _binary_metrics(pred, gt, eps=1e-7, metric: str = "dice"):
    """Per-case metrics, averaged over cases — matches eval_all_models.py /
    eval_all_nnunet.py, i.e. the metric used for all training-set results in the paper
    (labelled "G-DSC" there).  Kept identical so the camera-ready rows are directly
    comparable with the reported baselines (e.g. train 0.6631).

    ``metric="gdsc"`` computes the official CARE Task-1 G-DSC (w_c = 1/|GT_c|^2)
    instead of per-case Dice, for cross-checking against the platform convention.
    """
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    denom = pred_b.sum() + gt_b.sum()
    if metric == "gdsc":
        # w_c = 1 / |gt_c|^2  for each class; G-DSC = 2 * sum(w_c * |p_c ∩ g_c|) / sum(w_c * (|p_c| + |g_c|))
        p_bg = ~pred_b
        g_bg = ~gt_b
        w_bg = 1.0 / (max(g_bg.sum(), 1) ** 2)
        w_sc = 1.0 / (max(gt_b.sum(), 1) ** 2)
        inter_bg = np.logical_and(p_bg, g_bg).sum()
        inter_sc = inter
        union_bg = p_bg.sum() + g_bg.sum()
        union_sc = denom
        numerator = 2 * (w_bg * inter_bg + w_sc * inter_sc)
        denominator = w_bg * union_bg + w_sc * union_sc
        dice = float(numerator / denominator) if denominator > 0 else 1.0
    else:
        dice = (2 * inter + eps) / (denom + eps) if denom > 0 else 1.0
    acc = float((pred_b == gt_b).mean())
    tp = inter
    fn = np.logical_and(~pred_b, gt_b).sum()
    sens = (tp + eps) / (tp + fn + eps) if (tp + fn) > 0 else 1.0
    return float(dice), float(acc), float(sens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", required=True)
    parser.add_argument("--s1", required=True, help="Stage 1 (cavity) model path")
    parser.add_argument("--s2", required=True, help="Stage 2 (scar) model path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dilations", type=str, default="none,2,3,4,5,6,8,10")
    parser.add_argument(
        "--pipeline",
        choices=["native", "hybrid"],
        default="native",
        help="Inference pipeline: native (both stages native resolution) or hybrid (mixed-spacing)",
    )
    parser.add_argument(
        "--metric",
        choices=["dice", "gdsc"],
        default="dice",
        help="Per-case metric: 'dice' (paper convention, default) or 'gdsc' (official CARE G-DSC, for cross-checking)",
    )
    parser.add_argument("--output", type=str, default=None)
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

    # ── Phase 1: Run inference ONCE, cache raw scar + LA predictions ──
    cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}  # rec → (la, scar, gt)
    for rec_dir in tqdm(records, desc="Inference", unit="case"):
        img_path = rec_dir / "enhanced.nii.gz"
        gt_path = rec_dir / "scarSegImgM.nii.gz"
        if not img_path.exists() or not gt_path.exists():
            continue
        gt = (nib.load(str(gt_path)).get_fdata() > 0).astype(np.uint8)
        if gt.sum() == 0:
            continue
        # Get raw scar prediction (no post-processing)
        _predict_fn = {
            "native": predict_mri_two_stage,
            "hybrid": predict_mri_two_stage_hybrid,
        }[args.pipeline]
        out = _predict_fn(
            img_path,
            s1,
            s2,
            use_tta=False,
            apply_mclahe=use_mclahe,
            s2_threshold=0.5,
            scar_dilation=None,  # None = no LA constraint
        )
        # Keep largest LA component once (invariant across dilations)
        la_raw = out.la_mask
        la_clean = keep_largest_component(la_raw)
        scar_raw = out.scar_mask
        if scar_raw.shape != gt.shape:
            t = torch.from_numpy(scar_raw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            scar_raw = F.interpolate(t, size=gt.shape, mode="nearest").squeeze().numpy().astype(np.uint8)
            t = torch.from_numpy(la_clean.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            la_clean = F.interpolate(t, size=gt.shape, mode="nearest").squeeze().numpy().astype(np.uint8)
        cache[rec_dir.name] = (la_clean, scar_raw, gt)

    # ── Phase 2: Dilate LA → AND with scar (no connected components, near-instant) ──
    spacing_xy = 0.625
    results: Dict[str, Dict] = {}
    for dm in tqdm(dilations, desc="Sweep dilation", unit="dm"):
        dice_vals, acc_vals, sen_vals = [], [], []
        if dm is not None and dm > 0:
            dp = max(1, int(round(dm / spacing_xy)))
            structure = np.ones((dp, dp, 1), dtype=bool)
        for rec_name, (la_clean, scar_raw, gt) in cache.items():
            if dm is None or dm <= 0:
                scar_proc = scar_raw
            else:
                la_dilated = binary_dilation(la_clean.astype(bool), structure=structure, iterations=1)
                scar_proc = (scar_raw.astype(bool) & la_dilated).astype(np.uint8)
            d, a, s = _binary_metrics(scar_proc, gt, metric=args.metric)
            dice_vals.append(d)
            acc_vals.append(a)
            sen_vals.append(s)
        label = "none" if dm is None else f"{dm}mm"
        results[label] = {"G-DSC": np.mean(dice_vals), "ACC": np.mean(acc_vals), "SEN": np.mean(sen_vals)}

    # ── Print summary ──
    lines = []
    lines.append("=" * 45)
    lines.append(f"  Scar Dilation Sweep ({len(cache)} cases)")
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
