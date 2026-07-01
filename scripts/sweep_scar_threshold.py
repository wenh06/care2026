"""Scar threshold sweep — find optimal binary threshold for G-DSC.

G-DSC weights each class by 1/|GT|², so scar (tiny class) gets huge weight.
Lower threshold → higher sensitivity (finds more scar) but lower precision
(more false positives).  The optimal balance for G-DSC is often below 0.5.

Usage:
    python scripts/sweep_scar_threshold.py --db-dir /Data1/... --thresholds 0.1,0.2,...,0.9
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from const import MRI_CANONICAL_SHAPE, MRI_STAGE1_SHAPE, MRI_STAGE2_CROP_SHAPE
from data_reader import CARE2026_MRI
from models import CARE2026_MRI_Stage1_Model, CARE2026_MRI_Stage2_Model
from predict import _resample_3d, _resample_mask, _stage1_tta, _stage2_tta, _zscore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", required=True)
    parser.add_argument("--s1", default="checkpoints/mri_stage1_model.safetensors")
    parser.add_argument("--s2", default="checkpoints/mri_stage2_model.safetensors")
    parser.add_argument("--thresholds", default="0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    device = torch.device(args.device)

    print("Loading models...")
    s1, _ = CARE2026_MRI_Stage1_Model.from_checkpoint(args.s1, device=device)
    s1.eval()
    s2, _ = CARE2026_MRI_Stage2_Model.from_checkpoint(args.s2, device=device)
    s2.eval()
    s2_hw = int(s2.config.get("train_crop_hw", 128))
    use_sdf = "2ch" in str(s2.config.get("backbone", ""))

    reader = CARE2026_MRI(db_dir=args.db_dir, task=1, verbose=0)
    canonical_shape = tuple(s2.config.get("canonical_shape", MRI_CANONICAL_SHAPE))
    stage1_shape = tuple(s1.config.get("patch_shape", MRI_STAGE1_SHAPE))
    stage2_crop_shape = tuple(s2.config.get("patch_shape", MRI_STAGE2_CROP_SHAPE))

    # Accumulate per-threshold metrics across all records
    all_metrics = {t: defaultdict(list) for t in thresholds}

    print(f"Processing {len(reader.all_records)} labeled records...")
    for rec in reader.all_records:
        img_path = reader.get_data_path(rec)
        gt_scar = reader.load_scar_ann(rec)

        nii = nib.load(str(img_path))
        img_native = nii.get_fdata().astype(np.float32)
        orig_shape = img_native.shape

        # ── Stage 1 ─────────────────────────────────────────────
        img_canonical = _resample_3d(img_native, canonical_shape)
        img_s1 = _resample_3d(img_canonical, stage1_shape)
        img_s1_norm = _zscore(img_s1)
        t_s1 = torch.from_numpy(img_s1_norm).unsqueeze(0).unsqueeze(0)
        s1.eval()
        with torch.no_grad():
            la_prob_s1 = _stage1_tta(s1, t_s1, device)
        la_mask_s1 = (la_prob_s1[1] >= 0.5).astype(np.uint8)
        la_s1_canonical = _resample_mask(la_mask_s1, canonical_shape)

        fg = np.argwhere(la_s1_canonical > 0)
        if len(fg) == 0:
            cx, cy, cz = (s // 2 for s in canonical_shape)
        else:
            cx, cy, cz = tuple(int(fg[:, i].mean()) for i in range(3))

        # ── Centroid crop ─────────────────────────────────────
        cH, cW, cD = canonical_shape
        tH, tW, tD = stage2_crop_shape

        def _crop_coords(center, size, dim_len):
            half = size // 2
            v_start, v_end = center - half, center - half + size
            pb, pa = max(0, -v_start), max(0, v_end - dim_len)
            return max(0, v_start), min(dim_len, v_end), pb, pa

        xs, xe, px0, px1 = _crop_coords(int(cx), tH, cH)
        ys, ye, py0, py1 = _crop_coords(int(cy), tW, cW)
        zs, ze, pz0, pz1 = _crop_coords(int(cz), tD, cD)

        crop = img_canonical[xs:xe, ys:ye, zs:ze]
        if any(p > 0 for p in [px0, px1, py0, py1, pz0, pz1]):
            crop = np.pad(crop, ((px0, px1), (py0, py1), (pz0, pz1)), mode="constant")

        crop_h, crop_w, crop_d = crop.shape
        img_s2_norm = _zscore(crop)

        # ── SDF channel (if 2ch) ────────────────────────────────
        if use_sdf:
            from scipy.ndimage import distance_transform_edt

            la_crop = la_s1_canonical[xs:xe, ys:ye, zs:ze]
            if any(p > 0 for p in [px0, px1, py0, py1, pz0, pz1]):
                la_crop = np.pad(la_crop, ((px0, px1), (py0, py1), (pz0, pz1)), mode="constant")
            sdf_out = distance_transform_edt(1 - la_crop)
            sdf_in = distance_transform_edt(la_crop)
            sdf = np.clip((sdf_out - sdf_in).astype(np.float32) * 0.625 / 4.0, -1.0, 1.0)
            sdf_resized = _resample_3d(sdf, (s2_hw, s2_hw, crop_d))
            img_s2_resized = _resample_3d(img_s2_norm, (s2_hw, s2_hw, crop_d))
            img_2ch = np.stack([img_s2_resized, sdf_resized], axis=0)
            t_s2 = torch.from_numpy(img_2ch).unsqueeze(0)
        else:
            img_s2_resized = _resample_3d(img_s2_norm, (s2_hw, s2_hw, crop_d))
            t_s2 = torch.from_numpy(img_s2_resized).unsqueeze(0).unsqueeze(0)

        # ── Stage 2 (TTA, capture probability map) ─────────────
        s2.eval()
        with torch.no_grad():
            _, scar_prob_s2 = _stage2_tta(s2, t_s2, device)
        # scar_prob_s2[1] is the foreground (scar) probability: (H, W, D)
        scar_prob = scar_prob_s2[1].astype(np.float32)
        scar_prob_crop = _resample_3d(scar_prob, (crop_h, crop_w, crop_d))
        scar_prob_unpad = scar_prob_crop[px0 : tH - px1, py0 : tW - py1, pz0 : tD - pz1]
        scar_prob_canonical = np.zeros(canonical_shape, dtype=np.float32)
        scar_prob_canonical[xs:xe, ys:ye, zs:ze] = scar_prob_unpad
        scar_prob_native = _resample_3d(scar_prob_canonical, orig_shape)

        # ── Evaluate at each threshold ─────────────────────────
        for thr in thresholds:
            scar_bin = (scar_prob_native >= thr).astype(np.uint8)

            # Compute G-DSC
            # w_c = 1 / |gt_c|^2  for each class; G-DSC = 2 * sum(w_c * |p_c ∩ g_c|) / sum(w_c * (|p_c| + |g_c|))
            # Two classes: BG (0) and scar (1)
            p_bg = scar_bin == 0
            g_bg = gt_scar == 0
            p_sc = scar_bin == 1
            g_sc = gt_scar == 1

            w_bg = 1.0 / (max(g_bg.sum(), 1) ** 2)
            w_sc = 1.0 / (max(g_sc.sum(), 1) ** 2)

            inter_bg = (p_bg & g_bg).sum()
            inter_sc = (p_sc & g_sc).sum()
            union_bg = p_bg.sum() + g_bg.sum()
            union_sc = p_sc.sum() + g_sc.sum()

            numerator = 2 * (w_bg * inter_bg + w_sc * inter_sc)
            denominator = w_bg * union_bg + w_sc * union_sc
            gdsc = float(numerator / denominator) if denominator > 0 else 1.0

            # ACC
            acc = float((scar_bin == gt_scar).mean())
            # SEN (recall of scar)
            sen = float(inter_sc / max(g_sc.sum(), 1))

            all_metrics[thr]["gdsc"].append(gdsc)
            all_metrics[thr]["acc"].append(acc)
            all_metrics[thr]["sen"].append(sen)

    # ── Print summary ───────────────────────────────────────────
    print(f"\n{'Thr':>6} {'G-DSC':>8} {'ACC':>8} {'SEN':>8}")
    print("-" * 32)
    best_thr, best_gdsc = 0.5, 0.0
    for thr in thresholds:
        gdsc = np.mean(all_metrics[thr]["gdsc"])
        acc = np.mean(all_metrics[thr]["acc"])
        sen = np.mean(all_metrics[thr]["sen"])
        marker = " ←" if gdsc > best_gdsc else ""
        if gdsc > best_gdsc:
            best_gdsc = gdsc
            best_thr = thr
        print(f"{thr:6.2f} {gdsc:8.4f} {acc:8.4f} {sen:8.4f}{marker}")

    print(f"\nBest threshold: {best_thr:.2f} (G-DSC={best_gdsc:.4f})")


if __name__ == "__main__":
    main()
