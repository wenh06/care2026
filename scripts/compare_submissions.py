"""Compare Task 1 / Task 2 predictions across multiple submission zips.

Usage::

    python scripts/compare_submissions.py \\
        results/CARE-Leftatrium-REVENGER-0709.zip \\
        results/CARE-Leftatrium-REVENGER-0710.zip \\
        results/CARE-Leftatrium-REVENGER-autodl.zip \\
        --labels "sub7" "sub8" "fixed" --task 1
"""

import argparse
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import nibabel as nib
import numpy as np

TASK_DIR = {
    1: "LA scar quantification",
    2: "LA cavity segmentation",
}


def _load_preds(zip_path: str, task: int) -> Dict[str, np.ndarray]:
    task_dir = TASK_DIR[task]
    preds = {}
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        rec_dir = os.path.join(tmp, task_dir)
        if not os.path.isdir(rec_dir):
            return {}
        for rec in sorted(os.listdir(rec_dir)):
            fpath = os.path.join(rec_dir, rec, f"{rec}_pred.nii.gz")
            if os.path.exists(fpath):
                preds[rec] = nib.load(fpath).get_fdata().astype(np.uint8)
    return preds


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    denom = a.sum() + b.sum()
    return float(2 * inter / denom) if denom > 0 else 0.0


def _centroid(mask: np.ndarray) -> Tuple[float, float, float]:
    fg = np.argwhere(mask > 0)
    if len(fg) == 0:
        return (0.0, 0.0, 0.0)
    return tuple(float(v) for v in fg.mean(axis=0))  # type: ignore[return-value]


def main():
    parser = argparse.ArgumentParser(description="Compare Task 1/2 predictions across submission zips")
    parser.add_argument("zips", nargs="+", type=str, help="Submission zip files (2 or more)")
    parser.add_argument("--labels", nargs="*", default=None, help="Short labels for each zip (default: stem of filename)")
    parser.add_argument("--task", type=int, choices=[1, 2], default=1, help="Task number (1=scar, 2=cavity)")
    parser.add_argument("--output", type=str, default=None, help="Save comparison table to file")
    args = parser.parse_args()

    if len(args.zips) < 2:
        print("ERROR: need at least 2 zip files to compare")
        sys.exit(1)

    # Labels
    if args.labels:
        if len(args.labels) != len(args.zips):
            print(f"ERROR: --labels count ({len(args.labels)}) != zip count ({len(args.zips)})")
            sys.exit(1)
        labels = args.labels
    else:
        labels = [Path(z).stem.replace("CARE-Leftatrium-", "").replace("REVENGER-", "")[:12] for z in args.zips]

    task_name = TASK_DIR[args.task]

    # Load
    all_preds = []
    all_labels = []
    for z, lbl in zip(args.zips, labels):
        preds = _load_preds(z, args.task)
        if not preds:
            print(f"WARNING: {lbl} has no {task_name} predictions, skipping")
            continue
        all_preds.append(preds)
        all_labels.append(lbl)

    if len(all_preds) < 2:
        print("ERROR: need at least 2 valid submissions with the requested task")
        sys.exit(1)

    common_cases = sorted(set.intersection(*[set(p.keys()) for p in all_preds]))
    if not common_cases:
        print("ERROR: no common cases found across submissions")
        sys.exit(1)

    n = len(all_preds)
    is_binary = args.task in (1, 2)

    lines = []

    # ── Per-case table ───────────────────────────────────────────────────
    header = f"{'Case':<8} {'Shape':<20}"
    for lbl in all_labels:
        header += f" {lbl:>10}"
    for i in range(n):
        header += f" {'CZ' + str(i):>8}" if is_binary else ""
    lines.append(header)
    lines.append("-" * len(header))

    all_counts = [[] for _ in range(n)]
    # Accumulate per-case pairwise Dice
    pair_dice_sums = {}
    pair_dice_counts = {}

    for rec in common_cases:
        masks = [p[rec] for p in all_preds]
        shape = masks[0].shape
        counts = [int(m.sum()) for m in masks]

        row = f"{rec:<8} {str(shape):<20}"
        for c in counts:
            row += f" {c:>10}"
        if is_binary:
            for m in masks:
                _, _, cz = _centroid(m)
                row += f" {cz:>8.1f}"
        lines.append(row)

        for i in range(n):
            all_counts[i].append(counts[i])
        for i in range(n):
            for j in range(i + 1, n):
                d = _dice(masks[i], masks[j])
                pair_dice_sums[(i, j)] = pair_dice_sums.get((i, j), 0.0) + d
                pair_dice_counts[(i, j)] = pair_dice_counts.get((i, j), 0) + 1

    # ── Mean row ─────────────────────────────────────────────────────────
    lines.append("-" * len(header))
    mean_row = f"{'Mean':<8} {'':<20}"
    for i in range(n):
        mean_row += f" {int(np.mean(all_counts[i])):>10}"
    lines.append(mean_row)

    # ── Pairwise Dice matrix ─────────────────────────────────────────────
    lines.append("")
    lines.append("Pairwise mean Dice:")
    header2 = f"{'':>12}"
    for lbl in all_labels:
        header2 += f" {lbl:>10}"
    lines.append(header2)
    for i in range(n):
        row2 = f"{all_labels[i]:>12}"
        for j in range(n):
            if i == j:
                row2 += f" {'—':>10}"
            else:
                key = (min(i, j), max(i, j))
                mean_d = pair_dice_sums.get(key, 0.0) / max(pair_dice_counts.get(key, 1), 1)
                row2 += f" {mean_d:>10.4f}"
        lines.append(row2)

    # ── FG change vs reference (first zip) ───────────────────────────────
    lines.append("")
    lines.append(f"FG change vs {all_labels[0]}:")
    ref_sum = sum(all_counts[0])
    for i in range(1, n):
        pct = (sum(all_counts[i]) / ref_sum - 1) * 100 if ref_sum > 0 else 0
        lines.append(f"  {all_labels[i]}: {pct:+.1f}%")

    # ── Output ───────────────────────────────────────────────────────────
    out = "\n".join(lines)
    print(out)

    if args.output:
        Path(args.output).write_text(out + "\n")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
