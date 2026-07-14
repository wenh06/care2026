"""Validate a CARE 2026 submission zip file.

Checks structure, naming, data types, and prediction statistics against
expected training-set distributions.

Usage::

    python scripts/check_submission.py results/CARE-Leftatrium-REVENGER.zip
"""

import argparse
import os
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

EXPECTED = {
    "LA scar quantification": {
        "cases": 10,
        "dtype": {0, 1},
        "desc": "binary scar mask",
    },
    "LA cavity segmentation": {
        "cases": 20,
        "dtype": {0, 1},
        "desc": "binary cavity mask",
    },
    "LA multi-structure segmentation": {
        "cases": 20,
        "dtype": {0, 1, 2, 3},
        "desc": "multi-class (bg=0, LA=1, PV=2, LAA=3)",
    },
}

CLASS_NAMES = {1: "LA", 2: "PV", 3: "LAA"}

TASK_SUBDIR = {
    "LA scar quantification": "task1",
    "LA cavity segmentation": "task2",
    "LA multi-structure segmentation": "task3",
}


def _find_input_nii(data_dir: str, task_name: str, rec: str) -> str | None:
    """Locate the original input NIfTI for a given validation record."""
    subdir = TASK_SUBDIR.get(task_name)
    if subdir is None:
        return None
    rec_dir = Path(data_dir) / subdir / "val_data" / rec
    if not rec_dir.is_dir():
        return None
    # Task 1/2: enhanced.nii.gz; Task 3: NNNN.nii.gz (zero-padded 4-digit)
    candidates = list(rec_dir.glob("*.nii.gz"))
    for p in candidates:
        if p.name == "enhanced.nii.gz" or p.stem.endswith(".nii"):
            return str(p)
    return str(candidates[0]) if candidates else None


def main():
    parser = argparse.ArgumentParser(description="Validate a CARE 2026 submission zip")
    parser.add_argument("zipfile", type=str, help="Path to the submission .zip file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only print errors")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to original validation data for shape/spacing cross-check",
    )
    args = parser.parse_args()

    zf_path = Path(args.zipfile)
    if not zf_path.exists():
        print(f"ERROR: file not found: {zf_path}")
        sys.exit(1)

    print(f"Checking: {zf_path.name} ({zf_path.stat().st_size / 1024 / 1024:.1f} MB)")
    errors = []

    with zipfile.ZipFile(zf_path, "r") as zf:
        names = sorted(zf.namelist())

        # Group by task directory
        tasks = defaultdict(list)
        for n in names:
            parts = n.rstrip("/").split("/")
            if len(parts) >= 3:
                tasks[parts[0]].append((parts[1], parts[2]))

        # Check expected tasks
        for task_name, spec in EXPECTED.items():
            if task_name not in tasks:
                errors.append(f"MISSING: {task_name}/ directory not found")
                continue
            entries = tasks[task_name]
            recs = sorted(set(r for r, _ in entries))
            n_found = len(recs)
            n_expected = spec["cases"]
            if n_found != n_expected:
                errors.append(f"{task_name}: expected {n_expected} cases, found {n_found}")

        # Check per-file validity
        with tempfile.TemporaryDirectory() as tmp:
            zf.extractall(tmp)
            for task_name, spec in EXPECTED.items():
                task_dir = os.path.join(tmp, task_name)
                if not os.path.isdir(task_dir):
                    continue
                for rec in sorted(os.listdir(task_dir)):
                    pred_name = f"{rec}_pred.nii.gz"
                    pred_path = os.path.join(task_dir, rec, pred_name)
                    if not os.path.exists(pred_path):
                        errors.append(f"{task_name}/{rec}: missing {pred_name}")
                        continue

                    pred_nii = nib.load(pred_path)
                    arr = pred_nii.get_fdata().astype(np.uint8)
                    uniq = set(arr.ravel().astype(int).tolist())
                    unexpected = uniq - spec["dtype"]
                    if unexpected:
                        errors.append(f"{task_name}/{rec}: unexpected values {unexpected} (expected {spec['dtype']})")

                    # Cross-check shape & spacing against original data
                    if args.data_dir:
                        src = _find_input_nii(args.data_dir, task_name, rec)
                        if src is None:
                            errors.append(f"{task_name}/{rec}: cannot find input NIfTI under {args.data_dir}")
                        else:
                            src_nii = nib.load(src)
                            if pred_nii.shape != src_nii.shape:
                                errors.append(
                                    f"{task_name}/{rec}: shape mismatch — pred {pred_nii.shape}, input {src_nii.shape}"
                                )
                            pred_zooms = tuple(float(z) for z in pred_nii.header.get_zooms()[:3])
                            src_zooms = tuple(float(z) for z in src_nii.header.get_zooms()[:3])
                            if not np.allclose(pred_zooms, src_zooms, atol=1e-3):
                                errors.append(f"{task_name}/{rec}: spacing mismatch — pred {pred_zooms}, input {src_zooms}")

                    # Print stats per case
                    total = int(arr.size)
                    if spec["dtype"] == {0, 1}:
                        fg = int((arr > 0).sum())
                        pct = fg / total * 100
                        if not args.quiet:
                            print(f"  {task_name}/{rec}: FG={fg:>10} / {total:>12} = {pct:.3f}%")
                    else:
                        parts = []
                        for c in sorted(uniq - {0}):
                            cnt = int((arr == c).sum())
                            parts.append(f"{CLASS_NAMES.get(c, c)}={cnt}")
                        if not args.quiet:
                            print(f"  {task_name}/{rec}: {', '.join(parts)}")

    if errors:
        print(f"\n{len(errors)} ERROR(S):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\nAll checks passed. Ready for submission.")


if __name__ == "__main__":
    main()
