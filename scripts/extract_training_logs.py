#!/usr/bin/env python3
"""Extract training logs from nnUNet_results directory.

Copies training_log_*.txt, debug.json, and progress.png from every fold
of every trainer, preserving directory hierarchy.

Usage::

    python scripts/extract_training_logs.py \
        --results-root /root/autodl-tmp/nnunet/nnUNet_results \
        --output-dir /root/autodl-tmp/nnunet/training_logs

Output layout::

    training_logs/
    ├── Dataset521_CARE2026MRI_Scar/
    │   ├── nnUNetTrainerScarGaussian__nnUNetPlans__3d_fullres/
    │   │   ├── fold_0/
    │   │   │   ├── training_log_2026_7_7_19_10_40.txt
    │   │   │   ├── debug.json
    │   │   │   └── progress.png
    │   │   ├── fold_1/ ...
    │   ├── nnUNetTrainerScarCavityWall__nnUNetPlans__3d_fullres/ ...
    ├── Dataset502_CARE2026MRI_Cavity/ ...
"""

import argparse
import shutil
import sys
from pathlib import Path

LOG_GLOB_PATTERNS = ["training_log_*.txt", "debug.json", "progress.png"]


def extract_logs(results_root: Path, output_dir: Path, verbose: bool = False) -> tuple[int, int]:
    """Copy training logs from results_root to output_dir.

    Returns (num_files_copied, num_folds_processed).
    """
    file_count = 0
    fold_count = 0

    ds_dirs = sorted(results_root.glob("Dataset*_*"))
    print(f"Found {len(ds_dirs)} dataset director{'y' if len(ds_dirs) == 1 else 'ies'}")
    sys.stdout.flush()

    for ds_dir in ds_dirs:
        ds_name = ds_dir.name
        if verbose:
            print(f"  {ds_name}/")

        trainer_dirs = [d for d in sorted(ds_dir.iterdir()) if d.is_dir()]
        for trainer_dir in trainer_dirs:
            trainer_name = trainer_dir.name
            fold_dirs = sorted(trainer_dir.glob("fold_*"))

            for fold_dir in fold_dirs:
                if not fold_dir.is_dir():
                    continue
                fold_count += 1
                dest_fold = output_dir / ds_name / trainer_name / fold_dir.name
                any_copied = False

                for pattern in LOG_GLOB_PATTERNS:
                    for src in sorted(fold_dir.glob(pattern)):
                        dest = dest_fold / src.name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dest)
                        file_count += 1
                        any_copied = True
                        if verbose:
                            rel = dest.relative_to(output_dir)
                            print(f"    {rel}")

                if verbose and not any_copied:
                    rel = dest_fold.relative_to(output_dir)
                    print(f"    {rel}/  (no logs found)")

    return file_count, fold_count


def main():
    parser = argparse.ArgumentParser(description="Extract training logs from nnUNet_results")
    parser.add_argument("--results-root", required=True, help="Path to nnUNet_results directory")
    parser.add_argument("--output-dir", required=True, help="Directory to copy logs into")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print every file copied")
    parser.add_argument("--output", default=None, help="Write summary to file")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.is_dir():
        print(f"ERROR: {results_root} is not a directory")
        return 1

    output_dir = Path(args.output_dir)
    print(f"Scanning {results_root} ...")
    n_files, n_folds = extract_logs(results_root, output_dir, verbose=args.verbose)
    print(f"\nCopied {n_files} file(s) from {n_folds} fold(s) to {output_dir.resolve()}")

    if args.output:
        out_path = Path(args.output)
        lines = [
            f"Source: {results_root.resolve()}",
            f"Output: {output_dir.resolve()}",
            f"Files: {n_files}",
            f"Folds: {n_folds}",
        ]
        out_path.write_text("\n".join(lines))
        print(f"Summary saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
