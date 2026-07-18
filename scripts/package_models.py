#!/usr/bin/env python3
"""Package nnUNet model directories into zip files for CI / GitHub Actions.

Maintains directory hierarchy so zips can be extracted directly into nnUNet_results.

Usage::

    # Full model (all folds)
    python /root/workspace/care2026/scripts/package_models.py \
        --results-root /root/autodl-tmp/nnunet/nnUNet_results \
        --output-dir /root/autodl-tmp/nnunet/model_zips

    # Single fold (checkpoint_best.pth only)
    python /root/workspace/care2026/scripts/package_models.py \
        --results-root /root/autodl-tmp/nnunet/nnUNet_results \
        --output-dir /root/autodl-tmp/nnunet/model_zips \
        --fold 0

    # Verbose — print every file added to each zip
    python ... --verbose
"""

import argparse
import zipfile
from pathlib import Path

# Trainer name -> zip suffix (appended to dataset ID)
TRAINER_SUFFIX = {
    "nnUNetTrainer__nnUNetPlans__3d_fullres": "",
    "nnUNetTrainerCTBoundary__nnUNetPlans__3d_fullres": "b",
    "nnUNetTrainerScarGaussian__nnUNetPlans__3d_fullres": "w",
}

# Files at the dataset-directory level to include for every model
JSON_FILES = ["dataset.json", "plans.json", "dataset_fingerprint.json"]


def _suffix(trainer_name: str) -> str:
    """Return filename suffix for a trainer, e.g. '' or 'b' or 'w'."""
    if trainer_name in TRAINER_SUFFIX:
        return TRAINER_SUFFIX[trainer_name]
    body = trainer_name.replace("nnUNetTrainer", "", 1).split("__")[0]
    return f"_{body.lower()}" if body else ""


CHECKPOINTS = ["checkpoint_best.pth", "checkpoint_final.pth"]


def _pack_full_model(zf: zipfile.ZipFile, ds_name: str, trainer_dir: Path, verbose: bool) -> list[str]:
    """Add all folds, best + final checkpoints per fold."""
    added = []
    for fold_dir in sorted(trainer_dir.glob("fold_*")):
        if not fold_dir.is_dir() or fold_dir.name.startswith("."):
            continue
        for ckpt in CHECKPOINTS:
            ckpt_path = fold_dir / ckpt
            if ckpt_path.is_file():
                arcname = str(Path(ds_name) / trainer_dir.name / fold_dir.name / ckpt)
                zf.write(ckpt_path, arcname)
                added.append(arcname)
                if verbose:
                    print(f"    + {arcname}")
    return added


def _pack_single_fold(zf: zipfile.ZipFile, ds_name: str, trainer_dir: Path, fold: int, verbose: bool) -> list[str]:
    """Add fold_N/ best + final checkpoints."""
    added = []
    for ckpt in CHECKPOINTS:
        ckpt_path = trainer_dir / f"fold_{fold}" / ckpt
        if ckpt_path.is_file():
            arcname = str(Path(ds_name) / trainer_dir.name / f"fold_{fold}" / ckpt)
            zf.write(ckpt_path, arcname)
            added.append(arcname)
            if verbose:
                print(f"    + {arcname}")
    return added


def package_models(results_root: Path, output_dir: Path, fold: int | None, verbose: bool = False) -> int:
    """Walk results_root, create one zip per trainer subdirectory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for ds_dir in sorted(results_root.glob("Dataset*_*")):
        ds_name = ds_dir.name
        ds_id = ds_name.split("_")[0].replace("Dataset", "")

        for trainer_dir in sorted(ds_dir.iterdir()):
            if not trainer_dir.is_dir():
                continue

            trainer_name = trainer_dir.name

            if fold is not None:
                fold_dir = trainer_dir / f"fold_{fold}"
                if not fold_dir.is_dir():
                    print(f"  SKIP  {ds_name}/{trainer_name}  (no fold_{fold})")
                    continue
            else:
                if not (trainer_dir / "fold_0").is_dir():
                    print(f"  SKIP  {ds_name}/{trainer_name}  (no fold_0)")
                    continue

            suffix = _suffix(trainer_name)
            if fold is not None:
                zip_name = f"{ds_id}{suffix}_f{fold}_model.zip"
            else:
                zip_name = f"{ds_id}{suffix}_model.zip"
            zip_path = output_dir / zip_name

            label = f"fold_{fold}" if fold is not None else "full"
            print(f"\n  {ds_name}/{trainer_name} [{label}]  ->  {zip_name}")

            added: list[str] = []

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                # --- JSON files (at trainer-directory level) ---
                for jf in JSON_FILES:
                    jf_path = trainer_dir / jf
                    if jf_path.is_file():
                        arcname = str(Path(ds_name) / trainer_name / jf)
                        zf.write(jf_path, arcname)
                        added.append(arcname)
                        if verbose:
                            print(f"    + {arcname}")
                    elif verbose:
                        print(f"    - {ds_name}/{trainer_name}/{jf}  (missing)")

                # --- model weights ---
                if fold is not None:
                    added += _pack_single_fold(zf, ds_name, trainer_dir, fold, verbose)
                else:
                    added += _pack_full_model(zf, ds_name, trainer_dir, verbose)

            size_mb = zip_path.stat().st_size / (1024**2)
            print(f"    {len(added)} file(s), {size_mb:.1f} MB")
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Package nnUNet models for CI")
    parser.add_argument(
        "--results-root",
        required=True,
        help="Path to nnUNet_results directory",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write zip files (default: current directory)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Package a single fold (checkpoint_best.pth only).  Omit for full model (all folds).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print every file added to each zip.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write list of created zip files to a text file",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.is_dir():
        print(f"ERROR: {results_root} is not a directory")
        return 1

    mode = f"fold_{args.fold}" if args.fold is not None else "full model (all folds)"
    print(f"Scanning {results_root} ...  [{mode}]")
    n = package_models(results_root, Path(args.output_dir), args.fold, verbose=args.verbose)
    print(f"\nCreated {n} zip(s) in {Path(args.output_dir).resolve()}")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text("\n".join(sorted(str(p) for p in Path(args.output_dir).glob("*_model.zip"))))
        print(f"File list saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
