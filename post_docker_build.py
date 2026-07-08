"""
Post-docker-build: validate that all required model weights are present
in the Docker image at build time.

Called during ``docker build`` after the repository is copied into the
image.  Does NOT download anything — models must be placed under
``checkpoints/`` locally before ``docker build``.

Expected layout::

    checkpoints/
      mri_stage1_model.safetensors  (or mri_stage1_model/ directory for nnUNet)
      mri_stage2_model.safetensors  (or mri_stage2_model/ directory for nnUNet)
      ct_model.safetensors          (or ct_model/ directory for nnUNet)
"""

import sys
from pathlib import Path

CHALLENGE_DIR = Path(__file__).resolve().parent
CKPT_DIR = CHALLENGE_DIR / "checkpoints"


def _exists(path: Path) -> bool:
    return path.exists()


def _check(name: str) -> bool:
    """Check that either a .safetensors file or a nnUNet directory exists."""
    vnet_file = CKPT_DIR / f"{name}.safetensors"
    nnunet_dir = CKPT_DIR / name
    if _exists(vnet_file):
        print(f"  [OK] {vnet_file}")
        return True
    if _exists(nnunet_dir):
        plans = nnunet_dir / "plans.json"
        folds = sorted(nnunet_dir.glob("fold_*"))
        if _exists(plans) and folds:
            print(f"  [OK] {nnunet_dir}  ({len(folds)} folds)")
            return True
        print(f"  [FAIL] {nnunet_dir} — missing plans.json or fold_*")
        return False
    print(f"  [FAIL] neither {vnet_file} nor {nnunet_dir} found")
    return False


def main():
    print("Checking model checkpoints ...")
    ok = True
    for name in ("mri_stage1_model", "mri_stage2_model", "ct_model"):
        if not _check(name):
            ok = False
    if not ok:
        print("\nMissing checkpoints.  Place models under checkpoints/ and rebuild.")
        sys.exit(1)
    print("\nAll checkpoints present.")


if __name__ == "__main__":
    main()
