"""
Post-docker-build: discover model weights and create symlinks.

Scans ``checkpoints/`` for nnUNet result directories (either placed there
directly or extracted from ``package_models.py`` zips).  Creates symbolic
links so ``pipeline.py`` and ``PredictCfg`` can find them.

Supported layouts
-----------------

**Layout A — exact names** (production Docker, placed manually)::

    checkpoints/
      ct_model/           (nnUNet dir: plans.json + fold_*/)
      mri_stage1_model/   (nnUNet dir)
      mri_stage2_model/   (nnUNet dir)

**Layout B — Dataset* names** (from ``package_models.py`` zip extraction)::

    checkpoints/
      Dataset500_CARE2026CT/
        nnUNetTrainer__nnUNetPlans__3d_fullres/
          plans.json, fold_0/, ...
      Dataset502_CARE2026MRI_Cavity/
        nnUNetTrainer__nnUNetPlans__3d_fullres/
          plans.json, fold_0/, ...
      Dataset521_CARE2026MRI_Scar/
        nnUNetTrainerScarGaussian__nnUNetPlans__3d_fullres/
          plans.json, fold_0/, ...

Auto-discovery maps dataset IDs to roles:

- Dataset 500 → ``ct_model``
- Dataset 502 → ``mri_stage1_model``
- Dataset 521 (ScarGaussian) → ``mri_stage2_model``
- Dataset 521 (default) → ``mri_stage2_model`` (fallback)

CI mode (``CARE2026_TEST=1``): allows missing models — only validates
that at least one model is present.  Production mode requires all three.
"""

import json
import os
import sys
from pathlib import Path

CHALLENGE_DIR = Path(__file__).resolve().parent
CKPT_DIR = CHALLENGE_DIR / "checkpoints"
PATHS_JSON = CKPT_DIR / "model_paths.json"

# Dataset ID → role assignment, ordered by preference
# (dataset_id, trainer_substring) → role_name
ROLE_RULES = [
    ("500", None, "ct_model"),
    ("502", None, "mri_stage1_model"),
    ("521", "ScarGaussian", "mri_stage2_model"),
    ("521", None, "mri_stage2_model"),
    ("501", None, "mri_stage2_model"),
]

EXPECTED_ROLES = ["ct_model", "mri_stage1_model", "mri_stage2_model"]


def _find_nnunet_dirs(root: Path) -> dict[str, tuple[str, str]]:
    """Find nnUNet directories under *root*.

    Returns ``{absolute_path: (dataset_name, trainer_name)}``.
    """
    found: dict[str, tuple[str, str]] = {}

    # Pattern: Dataset{N}_{name}/nnUNetTrainer{...}/
    for ds_dir in sorted(root.glob("Dataset*_*")):
        if not ds_dir.is_dir():
            continue
        for trainer_dir in sorted(ds_dir.iterdir()):
            if not trainer_dir.is_dir():
                continue
            key = str(trainer_dir.resolve())
            if key in found:
                continue
            if (trainer_dir / "plans.json").exists() and list(trainer_dir.glob("fold_*")):
                found[key] = (ds_dir.name, trainer_dir.name)

    # Pattern: loose nnUNet dir directly under checkpoints/
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        key = str(d.resolve())
        if key in found:
            continue
        if (d / "plans.json").exists() and list(d.glob("fold_*")):
            found[key] = (d.name, d.name)

    return found


def _assign_roles(discovered: dict[str, tuple[str, str]]) -> dict[str, Path]:
    """Map discovered directories to role names."""
    roles: dict[str, Path] = {}
    used: set[str] = set()

    for ds_id_pat, trainer_pat, role in ROLE_RULES:
        if role in roles:
            continue
        for path, (ds_name, trainer_name) in discovered.items():
            if path in used:
                continue
            ds_id = ds_name.split("_")[0].replace("Dataset", "")
            if ds_id != ds_id_pat:
                continue
            if trainer_pat is not None and trainer_pat not in trainer_name:
                continue
            roles[role] = Path(path)
            used.add(path)
            break

    return roles


def main() -> None:
    is_ci = os.environ.get("CARE2026_TEST", "") == "1"
    mode = "CI" if is_ci else "production"

    print(f"Discovering models in {CKPT_DIR}  [{mode} mode] ...\n")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    roles: dict[str, Path] = {}

    # --- Pass 1: check for exact-name directories ---
    for role in EXPECTED_ROLES:
        d = CKPT_DIR / role
        if d.is_dir() and (d / "plans.json").exists() and list(d.glob("fold_*")):
            roles[role] = d
            n_folds = len(list(d.glob("fold_*")))
            print(f"  [OK] {role}  <-  {d.name}  ({n_folds} folds)")

    # --- Pass 2: auto-discover from Dataset* dirs ---
    if len(roles) < len(EXPECTED_ROLES):
        discovered = _find_nnunet_dirs(CKPT_DIR)
        auto = _assign_roles(discovered)
        for role, path in auto.items():
            if role not in roles:
                n_folds = len(list(path.glob("fold_*")))
                roles[role] = path
                print(f"  [OK] {role}  <-  {path.relative_to(CKPT_DIR)}  ({n_folds} folds, auto)")

    # --- Create symlinks for any role where the name doesn't match ---
    for role, path in roles.items():
        link = CKPT_DIR / role
        if not link.exists():
            target = path.relative_to(CKPT_DIR, walk=True) if path.is_relative_to(CKPT_DIR) else path
            link.symlink_to(target)
            print(f"  symlink: {role} -> {target}")

    # --- Write path config ---
    path_config = {role: str(p) for role, p in roles.items()}
    PATHS_JSON.write_text(json.dumps(path_config, indent=2) + "\n")
    print(f"\n  Paths saved to {PATHS_JSON.relative_to(CHALLENGE_DIR)}")

    # --- Validate ---
    required = EXPECTED_ROLES if not is_ci else ["ct_model"]  # CI may test one task at a time
    missing = [r for r in required if r not in roles]
    if missing:
        print(f"\n  [FAIL] Missing: {missing}")
        if is_ci:
            print("  CI mode — continuing with partial models.")
        else:
            print("  Place models under checkpoints/ and rebuild.")
            sys.exit(1)

    print("\nAll required checkpoints present.")


if __name__ == "__main__":
    main()
