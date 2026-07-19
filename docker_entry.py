#!/usr/bin/env python3
"""Docker entry point for CARE 2026 Left Atrium challenge.

Reads images from ``/input``, writes predictions to ``/output``.
Auto-detects which models are available and runs the corresponding tasks.
No CLI arguments required.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import torch

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("nnUNet_extTrainer", os.environ.get("nnUNet_extTrainer", "/challenge/models"))
warnings.filterwarnings("ignore", category=DeprecationWarning, module="batchgenerators")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="scipy")
warnings.filterwarnings("ignore", category=UserWarning, module="google")

_CHALLENGE_DIR = Path(__file__).resolve().parent
_CKPT_DIR = _CHALLENGE_DIR / "checkpoints"
_PATHS_JSON = _CKPT_DIR / "model_paths.json"
_TEST_FLAG = os.environ.get("CARE2026_ACTION_TEST", "") == "1"


def _load_model_paths() -> dict[str, str]:
    """Read model paths from post_docker_build.py output, or use defaults."""
    if _PATHS_JSON.is_file():
        return json.loads(_PATHS_JSON.read_text())

    # Fallback: use PredictCfg defaults
    from cfg import PredictCfg

    paths = {}
    for role in ("ct_model", "mri_stage1_model", "mri_stage2_model"):
        key = f"mri_{role}" if role.startswith("mri") else role
        val = getattr(PredictCfg, role, None) or getattr(PredictCfg, key, None)
        if val:
            paths[role] = str(val)
    return paths


def _detect_available_data(input_dir: str = "/input") -> set[int]:
    """Return tasks (1, 2, 3) that have test data under ``taskN/``."""
    input_path = Path(input_dir)
    available: set[int] = set()
    for task_num in (1, 2, 3):
        data_dir = input_path / f"task{task_num}"
        if not data_dir.exists():
            continue
        recs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("test_")]
        if recs:
            available.add(task_num)
    return available


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_ci = os.environ.get("CARE2026_TEST", "") == "1"

    model_paths = _load_model_paths()
    if not model_paths:
        print("No models found.  Check checkpoints/model_paths.json or PredictCfg.")
        sys.exit(1)

    from pipeline import _load_model, run_all_tasks

    mri_s1 = mri_s2 = ct_model = None
    mri_ready = False

    # --- Load CT model ---
    if "ct_model" in model_paths:
        try:
            ct_model = _load_model("ct", model_paths["ct_model"], device)
        except Exception as e:
            print(f"[WARN] CT model load failed: {e}")

    # --- Load MRI models ---
    s1_path = model_paths.get("mri_stage1_model", "")
    s2_path = model_paths.get("mri_stage2_model", "")
    if s1_path or s2_path:
        try:
            if s1_path:
                mri_s1 = _load_model("mri_stage1", s1_path, device)
            if s2_path:
                mri_s2 = _load_model("mri_stage2", s2_path, device)
            mri_ready = mri_s1 is not None and mri_s2 is not None
        except Exception as e:
            print(f"[WARN] MRI model load failed: {e}")

    if ct_model is None and not mri_ready:
        print("No usable models loaded.  Aborting.")
        sys.exit(1)

    # --- Detect which tasks have input data ---
    available_data = _detect_available_data()
    print(f"Input data available for tasks: {sorted(available_data) if available_data else 'none'}")

    # --- Determine which tasks to run (model ∩ data) ---
    tasks: list[int] = []
    if mri_ready:
        mri_tasks = [t for t in [1, 2] if t in available_data]
        tasks.extend(mri_tasks)
        if not mri_tasks:
            print("[WARN] MRI models loaded but no Task 1/2 input data; skipping.")
    if ct_model is not None:
        if 3 in available_data:
            tasks.append(3)
        else:
            print("[WARN] CT model loaded but no Task 3 input data; skipping.")

    if is_ci and not tasks:
        tasks = [1, 2, 3]  # CI mode: run all, let pipeline skip gracefully

    # --- CI speed optimisations ---
    use_tta = not _TEST_FLAG  # TTA off in CI (8x faster)

    print(f"Models loaded.  Running tasks: {tasks}")
    print(f"  Device:           {device}")
    print("  MRI pipeline:     hybrid")
    print(f"  TTA:              {'off' if _TEST_FLAG else 'on'}{' (CI fast mode)' if _TEST_FLAG else ''}")
    print()

    run_all_tasks(
        mri_stage1_model=mri_s1,
        mri_stage2_model=mri_s2,
        ct_model=ct_model,
        val_data_root="/input",
        results_dir="/output",
        device=device,
        use_tta=use_tta,
        overwrite=False,
        scar_dilation=None,
        tasks=tasks,
        mri_pipeline="hybrid",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
