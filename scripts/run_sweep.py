"""
Experiment sweep runner — define parameter grids, run all combinations,
organise checkpoints and logs per experiment.

Usage:
    python scripts/run_sweep.py --task mri --stage 2 --db-dir /Data/...

The SWEEP dict below defines the parameter space.  Edit it to add/remove
combinations.  Each key maps to a list of values; the Cartesian product
of all keys is run.  Set a key to a single value (list of length 1) to
keep it fixed across runs.

Logs go to log/experiments/{run_tag}/{exp_name}/
Checkpoints go to checkpoints/experiments/{run_tag}/{exp_name}/
"""

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ============================================================================
# Parameter sweep definition — edit this to add/remove experiments
# ============================================================================
SWEEPS = {
    # --- Scar (Stage 2) ablation ---
    "scar_ablation": {
        "task": "mri",
        "stage": 2,
        "db_dir": None,
        "grid": {
            "backbone": ["vnet", "nested_vnet"],
            "optimizer": ["adamw", "sgd"],
            "n_epochs": [200, 400],
            "val_ratio": [0.1],
        },
        "fixed": {"mclahe": True, "batch_size": 4},
        # Conditional overrides: when optimizer=adamw → lr=3e-4, lr_scheduler=cosine
        # when optimizer=sgd → lr=1e-2, lr_scheduler=poly
        "conditional": {
            ("optimizer", "adamw"): {"lr": 3e-4, "lr_scheduler": "cosine"},
            ("optimizer", "sgd"): {"lr": 1e-2, "lr_scheduler": "poly"},
        },
    },
    # --- Lightweight quick test ---
    "scar_quick": {
        "task": "mri",
        "stage": 2,
        "db_dir": None,
        "grid": {
            "backbone": ["vnet", "nested_vnet"],
            "optimizer": ["sgd"],
            "n_epochs": [100],
            "val_ratio": [0.1],
        },
        "fixed": {"mclahe": True, "batch_size": 4},
    },
}

# ============================================================================


def _exp_name(sweep_name, combo, idx):
    """Generate a compact experiment name from parameter values."""
    parts = []
    for k, v in combo.items():
        # Abbreviate common keys
        abbr = {"backbone": "bb", "optimizer": "opt", "n_epochs": "ep", "val_ratio": "val"}
        short_k = abbr.get(k, k)
        short_v = str(v).replace("_", "")
        parts.append(f"{short_k}{short_v}")
    return f"{sweep_name}_{idx:02d}_{'-'.join(parts)}"


def main():
    parser = argparse.ArgumentParser(description="Run experiment sweeps")
    parser.add_argument("--sweep", required=True, choices=list(SWEEPS.keys()), help="Sweep name")
    parser.add_argument("--db-dir", required=True, help="CARE2026 dataset root")
    parser.add_argument("--run-tag", default=None, help="Custom run tag (default: timestamp)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--device", default="0", help="CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()

    sweep = SWEEPS[args.sweep]
    grid = sweep["grid"]
    fixed = sweep.get("fixed", {})
    task = sweep["task"]
    stage = sweep.get("stage", 2)
    run_tag = args.run_tag or time.strftime("%Y%m%d_%H%M%S")

    # Generate all combinations
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combinations = list(itertools.product(*values))

    base_dir = Path(__file__).resolve().parents[1]
    exp_base = base_dir / "log" / "experiments" / run_tag
    ckpt_base = base_dir / "checkpoints" / "experiments" / run_tag

    print(f"Sweep: {args.sweep}")
    print(f"Combinations: {len(combinations)}")
    print(f"Log dir: {exp_base}")
    print(f"Checkpoint dir: {ckpt_base}")
    print()

    for idx, combo_values in enumerate(combinations):
        combo = dict(zip(keys, combo_values))
        combo.update(fixed)
        # Apply conditional overrides
        for (cond_key, cond_val), overrides in sweep.get("conditional", {}).items():
            if cond_key in combo and combo[cond_key] == cond_val:
                combo.update(overrides)
        exp_name = _exp_name(args.sweep, combo, idx)

        exp_log_dir = exp_base / exp_name
        exp_ckpt_dir = ckpt_base / exp_name
        exp_log_dir.mkdir(parents=True, exist_ok=True)
        exp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Build CLI args
        cli_args = [
            "python",
            "trainer.py",
            "--task",
            task,
        ]
        if stage:
            cli_args += ["--stage", str(stage)]
        cli_args += ["--db-dir", args.db_dir]

        # Map grid keys to trainer CLI args
        KEY_TO_CLI = {
            "n_epochs": "epochs",
            "batch_size": "batch-size",
            "val_ratio": "val-ratio",
            "lr_scheduler": "lr-scheduler",
        }
        skip_keys = {"task", "stage", "db_dir"}
        for k, v in combo.items():
            if k in skip_keys:
                continue
            cli_flag = KEY_TO_CLI.get(k, k.replace("_", "-"))
            if isinstance(v, bool):
                cli_args += [f"--{cli_flag}", str(v).lower()]
            else:
                cli_args += [f"--{cli_flag}", str(v)]

        log_file = exp_log_dir / "train.log"
        ckpt_file = exp_ckpt_dir / f"{exp_name}.safetensors"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.device
        if "PYTORCH_ALLOC_CONF" not in env:
            env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

        # Print and (optionally) run
        cmd_str = " \\\n  ".join(cli_args)
        print(f"[{idx+1}/{len(combinations)}] {exp_name}")
        print(f"  Log: {log_file}")
        if args.dry_run:
            print(f"  CMD: {' '.join(cli_args)}")
            print()
            continue

        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                cli_args,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(base_dir),
            )
            proc.wait()

        # Find best checkpoint and copy to organized dir
        best_ckpts = sorted(
            base_dir.glob(f"checkpoints/BestModel_*{exp_name[:20]}*"),
            key=os.path.getmtime,
            reverse=True,
        )
        if best_ckpts:
            import shutil

            shutil.copy2(best_ckpts[0], ckpt_file)
            print(f"  Best ckpt: {best_ckpts[0].name} → {ckpt_file}")
        else:
            print("  WARNING: no best checkpoint found")

        print()


if __name__ == "__main__":
    main()
