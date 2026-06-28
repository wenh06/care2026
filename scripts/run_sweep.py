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
    parser.add_argument("--repeats", type=int, default=1, help="Repeat each combination N times (different seed)")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed")
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

    total_runs = len(combinations) * args.repeats
    print(f"Sweep: {args.sweep}")
    print(f"Combinations: {len(combinations)} × {args.repeats} repeats = {total_runs} runs")
    print(f"Log dir: {exp_base}")
    print(f"Checkpoint dir: {ckpt_base}")
    print()

    for repeat in range(args.repeats):
        seed = args.base_seed + repeat
        for idx, combo_values in enumerate(combinations):
            combo = dict(zip(keys, combo_values))
            combo.update(fixed)
            combo["random_seed"] = seed
            # Apply conditional overrides
            for (cond_key, cond_val), overrides in sweep.get("conditional", {}).items():
                if cond_key in combo and combo[cond_key] == cond_val:
                    combo.update(overrides)
            run_label = f"{_exp_name(args.sweep, combo, idx)}_r{repeat}"
            run_num = repeat * len(combinations) + idx + 1

            exp_log_dir = exp_base / run_label
            exp_ckpt_dir = ckpt_base / run_label
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

            KEY_TO_CLI = {
                "n_epochs": "epochs",
                "batch_size": "batch-size",
                "val_ratio": "val-ratio",
                "lr_scheduler": "lr-scheduler",
                "random_seed": "random-seed",
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
            ckpt_file = exp_ckpt_dir / f"{run_label}.safetensors"

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = args.device
            if "PYTORCH_ALLOC_CONF" not in env:
                env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

            cmd_str = " \\\n  ".join(cli_args)
            print(f"[{run_num}/{total_runs}] {run_label} (seed={seed})")
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

            best_ckpts = sorted(
                base_dir.glob(f"checkpoints/BestModel_*{run_label[:20]}*"),
                key=os.path.getmtime,
                reverse=True,
            )
            if best_ckpts:
                import shutil

                shutil.copy2(best_ckpts[0], ckpt_file)
                print(f"  Best ckpt: {best_ckpts[0].name} → {ckpt_file}")
            else:
                print("  WARNING: no best checkpoint found")

            # --- Append summary row ---
            _append_summary(exp_base, run_num, run_label, combo, seed, log_file, ckpt_file, base_dir)

            print()


def _append_summary(exp_base, run_num, run_label, combo, seed, log_file, ckpt_file, base_dir):
    import csv

    summary_path = exp_base / "summary.csv"
    write_header = not summary_path.exists()

    # Try to extract best val metric from training log txt
    best_val = None
    log_path = base_dir / log_file if not str(log_file).startswith("/") else Path(str(log_file))
    if log_path.exists():
        try:
            with open(log_path) as f:
                for line in f:
                    m = __import__("re").search(r"best metric = ([\d.]+)", line)
                    if m:
                        best_val = m.group(1)
                        break
        except Exception:
            pass

    with open(summary_path, "a", newline="") as f:
        fieldnames = ["run", "label", "seed"] + sorted(combo.keys()) + ["best_val", "log_file", "ckpt_file"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {
            "run": run_num,
            "label": run_label,
            "seed": seed,
            **combo,
            "best_val": best_val or "",
            "log_file": str(log_file.relative_to(base_dir)),
            "ckpt_file": str(ckpt_file.relative_to(base_dir)),
        }
        writer.writerow(row)


if __name__ == "__main__":
    main()
