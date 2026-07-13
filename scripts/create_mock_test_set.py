"""Create a mock test set from validation data for local Docker testing.

Symlinks validation cases into the test-phase directory structure.
No files are copied — the mock set uses symlinks to save disk space.

Usage::

    python scripts/create_mock_test_set.py --val-dir /Data1/wenh06/CARE2026-LeftAtrium
    python scripts/create_mock_test_set.py --val-dir ... --output tmp/mock-test-set --n 3

Output structure::

    <output>/
    ├── task1/
    │   ├── test_01/enhanced.nii.gz
    │   └── ...
    ├── task2/
    │   ├── test_01/enhanced.nii.gz
    │   └── ...
    └── task3/
        ├── test_01/0001.nii.gz
        └── ...
"""

import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create a mock test set from validation data")
    parser.add_argument("--val-dir", required=True, help="Parent directory of task1/, task2/, task3/")
    parser.add_argument("--output", default="tmp/mock-test-set", help="Output directory (default: tmp/mock-test-set)")
    parser.add_argument("--n", type=int, default=5, help="Number of cases per task (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    val_dir = Path(args.val_dir)
    out_dir = Path(args.output)
    rng = random.Random(args.seed)

    # Remove existing mock set
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)

    tasks = {
        1: {"dir": val_dir / "task1" / "val_data", "prefix": "val_", "img_name": "enhanced.nii.gz"},
        2: {"dir": val_dir / "task2" / "val_data", "prefix": "val_", "img_name": "enhanced.nii.gz"},
        3: {"dir": val_dir / "task3" / "val_data", "prefix": "val_", "img_name": None},  # CT uses case ID naming
    }

    for task_id, cfg in tasks.items():
        data_dir = cfg["dir"]
        if not data_dir.exists():
            print(f"[Task {task_id}] Validation directory not found: {data_dir}  — skipping")
            continue

        # Discover available validation cases
        cases = sorted(
            [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith(cfg["prefix"])],
            key=lambda p: int(p.name.split("_")[1]),
        )
        n_available = len(cases)
        n_select = min(args.n, n_available)
        selected = rng.sample(cases, n_select)

        task_out = out_dir / f"task{task_id}"
        task_out.mkdir(parents=True, exist_ok=True)

        for idx, case_dir in enumerate(selected, start=1):
            test_name = f"test_{idx:02d}"
            case_out = task_out / test_name
            case_out.mkdir(parents=True, exist_ok=True)

            if cfg["img_name"] is not None:
                # Task 1/2: enhanced.nii.gz
                src = case_dir / cfg["img_name"]
                dst = case_out / cfg["img_name"]
            else:
                # Task 3: NNNN.nii.gz (case ID naming)
                rec_num = int(case_dir.name.split("_")[1])
                src = case_dir / f"{rec_num:04d}.nii.gz"
                dst = case_out / f"{idx:04d}.nii.gz"

            if not src.exists():
                print(f"  [Task {task_id}] WARNING: {src} not found — skipping")
                continue
            dst.symlink_to(src.resolve())
            print(f"  [Task {task_id}] {test_name} <- {case_dir.name}")

    print(f"\nCreated mock test set: {out_dir.resolve()}")
    print(f"  Cases per task: {args.n}")
    print("\nTest with Docker:")
    print(f"  docker run -v {out_dir.resolve()}:/input:ro -v $PWD/tmp/mock-output:/output ...")


if __name__ == "__main__":
    main()
