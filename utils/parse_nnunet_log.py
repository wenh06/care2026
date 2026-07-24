"""Parse epoch-level metrics from an nnUNet training log file.

Supports both binary (single pseudo_dice) and multi-class (list of per-class
dice values).  Class names are read from ``dataset.json`` if available.

Usage::

    df = parse_nnunet_log("tmp/nnUNet_results/.../fold_0/training_log_*.txt")
    print(df.head())
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

__all__ = ["parse_nnunet_log"]

_EPOCH_RE = re.compile(r"Epoch (\d+)\s*$")
_LR_RE = re.compile(r"Current learning rate: ([\d.eE+-]+)")
_TRAIN_LOSS_RE = re.compile(r"train_loss ([\-\d.eE+]+)")
_VAL_LOSS_RE = re.compile(r"val_loss ([\-\d.eE+]+)")
_DICE_RE = re.compile(r"Pseudo dice \[(.*)\]")
_DICE_VAL_RE = re.compile(r"np\.float32\(([\d.eE+]+)\)")
_TIME_RE = re.compile(r"Epoch time: ([\d.]+)")


def _resolve_class_names(log_path: str) -> Dict[int, str]:
    """Try to read class names from the dataset.json next to the log file."""
    p = Path(log_path)
    trainer_dir = p.parent.parent  # fold_0/ → trainer_dir/
    for try_path in [
        trainer_dir / "dataset.json",
        trainer_dir.parent / "dataset.json",
    ]:
        if try_path.exists():
            data = json.loads(try_path.read_text())
            labels = data.get("labels", {})
            if labels:
                return {int(v): k for k, v in labels.items() if int(v) > 0}
    return {}


def parse_nnunet_log(log_path: str, class_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Parse nnUNet training log into a DataFrame.

    Parameters
    ----------
    log_path : str
        Path to ``training_log_*.txt``.
    class_names : list of str, optional
        Per-class names for multi-class dice columns (class 1, 2, ...).
        If None, auto-detected from ``dataset.json`` in the trainer directory.

    Returns
    -------
    pd.DataFrame
        Columns: epoch, lr, train_loss, val_loss, epoch_time_s, and
        ``mean_dice`` + per-class ``dice_<cls>`` columns.
    """
    log_path = str(log_path)
    rows: List[dict] = []
    current: Optional[dict] = None

    with open(log_path) as f:
        for line in f:
            line = line.rstrip()
            m = _EPOCH_RE.search(line)
            if m:
                if current is not None and "epoch" in current:
                    rows.append(current)
                current = {"epoch": int(m.group(1))}
                continue
            if current is None:
                continue
            m = _LR_RE.search(line)
            if m:
                current["lr"] = float(m.group(1))
                continue
            m = _TRAIN_LOSS_RE.search(line)
            if m:
                current["train_loss"] = float(m.group(1))
                continue
            m = _VAL_LOSS_RE.search(line)
            if m:
                current["val_loss"] = float(m.group(1))
                continue
            m = _DICE_RE.search(line)
            if m:
                vals = [float(v) for v in _DICE_VAL_RE.findall(m.group(1))]
                if len(vals) == 1:
                    current["mean_dice"] = vals[0]
                else:
                    current["mean_dice"] = sum(vals) / len(vals)
                    for i, v in enumerate(vals, start=1):
                        current[f"dice_{i}"] = v
                continue
            m = _TIME_RE.search(line)
            if m:
                current["epoch_time_s"] = float(m.group(1))
                continue

    if current is not None and "epoch" in current:
        rows.append(current)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("epoch").reset_index(drop=True)
    df["epoch"] = df["epoch"].astype(int)

    # Resolve class names from dataset.json if not provided
    if class_names is None:
        name_map = _resolve_class_names(log_path)
        class_names = [name_map.get(i, f"cls{i}") for i in sorted(name_map)]

    # Rename per-class dice columns if names available
    if class_names:
        renames = {f"dice_{i+1}": f"dice_{name}" for i, name in enumerate(class_names) if f"dice_{i+1}" in df.columns}
        if renames:
            df = df.rename(columns=renames)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _short_name(trainer_name: str) -> str:
    """Abbreviate trainer name for display."""
    return (
        trainer_name.replace("nnUNetTrainer", "")
        .replace("__nnUNetPlans__3d_fullres", "")
        .replace("__nnUNetResEncUNetMPlans__3d_fullres", " (ResEncM)")
        .strip("_")
        or "default"
    )


def main():
    parser = argparse.ArgumentParser(description="Parse nnUNet training logs into a CSV + summary table")
    parser.add_argument("--log-dir", required=True, help="Root directory of extracted training logs")
    parser.add_argument("--output", default=None, help="Save merged epoch-level DataFrame as CSV")
    parser.add_argument("--summary", default=None, help="Save per-fold best-epoch summary as CSV")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"ERROR: {log_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    all_dfs: list[pd.DataFrame] = []
    summaries: list[dict] = []
    total_folds = 0

    for txt in sorted(log_dir.rglob("training_log_*.txt")):
        rel = txt.relative_to(log_dir)
        parts = rel.parts
        ds_name = parts[0]
        trainer_name = parts[1]
        fold = parts[2].replace("fold_", "")

        df = parse_nnunet_log(str(txt))
        if df.empty:
            continue

        df["dataset"] = ds_name
        df["trainer"] = trainer_name
        df["fold"] = int(fold)
        all_dfs.append(df)
        total_folds += 1

        best = df.loc[df["mean_dice"].idxmax()]
        summaries.append(
            {
                "dataset": ds_name,
                "trainer": trainer_name,
                "fold": int(fold),
                "best_epoch": int(best["epoch"]),
                "best_ema_dice": best["mean_dice"],
                "train_loss": best["train_loss"],
                "val_loss": best["val_loss"],
                "total_epochs": int(df["epoch"].max()),
            }
        )

    if not all_dfs:
        print("No training logs found.")
        sys.exit(0)

    merged = pd.concat(all_dfs, ignore_index=True)

    # ── Save CSV ───────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False)
        print(f"Merged epoch-level data ({len(merged)} rows) saved to {out_path}")

    if args.summary:
        sum_df = pd.DataFrame(summaries).sort_values(["dataset", "trainer", "fold"])
        out_path = Path(args.summary)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sum_df.to_csv(out_path, index=False)
        print(f"Per-fold summary ({len(sum_df)} folds) saved to {out_path}")

    # ── Print summary table ────────────────────────────────────────────
    print()
    print("=" * 95)
    print(f"{'Dataset/Trainer':<60} {'Fold':>4}  {'BestEMA':>7}  {'@Ep':>5}  {'Ep':>5}")
    print("-" * 95)

    prev_ds = ""
    for s in summaries:
        ds = s["dataset"].replace("Dataset", "").split("_")[0]
        if ds != prev_ds:
            if prev_ds:
                print()
            prev_ds = ds
        short = f"{s['dataset']}/{_short_name(s['trainer'])}"
        if len(short) > 59:
            short = short[:56] + "..."
        print(f"{short:<60}  {s['fold']:>4}  {s['best_ema_dice']:>7.4f}  {s['best_epoch']:>5}  {s['total_epochs']:>5}")

    print("-" * 95)
    print(f"{total_folds} fold(s) total\n")


if __name__ == "__main__":
    main()
