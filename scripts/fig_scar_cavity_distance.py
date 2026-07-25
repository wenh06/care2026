"""Plot scar-to-cavity distance distribution for the paper.

Computes the 3D Euclidean distance from each ground-truth scar voxel to the
nearest ground-truth cavity voxel across all 60 Task~1 training cases, then
plots a cumulative histogram showing the fraction of scar voxels covered as a
function of distance from the cavity wall.

Usage::

    python scripts/fig_scar_cavity_distance.py --db-dir <CARE2026_data_root> \\
        --output figures/scar_cavity_distance.pdf
"""

import argparse
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm.auto import tqdm

# Use non-interactive backend
matplotlib.use("Agg")

# Project-consistent style
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


def compute_distances(db_dir: Path, max_dist_mm: float = 20.0) -> List[float]:
    """Compute scar-to-cavity distances for every scar voxel across all cases.

    Returns a flat list of distances (in mm) aggregated over all scar voxels.
    """
    task1_dir = db_dir / "LA scar quantification（MRI）" / "train_data"
    records = sorted(
        [d for d in task1_dir.iterdir() if d.is_dir() and d.name.startswith("train_")],
        key=lambda p: int(p.name.split("_")[1]),
    )

    all_distances: List[float] = []
    spacing_info: dict = {}

    for rec_dir in tqdm(records, desc="Computing distances", unit="case"):
        scar_path = rec_dir / "scarSegImgM.nii.gz"
        cavity_path = rec_dir / "atriumSegImgMO.nii.gz"
        img_path = rec_dir / "enhanced.nii.gz"
        if not scar_path.exists() or not cavity_path.exists():
            continue

        scar = (nib.load(str(scar_path)).get_fdata() > 0).astype(np.uint8)
        cavity = (nib.load(str(cavity_path)).get_fdata() > 0).astype(np.uint8)
        if scar.sum() == 0 or cavity.sum() == 0:
            continue

        # Get voxel spacing from image header
        zooms = nib.load(str(img_path)).header.get_zooms()[:3]  # (z, y, x) in mm
        spacing_info[rec_dir.name] = tuple(float(z) for z in zooms)

        # 3D Euclidean distance from each voxel to nearest cavity voxel
        # distance_transform_edt returns distance in voxel units; scale by spacing
        dist_vox = distance_transform_edt(~cavity.astype(bool), sampling=zooms)
        scar_distances = dist_vox[scar.astype(bool)]
        all_distances.extend(scar_distances.tolist())

    # Print summary
    dists = np.array(all_distances)
    print(f"\nScar voxel statistics across {len(records)} cases:")
    print(f"  Total scar voxels: {len(dists):,}")
    print(f"  Mean distance to cavity: {dists.mean():.2f} mm")
    print(f"  Median distance to cavity: {np.median(dists):.2f} mm")
    print(f"  Max distance: {dists.max():.2f} mm")
    for d_mm in [2, 3, 4, 5, 6, 8, 10]:
        cov = (dists <= d_mm).mean() * 100
        print(f"  Coverage at {d_mm} mm: {cov:.1f}%")

    return all_distances


def plot_histogram(distances: List[float], output_path: Path):
    """Plot cumulative histogram of scar-to-cavity distances."""
    dists = np.array(distances)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))

    # --- Left: probability density histogram ---
    ax = axes[0]
    max_dist = 10.0
    bins = np.linspace(0, max_dist, 81)  # ~0.125 mm bins
    ax.hist(
        dists[dists <= max_dist],
        bins=bins,
        density=False,
        weights=np.ones_like(dists[dists <= max_dist]) / len(dists),
        color="#2171b5",
        edgecolor="white",
        linewidth=0.3,
        alpha=0.85,
    )
    ax.set_xlabel("Distance to nearest cavity voxel (mm)")
    ax.set_ylabel("Fraction of scar voxels")

    # --- Right: cumulative coverage ---
    ax = axes[1]
    max_plot = 10.0
    xs = np.linspace(0, max_plot, 301)
    ys = [(dists <= x).mean() * 100 for x in xs]
    ax.plot(xs, ys, color="#2171b5", linewidth=1.8)
    # Mark key thresholds
    for d_mm in [2, 5]:
        cov = (dists <= d_mm).mean() * 100
        ax.axvline(x=d_mm, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.text(d_mm + 0.15, cov - 6, f"{cov:.1f}%", fontsize=8, color="#333333", va="top", ha="left")
    ax.set_xlabel("Distance threshold (mm)")
    ax.set_ylabel("Cumulative coverage (%)")
    ax.set_xlim(0, max_plot)
    ax.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"\nFigure saved to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot scar-to-cavity distance distribution for the paper.")
    parser.add_argument("--db-dir", required=True, help="CARE2026 data root")
    parser.add_argument(
        "--output",
        type=str,
        default="figures/scar_cavity_distance.pdf",
        help="Output figure path (default: figures/scar_cavity_distance.pdf)",
    )
    parser.add_argument("--max-dist", type=float, default=20.0, help="Maximum distance to consider (mm)")
    args = parser.parse_args()

    db_dir = Path(args.db_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    distances = compute_distances(db_dir, max_dist_mm=args.max_dist)
    plot_histogram(distances, output_path)


if __name__ == "__main__":
    main()
