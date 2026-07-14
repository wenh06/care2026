"""Centroid-based 3D cropping shared by data prep and inference."""

from typing import Tuple

import numpy as np


def centroid_crop_3d(
    image: np.ndarray,
    centroid: Tuple[int, int, int],
    crop_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, Tuple[int, int, int], Tuple[int, int, int]]:
    """Crop *image* around *centroid* with ``_clamp``-style window placement.

    The crop window is shifted inward to stay within image bounds.
    Zero-padding is only applied when the image itself is smaller than
    *crop_shape* (never for images at least as large as the crop).

    Parameters
    ----------
    image : np.ndarray, shape (H, W, D)
    centroid : (cx, cy, cz)
        Centre of the crop window (voxel coordinates).
    crop_shape : (cH, cW, cD)
        Target crop size in voxels.

    Returns
    -------
    cropped : np.ndarray of shape *crop_shape* (or padded to it)
    start_xyz : (x0, y0, z0) — crop start index in *image*
    pad_xyz : (px, py, pz) — zero-padding added at the end of each axis
    """
    cH, cW, cD = crop_shape
    H, W, D = image.shape
    cx, cy, cz = centroid

    def _clamp_start(center: int, size: int, max_dim: int) -> int:
        start = center - size // 2
        return int(np.clip(start, 0, max(max_dim - size, 0)))

    x0 = _clamp_start(cx, cH, H)
    y0 = _clamp_start(cy, cW, W)
    z0 = _clamp_start(cz, cD, D)

    cropped = image[x0 : x0 + cH, y0 : y0 + cW, z0 : z0 + cD]
    px = max(0, cH - cropped.shape[0])
    py = max(0, cW - cropped.shape[1])
    pz = max(0, cD - cropped.shape[2])
    if px > 0 or py > 0 or pz > 0:
        cropped = np.pad(cropped, [(0, px), (0, py), (0, pz)], mode="constant", constant_values=0.0)

    return cropped, (x0, y0, z0), (px, py, pz)
