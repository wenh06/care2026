"""Centroid-based 3D cropping shared by data prep and inference."""

from typing import Tuple

import numpy as np


def centroid_crop_3d(
    image: np.ndarray,
    centroid: Tuple[int, int, int],
    crop_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, Tuple[int, int, int], Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """Crop *image* around *centroid*, keeping the centroid centred.

    The crop is always centred on *centroid*; when the crop window extends
    beyond image bounds, zero-padding is added **evenly on both sides** so
    the valid data stays in the middle of the crop.  This matches the
    legacy ``_crop_coords`` behaviour and ensures the region of interest
    is never shifted to one end of the crop.

    Parameters
    ----------
    image : np.ndarray, shape (H, W, D)
    centroid : (cx, cy, cz)
        Centre of the crop window (voxel coordinates).
    crop_shape : (cH, cW, cD)
        Target crop size in voxels.

    Returns
    -------
    cropped : np.ndarray of shape *crop_shape* (padded as needed)
    start_xyz : (x0, y0, z0) — inclusive start index of valid data in *image*
    pad_xyz : ((pb_x, pa_x), (pb_y, pa_y), (pb_z, pa_z)) — zero-padding
        *before* and *after* each axis
    """
    cH, cW, cD = crop_shape
    H, W, D = image.shape
    cx, cy, cz = centroid

    def _crop_coords(center: int, size: int, dim_len: int) -> Tuple[int, int, int, int]:
        half = size // 2
        v_start = center - half
        v_end = v_start + size
        pb = max(0, -v_start)
        pa = max(0, v_end - dim_len)
        start = max(0, v_start)
        end = min(dim_len, v_end)
        return start, end, pb, pa

    xs, xe, pb_x, pa_x = _crop_coords(cx, cH, H)
    ys, ye, pb_y, pa_y = _crop_coords(cy, cW, W)
    zs, ze, pb_z, pa_z = _crop_coords(cz, cD, D)

    cropped = image[xs:xe, ys:ye, zs:ze]
    if pb_x > 0 or pa_x > 0 or pb_y > 0 or pa_y > 0 or pb_z > 0 or pa_z > 0:
        cropped = np.pad(
            cropped,
            [(pb_x, pa_x), (pb_y, pa_y), (pb_z, pa_z)],
            mode="constant",
            constant_values=0.0,
        )

    return cropped, (xs, ys, zs), ((pb_x, pa_x), (pb_y, pa_y), (pb_z, pa_z))
