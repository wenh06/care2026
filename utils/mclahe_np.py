#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pure-numpy re-implementation of :mod:`utils.mclahe` (the TensorFlow version
from https://github.com/VincentStimper/mclahe), operator for operator, so the
TF runtime dependency can be dropped.

Same signature and defaults as ``utils.mclahe.mclahe``; numerically equivalent
to the TF implementation up to float rounding (validated to max|delta| ~1e-6
on 2D/3D volumes; exact binning requires a power-of-two ``nbins`` — the
default 128 is one).

Only the ``adaptive_hist_range=False`` path is implemented; that is the only
configuration used in this repo (``mclahe(image)`` with defaults).
"""

from itertools import product

import numpy as np

__all__ = ["mclahe"]


def _blockify(arr, nblk, kernel, dim):
    """Rearrange a grid of shape (nblk[i] * kernel[i], ...) into
    (nblk..., kernel...) blocks (same reshape+transpose as the TF graph)."""
    new_shape = np.reshape(np.column_stack((nblk, kernel)), (2 * dim,))
    perm = tuple(2 * i for i in range(dim)) + tuple(2 * i + 1 for i in range(dim))
    return np.transpose(arr.reshape(new_shape), perm)


def mclahe(x, kernel_size=None, clip_limit=0.01, nbins=128, adaptive_hist_range=False):
    """Contrast limited adaptive histogram equalization (pure numpy).

    Drop-in replacement for ``utils.mclahe.mclahe``; output scaled to [0, 1].
    """
    if adaptive_hist_range:
        raise NotImplementedError("adaptive_hist_range=True is not ported (unused in this repo)")

    x = np.asarray(x, dtype=np.float32)
    n_bins = int(nbins)

    if kernel_size is None:
        kernel_size = tuple(max(1, s // 8) for s in x.shape)
    kernel_size = np.asarray(kernel_size, dtype=np.int64)

    assert len(kernel_size) == x.ndim

    dim = x.ndim

    # Normalize data
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max <= x_min:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - x_min) / (x_max - x_min)

    # Pad data
    x_shape = np.array(x.shape, dtype=np.int64)
    padding_x_length = kernel_size - 1 - ((x_shape - 1) % kernel_size)
    padding_x = np.column_stack(((padding_x_length + 1) // 2, padding_x_length // 2))
    padding_hist = np.column_stack((kernel_size // 2, (kernel_size + 1) // 2)) + padding_x
    x_hist_padded = np.pad(x, padding_hist, "symmetric")

    # Interpolation grid: starts at kernel//2 inside the (larger) hist padding
    n_blocks = np.ceil(np.array(x.shape) / kernel_size).astype(np.int64)
    grid_begin = kernel_size // 2
    grid_shape = x_shape + padding_x_length
    x_grid = x_hist_padded[tuple(slice(b, b + s) for b, s in zip(grid_begin, grid_shape))]

    # Per-block histograms over the (n_blocks+1) corner-centered blocks
    n_blocks_hist = n_blocks + 1
    x_hist_blocks = _blockify(x_hist_padded, n_blocks_hist, kernel_size, dim)
    kprod = int(np.prod(kernel_size))
    nrows = int(np.prod(n_blocks_hist))
    flat = (x_hist_blocks.reshape(nrows, kprod) * n_bins).astype(np.int32)
    np.minimum(flat, n_bins - 1, out=flat)
    flat += (np.arange(nrows, dtype=np.int32) * n_bins)[:, None]
    hist = np.bincount(flat.ravel(), minlength=nrows * n_bins)
    del flat, x_hist_blocks
    hist = hist.reshape(tuple(n_blocks_hist) + (n_bins,)).astype(np.float32)

    # Clip histogram, redistribute excess uniformly over all bins
    thresh = float(np.prod(kernel_size)) * clip_limit
    n_to_high = np.sum(np.maximum(hist - thresh, 0.0), axis=-1, keepdims=True)
    hist_clipped = np.minimum(hist, thresh) + n_to_high / n_bins
    cdf = np.cumsum(hist_clipped, axis=-1)
    cdf_min = cdf[..., :1]
    cdf_max = cdf[..., -1:]
    cdf_norm = np.where(cdf_min == cdf_max, np.ones_like(cdf_max), cdf_max - cdf_min)
    mapping = (cdf - cdf_min) / cdf_norm  # (n_blocks_hist..., nbins)

    # Interpolation blocks and their global bins
    x_blocks = _blockify(x_grid, n_blocks, kernel_size, dim)
    block_bins = (x_blocks * n_bins).astype(np.int32)
    np.minimum(block_bins, n_bins - 1, out=block_bins)
    del x_blocks, x_grid, x_hist_padded

    # Blend the 2^dim corner mappings (trilinear for 3D volumes).
    # Gather via flat indices: row_base maps block coords -> row offset in the
    # flattened (n_blocks_hist..., nbins) mapping table.
    map_flat = mapping.reshape(-1)
    row_base = (np.arange(int(np.prod(n_blocks_hist)), dtype=np.int32) * n_bins).reshape(tuple(n_blocks_hist))
    res = np.zeros(block_bins.shape, dtype=np.float32)
    for ind_map in product([0, 1], repeat=dim):
        corner = tuple(slice(im, im + nb) for im, nb in zip(ind_map, n_blocks))
        r = row_base[corner]
        r = r.reshape(r.shape + (1,) * dim)
        res_sub = map_flat.take(r + block_bins)
        for axis in range(dim):
            coeff = np.arange(kernel_size[axis], dtype=np.float32) / kernel_size[axis]
            if kernel_size[axis] % 2 == 0:
                coeff = 0.5 / kernel_size[axis] + coeff
            if ind_map[axis] == 0:
                coeff = 1.0 - coeff
            shape = [1] * (dim + axis) + [int(kernel_size[axis])] + [1] * (dim - 1 - axis)
            res_sub = res_sub * np.reshape(coeff, shape)
        res += res_sub

    # Rescale
    res_min, res_max = float(res.min()), float(res.max())
    res = (res - res_min) / (res_max - res_min)

    # Blocks back to grid layout, crop the half-tile padding
    inv_perm = tuple(j for axis in range(dim) for j in (axis, axis + dim))
    res_grid = np.transpose(res, inv_perm).reshape(tuple(int(n * k) for n, k in zip(n_blocks, kernel_size)))
    crop = tuple(slice(int(px), int(px) + s) for px, s in zip(padding_x[:, 0], x_shape))
    return res_grid[crop]
