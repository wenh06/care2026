"""
Inference utilities for the CARE 2026 Left Atrium challenge.

Provides volume-level prediction functions for:

- MRI (Tasks 1 & 2): two-stage coarse-to-fine inference with the dual-head VNet.
  Stage 1 uses the whole volume (resized) to locate the LA bounding box; Stage 2
  crops to that box, resizes to the canonical patch shape, and returns both the
  LA cavity mask and the scar mask.

- CT (Task 3): sliding-window inference with 128³ patches and Gaussian overlap
  weighting to avoid hard boundary artefacts.

Both functions support test-time augmentation (TTA) via axis-flip averaging.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from const import CT_HU_MAX, CT_HU_MIN, CT_NUM_CLASSES, CT_TARGET_SPACING, MRI_PATCH_SHAPE
from data_reader import CARE2026_CT, CARE2026_MRI
from outputs import CARE2026Outputs

__all__ = [
    "predict_mri",
    "predict_ct",
    "sliding_window_inference",
]

# Axes used for TTA flip combinations (spatial axes of a 5-D (B,C,H,W,D) tensor)
_TTA_FLIP_AXES: List[Tuple[int, ...]] = [
    (),  # no flip
    (2,),
    (3,),
    (4,),
    (2, 3),
    (2, 4),
    (3, 4),
    (2, 3, 4),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gaussian_weight(patch_size: int, sigma_scale: float = 0.125) -> np.ndarray:
    """Gaussian weight map for a cubic patch of side *patch_size*."""
    sigma = patch_size * sigma_scale
    coords = np.arange(patch_size) - patch_size / 2.0 + 0.5
    g1d = np.exp(-(coords**2) / (2 * sigma**2))
    g3d = g1d[:, None, None] * g1d[None, :, None] * g1d[None, None, :]
    g3d = g3d / g3d.max()
    return g3d.astype(np.float32)


def _resample_3d(arr: np.ndarray, target_shape: Sequence[int], mode: str = "trilinear") -> np.ndarray:
    """Trilinear or nearest-neighbour resampling via ``torch.nn.functional``."""
    t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    align = mode == "trilinear"
    out = F.interpolate(t, size=tuple(target_shape), mode=mode, align_corners=align if align else None)
    return out.squeeze().numpy().astype(arr.dtype)


def _resample_mask(mask: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """Nearest-neighbour resampling for integer masks."""
    t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    out = F.interpolate(t, size=tuple(target_shape), mode="nearest")
    return out.squeeze().numpy().astype(np.uint8)


# ---------------------------------------------------------------------------
# MRI inference
# ---------------------------------------------------------------------------


def _run_mri_model(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward pass through the MRI dual-head model.

    Parameters
    ----------
    model : MRI model (CARE2026_MRI_Model)
    img_tensor : (1, 1, H, W, D) float32 tensor, already on CPU
    device : inference device

    Returns
    -------
    la_prob : (2, H, W, D) float32 softmax probabilities
    scar_prob : (2, H, W, D) float32 softmax probabilities
    """
    img_tensor = img_tensor.to(device, dtype=torch.float32)
    out = model.forward(img_tensor)
    la_prob = torch.softmax(out["la_logits"], dim=1).squeeze(0).cpu().numpy()  # (2, H, W, D)
    scar_prob = torch.softmax(out["scar_logits"], dim=1).squeeze(0).cpu().numpy()  # (2, H, W, D)
    return la_prob, scar_prob


def _mri_tta(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """TTA: average softmax probabilities over all 8 flip combinations."""
    la_acc = np.zeros((2, *img_tensor.shape[2:]), dtype=np.float32)
    scar_acc = np.zeros_like(la_acc)

    for axes in _TTA_FLIP_AXES:
        aug = torch.flip(img_tensor, dims=list(axes)) if axes else img_tensor
        la_p, scar_p = _run_mri_model(model, aug, device)
        if axes:
            # Flip back the spatial axes of the probability maps (spatial axes 1..3)
            spatial_axes = tuple(ax - 1 for ax in axes)  # (2,3,4) → (1,2,3)
            la_p = np.flip(la_p, axis=spatial_axes).copy()
            scar_p = np.flip(scar_p, axis=spatial_axes).copy()
        la_acc += la_p
        scar_acc += scar_p

    n = len(_TTA_FLIP_AXES)
    return la_acc / n, scar_acc / n


def predict_mri(
    img_path: Union[str, Path],
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
    use_tta: bool = True,
    patch_shape: Tuple[int, int, int] = MRI_PATCH_SHAPE,
    pad: int = 7,
) -> CARE2026Outputs:
    """Run coarse-to-fine MRI inference and return predictions in the original voxel space.

    The two-stage strategy mirrors the training pre-processing:

    1. **Coarse pass**: resize the entire volume to *patch_shape* and run the
       model's LA-cavity head to get a coarse LA mask.
    2. **Fine pass**: un-resize the coarse mask to the original space, compute
       the LA bounding box, crop the original volume to that box (+ *pad* voxels),
       resize the crop to *patch_shape*, and run the full model.  The fine
       predictions are then un-resized and placed back into an output volume of
       the original shape.

    Parameters
    ----------
    img_path : path-like
        Path to the LGE-MRI NIfTI file (``enhanced.nii.gz``).
    model : CARE2026_MRI_Model
        Trained dual-head VNet (must already be on the correct device and in eval
        mode).
    device : torch.device, optional
        Inference device.  Defaults to the model's current device.
    use_tta : bool, default True
        Whether to apply 8-fold flip TTA.
    patch_shape : (H, W, D), default MRI_PATCH_SHAPE
        Canonical spatial shape used during training.
    pad : int, default 7
        Voxel padding added around the coarse LA bounding box.

    Returns
    -------
    CARE2026Outputs
        ``la_mask`` and ``scar_mask`` both in the original voxel space (same
        shape as the input volume).  ``source_affine`` and ``source_header``
        are populated from the input NIfTI.
    """
    if device is None:
        device = next(model.parameters()).device

    nii = nib.load(str(img_path))
    image_raw = nii.get_fdata().astype(np.float32)  # (H, W, D)
    orig_shape = image_raw.shape  # spatial dims of the original volume

    # --- Stage 1: coarse LA localisation ---
    img_coarse = CARE2026_MRI.resample_data(image_raw, patch_shape)
    mean, std = float(img_coarse.mean()), float(img_coarse.std())
    img_coarse_norm = (img_coarse - mean) / (std + 1e-8)

    t_coarse = torch.from_numpy(img_coarse_norm).unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D)
    model.eval()
    with torch.no_grad():
        if use_tta:
            la_prob_c, _ = _mri_tta(model, t_coarse, device)
        else:
            la_prob_c, _ = _run_mri_model(model, t_coarse, device)

    la_coarse = (la_prob_c.argmax(axis=0)).astype(np.uint8)  # (H', W', D')

    # Un-resize the coarse LA mask to the original shape to compute the bbox
    la_coarse_orig = _resample_mask(la_coarse, orig_shape)

    # Bounding box with padding
    fg = np.argwhere(la_coarse_orig > 0)
    if len(fg) == 0:
        # Fallback: LA not found — use whole volume
        x0, y0, z0 = 0, 0, 0
        x1, y1, z1 = orig_shape
    else:
        mins = fg.min(axis=0)
        maxs = fg.max(axis=0)
        x0 = max(0, mins[0] - pad)
        y0 = max(0, mins[1] - pad)
        z0 = max(0, mins[2] - pad)
        x1 = min(orig_shape[0], maxs[0] + pad)
        y1 = min(orig_shape[1], maxs[1] + pad)
        z1 = min(orig_shape[2], maxs[2] + pad)

    # --- Stage 2: fine segmentation on the cropped region ---
    crop = image_raw[x0:x1, y0:y1, z0:z1]
    crop_shape = crop.shape

    img_fine = CARE2026_MRI.resample_data(crop, patch_shape)
    mean, std = float(img_fine.mean()), float(img_fine.std())
    img_fine_norm = (img_fine - mean) / (std + 1e-8)

    t_fine = torch.from_numpy(img_fine_norm).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        if use_tta:
            la_prob_f, scar_prob_f = _mri_tta(model, t_fine, device)
        else:
            la_prob_f, scar_prob_f = _run_mri_model(model, t_fine, device)

    la_fine = la_prob_f.argmax(axis=0).astype(np.uint8)  # (H', W', D')
    scar_fine = scar_prob_f.argmax(axis=0).astype(np.uint8)

    # Un-resize predictions back to the cropped region shape
    la_crop = _resample_mask(la_fine, crop_shape)
    scar_crop = _resample_mask(scar_fine, crop_shape)

    # Place cropped predictions into a full-size output volume
    la_out = np.zeros(orig_shape, dtype=np.uint8)
    scar_out = np.zeros(orig_shape, dtype=np.uint8)
    la_out[x0:x1, y0:y1, z0:z1] = la_crop
    scar_out[x0:x1, y0:y1, z0:z1] = scar_crop

    return CARE2026Outputs(
        task="mri",
        la_mask=la_out,
        scar_mask=scar_out,
        source_affine=nii.affine,
        source_header=nii.header,
    )


# ---------------------------------------------------------------------------
# CT inference — sliding window
# ---------------------------------------------------------------------------


def sliding_window_inference(
    volume: np.ndarray,
    model_fn,
    patch_size: int,
    stride: int,
    n_classes: int,
    device: torch.device,
    use_gaussian: bool = True,
) -> np.ndarray:
    """Sliding-window inference on a 3-D volume.

    Parameters
    ----------
    volume : (H, W, D) float32 numpy array
        Pre-processed image (already HU-clipped and normalised to [0,1]).
    model_fn : callable
        Function ``(tensor: (1,1,ps,ps,ps)) → softmax: (n_classes,ps,ps,ps)``
        that runs the model and returns class probabilities as a numpy array.
    patch_size : int
        Cubic patch side length (voxels).
    stride : int
        Step between patch centres.
    n_classes : int
        Number of output classes.
    device : torch.device
        Inference device.
    use_gaussian : bool, default True
        Weight overlapping patches by a Gaussian map (smoother boundaries).

    Returns
    -------
    numpy.ndarray of shape (H, W, D)
        Predicted class label per voxel.
    """
    H, W, D = volume.shape
    ps = patch_size

    # Pad volume so every dimension is a multiple of stride (or at least ≥ ps)
    def _pad_dim(sz: int) -> int:
        if sz < ps:
            return ps - sz
        rem = (sz - ps) % stride
        return (stride - rem) % stride

    pH, pW, pD = _pad_dim(H), _pad_dim(W), _pad_dim(D)
    vol_pad = np.pad(volume, ((0, pH), (0, pW), (0, pD)), mode="constant", constant_values=0.0)
    H2, W2, D2 = vol_pad.shape

    weight_map = _make_gaussian_weight(ps) if use_gaussian else np.ones((ps, ps, ps), dtype=np.float32)

    prob_acc = np.zeros((n_classes, H2, W2, D2), dtype=np.float32)
    weight_acc = np.zeros((H2, W2, D2), dtype=np.float32)

    xs = list(range(0, H2 - ps + 1, stride))
    ys = list(range(0, W2 - ps + 1, stride))
    zs = list(range(0, D2 - ps + 1, stride))

    for x in xs:
        for y in ys:
            for z in zs:
                patch = vol_pad[x : x + ps, y : y + ps, z : z + ps]
                t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
                prob = model_fn(t)  # (n_classes, ps, ps, ps)
                prob_acc[:, x : x + ps, y : y + ps, z : z + ps] += prob * weight_map
                weight_acc[x : x + ps, y : y + ps, z : z + ps] += weight_map

    # Avoid division by zero in un-touched regions
    weight_acc = np.maximum(weight_acc, 1e-8)
    prob_acc /= weight_acc

    # Trim padding back to original shape
    prob_acc = prob_acc[:, :H, :W, :D]
    return prob_acc.argmax(axis=0).astype(np.uint8)


def _run_ct_model(
    model: torch.nn.Module,
    patch_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Run CPS CT model on a single patch, return averaged softmax probabilities.

    Parameters
    ----------
    patch_tensor : (1, 1, ps, ps, ps) float32 tensor
    device : inference device

    Returns
    -------
    numpy.ndarray of shape (n_classes, ps, ps, ps)
    """
    patch_tensor = patch_tensor.to(device, dtype=torch.float32)
    out = model.forward(patch_tensor)
    # Average the two CPS branches
    prob = torch.softmax((out["logits1"] + out["logits2"]) / 2.0, dim=1)
    return prob.squeeze(0).cpu().numpy()  # (n_classes, ps, ps, ps)


def _ct_tta_model_fn(model: torch.nn.Module, device: torch.device, use_tta: bool):
    """Return a callable suitable for :func:`sliding_window_inference`."""

    def _fn(patch_t: torch.Tensor) -> np.ndarray:
        if not use_tta:
            return _run_ct_model(model, patch_t, device)
        acc = np.zeros((CT_NUM_CLASSES, patch_t.shape[2], patch_t.shape[3], patch_t.shape[4]), dtype=np.float32)
        for axes in _TTA_FLIP_AXES:
            aug = torch.flip(patch_t, dims=list(axes)) if axes else patch_t
            prob = _run_ct_model(model, aug, device)
            if axes:
                spatial_axes = tuple(ax - 1 for ax in axes)
                prob = np.flip(prob, axis=spatial_axes).copy()
            acc += prob
        return acc / len(_TTA_FLIP_AXES)

    return _fn


def predict_ct(
    img_path: Union[str, Path],
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
    use_tta: bool = True,
    patch_size: int = 128,
    stride: Optional[int] = None,
) -> CARE2026Outputs:
    """Sliding-window CT inference with resampling to/from 0.5 mm isotropic space.

    Steps:

    1. Load CT and record the original shape and affine.
    2. HU clip to ``[CT_HU_MIN, CT_HU_MAX]`` and normalise to ``[0, 1]``.
    3. Resample to 0.5 mm isotropic (matching training preprocessing).
    4. Sliding-window inference (optional TTA).
    5. Resample the predicted label map back to the original shape.
    6. Return a :class:`CARE2026Outputs` with the original affine attached.

    Parameters
    ----------
    img_path : path-like
        Path to the CT NIfTI file.
    model : CARE2026_CT_Model
        Trained CPS model (must already be in eval mode on the correct device).
    device : torch.device, optional
        Inference device.  Defaults to the model's current device.
    use_tta : bool, default True
        Whether to apply 8-fold flip TTA per patch.
    patch_size : int, default 128
        Cubic patch side length (voxels in 0.5 mm isotropic space).
    stride : int, optional
        Sliding-window stride.  Defaults to ``patch_size // 2``.

    Returns
    -------
    CARE2026Outputs
        ``ct_mask`` in the **original** voxel space.  ``source_affine`` and
        ``source_header`` are populated for NIfTI export.
    """
    if device is None:
        device = next(model.parameters()).device
    if stride is None:
        stride = patch_size // 2

    nii = nib.load(str(img_path))
    image_raw = nii.get_fdata().astype(np.float32)  # (H, W, D)
    orig_shape = image_raw.shape
    zooms = np.array(nii.header.get_zooms()[:3], dtype=np.float64)

    # HU clip + normalise
    image = np.clip(image_raw, CT_HU_MIN, CT_HU_MAX)
    image = (image - CT_HU_MIN) / (CT_HU_MAX - CT_HU_MIN)

    # Resample to 0.5 mm isotropic
    target_spacing = np.array(CT_TARGET_SPACING, dtype=np.float64)
    iso_shape = tuple(int(np.round(orig_shape[i] * zooms[i] / target_spacing[i])) for i in range(3))
    image_iso = CARE2026_CT.resample_data(image, iso_shape)

    # Sliding-window inference
    model_fn = _ct_tta_model_fn(model, device, use_tta)
    model.eval()
    with torch.no_grad():
        pred_iso = sliding_window_inference(
            volume=image_iso,
            model_fn=model_fn,
            patch_size=patch_size,
            stride=stride,
            n_classes=CT_NUM_CLASSES,
            device=device,
        )

    # Resample prediction back to original voxel space
    pred_orig = _resample_mask(pred_iso, orig_shape)

    return CARE2026Outputs(
        task="ct",
        ct_mask=pred_orig,
        source_affine=nii.affine,
        source_header=nii.header,
    )


# ---------------------------------------------------------------------------
# CLI entry point (Docker container)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from torch_ecg.utils.misc import str2bool

    from cfg import BaseCfg
    from models import CARE2026_CT_Model, CARE2026_MRI_Model
    from pipeline import run_task1_inference, run_task2_inference, run_task3_inference

    parser = argparse.ArgumentParser(description="CARE2026 Left Atrium Challenge — inference CLI")
    parser.add_argument(
        "--input_dir", type=str, default="/input", help="Validation data root (contains task1/, task2/, task3/)"
    )
    parser.add_argument("--output_dir", type=str, default="/output", help="Results output directory")
    parser.add_argument("--model_dir", type=str, default=str(BaseCfg.model_dir), help="Directory of model checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tta", type=str2bool, default=True)
    parser.add_argument("--tasks", type=str, default="1,2,3", help="Comma-separated list of tasks to run, e.g. '1,2,3'")
    args = parser.parse_args()

    if "cuda" in args.device and not torch.cuda.is_available():
        args.device = "cpu"
        warnings.warn("CUDA not available. Falling back to CPU.")
    device = torch.device(args.device)

    tasks = [int(t.strip()) for t in args.tasks.split(",")]
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()

    mri_model, ct_model = None, None

    if 1 in tasks or 2 in tasks:
        mri_ckpt = model_dir / "mri_model.pth.tar"
        if mri_ckpt.exists():
            mri_model = CARE2026_MRI_Model.from_checkpoint(str(mri_ckpt), device=device)[0]
            mri_model = mri_model.to(device).eval()
        else:
            warnings.warn(f"MRI checkpoint not found: {mri_ckpt}")

    if 3 in tasks:
        ct_ckpt = model_dir / "ct_model.pth.tar"
        if ct_ckpt.exists():
            ct_model = CARE2026_CT_Model.from_checkpoint(str(ct_ckpt), device=device)[0]
            ct_model = ct_model.to(device).eval()
        else:
            warnings.warn(f"CT checkpoint not found: {ct_ckpt}")

    if 1 in tasks and mri_model is not None:
        run_task1_inference(mri_model, input_dir, output_dir, device=device, use_tta=args.tta)
    if 2 in tasks and mri_model is not None:
        run_task2_inference(mri_model, input_dir, output_dir, device=device, use_tta=args.tta)
    if 3 in tasks and ct_model is not None:
        run_task3_inference(ct_model, input_dir, output_dir, device=device, use_tta=args.tta)
