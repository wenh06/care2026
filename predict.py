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

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
from scipy.ndimage import label as _nd_label

from const import (
    CT_HU_MAX,
    CT_HU_MIN,
    CT_NUM_CLASSES,
    CT_TARGET_SPACING,
    MRI_CANONICAL_SHAPE,
    MRI_STAGE1_SHAPE,
    MRI_STAGE2_CROP_SHAPE,
)
from data_reader import CARE2026_CT
from outputs import CARE2026Outputs
from utils.mclahe import mclahe as _mclahe

__all__ = [
    "predict_mri_two_stage",
    "predict_ct",
    "sliding_window_inference",
    "keep_largest_component",
    "postprocess_mri_masks",
    "postprocess_ct_mask",
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


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary (0/1) integer array of any shape.

    Returns
    -------
    np.ndarray
        Same shape/dtype as *mask* with all but the largest connected
        component zeroed out.  Returns *mask* unchanged if it is
        all-zero or contains exactly one component.
    """
    if mask.max() == 0:
        return mask
    labeled, n_comp = _nd_label(mask)
    if n_comp == 1:
        return mask
    # bincount index 0 is background; shift by 1
    sizes = np.bincount(labeled.ravel())[1:]
    largest_label = int(sizes.argmax()) + 1
    return (labeled == largest_label).astype(mask.dtype)


def postprocess_mri_masks(
    la_mask: np.ndarray,
    scar_mask: np.ndarray,
    dilation_mm: float = 2.0,
    in_plane_spacing: Tuple[float, float] = (0.625, 0.625),
) -> Tuple[np.ndarray, np.ndarray]:
    """Post-process MRI segmentation outputs.

    Steps applied:

    1. **LA cavity**: keep largest connected component.
    2. **Scar**: constrain to a *dilated* LA cavity mask.  Scar is
       anatomically located in the atrial wall (~1–3 mm thick)
       surrounding the blood pool, NOT inside the cavity itself.
       A dilation of ~2 mm (≈ 3 px in-plane) captures >92 % of
       true scar while suppressing distant false positives.

    Parameters
    ----------
    la_mask, scar_mask : np.ndarray
        Binary (0/1) uint8 arrays in the same voxel space.
    dilation_mm : float, default 2.0
        Dilation radius in millimetres.
    in_plane_spacing : (float, float), default (0.625, 0.625)
        Voxel spacing (mm/px) for the X and Y axes.

    Returns
    -------
    la_mask_clean, scar_mask_clean : np.ndarray
    """
    la_clean = keep_largest_component(la_mask)

    # Dilate the cavity mask to cover the atrial wall (~2 mm)
    dilation_px = max(1, int(np.round(dilation_mm / max(in_plane_spacing))))
    structure = np.ones((dilation_px, dilation_px, 1), dtype=bool)
    la_dilated = binary_dilation(la_clean.astype(bool), structure=structure, iterations=1)

    scar_clean = (scar_mask.astype(bool) & la_dilated).astype(scar_mask.dtype)
    return la_clean, scar_clean


def postprocess_ct_mask(ct_mask: np.ndarray, n_classes: int) -> np.ndarray:
    """Post-process CT multi-class segmentation output.

    For each foreground class independently, keep only the largest connected
    component (each cardiac structure is topologically a single body).

    Parameters
    ----------
    ct_mask : np.ndarray
        Integer array with values in ``[0, n_classes)``.
    n_classes : int
        Total number of classes including background (class 0).

    Returns
    -------
    np.ndarray
        Same shape/dtype as *ct_mask*.
    """
    out = ct_mask.copy()
    for cls in range(1, n_classes):
        binary = (ct_mask == cls).astype(np.uint8)
        if binary.max() == 0:
            continue
        cleaned = keep_largest_component(binary)
        # Zero out voxels that were removed
        out[ct_mask == cls] = 0
        out[cleaned == 1] = cls
    return out


# ---------------------------------------------------------------------------
# MRI inference
# ---------------------------------------------------------------------------


def _check_model_consistency(
    stage1_model: torch.nn.Module,
    stage2_model: torch.nn.Module,
) -> bool:
    """Verify Stage-1 and Stage-2 models are compatible and return ``apply_mclahe``."""
    tc1 = getattr(stage1_model, "train_config", {}) or {}
    tc2 = getattr(stage2_model, "train_config", {}) or {}

    s1_mclahe = bool(tc1.get("apply_mclahe", False))
    s2_mclahe = bool(tc2.get("apply_mclahe", False))
    if s1_mclahe != s2_mclahe:
        raise ValueError(
            f"Stage 1 and Stage 2 models disagree on apply_mclahe: "
            f"Stage 1={s1_mclahe}, Stage 2={s2_mclahe}. "
            "Both models must be trained with the same CLAHE setting."
        )

    s1_task, s2_task = tc1.get("task"), tc2.get("task")
    s1_stage, s2_stage = tc1.get("stage"), tc2.get("stage")
    if s1_task != "mri" or s2_task != "mri":
        raise ValueError(f"Both models must be MRI: Stage 1 task={s1_task}, Stage 2 task={s2_task}")
    if s1_stage != 1 or s2_stage != 2:
        raise ValueError(f"Stage mismatch: Stage 1 stage={s1_stage}, Stage 2 stage={s2_stage} (expected 1 / 2)")

    return s2_mclahe


def _run_stage1_model(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Forward pass through the Stage-1 single-head VNet.

    Parameters
    ----------
    model : CARE2026_MRI_Stage1_Model
    img_tensor : (1, 1, H, W, D) float32 tensor (on CPU)
    device : inference device

    Returns
    -------
    la_prob : (2, H, W, D) float32 softmax probabilities for binary LA
    """
    img_tensor = img_tensor.to(device, dtype=torch.float32)
    out = model.forward(img_tensor)
    la_prob = torch.softmax(out["la_logits"], dim=1).squeeze(0).detach().cpu().numpy()  # (2, H, W, D)
    return la_prob


def _run_stage2_model(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward pass through the Stage-2 model (dual-head or scar-only).

    Returns
    -------
    la_prob, scar_prob : each (2, H, W, D) float32
        ``la_prob`` is all zeros for scar-only models.
    """
    img_tensor = img_tensor.to(device, dtype=torch.float32)
    out = model.forward(img_tensor)
    scar_prob = torch.softmax(out["scar_logits"], dim=1).squeeze(0).detach().cpu().numpy()
    if "la_logits" in out:
        la_prob = torch.softmax(out["la_logits"], dim=1).squeeze(0).detach().cpu().numpy()
    else:
        la_prob = np.zeros_like(scar_prob)
    return la_prob, scar_prob


def _stage1_tta(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """8-fold flip TTA for Stage-1 model."""
    acc = np.zeros((2, *img_tensor.shape[2:]), dtype=np.float32)
    for axes in _TTA_FLIP_AXES:
        aug = torch.flip(img_tensor, dims=list(axes)) if axes else img_tensor
        prob = _run_stage1_model(model, aug, device)
        if axes:
            spatial_axes = tuple(ax - 1 for ax in axes)
            prob = np.flip(prob, axis=spatial_axes).copy()
        acc += prob
    return acc / len(_TTA_FLIP_AXES)


def _stage2_tta(
    model: torch.nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """8-fold flip TTA for Stage-2 model."""
    la_acc = np.zeros((2, *img_tensor.shape[2:]), dtype=np.float32)
    scar_acc = np.zeros_like(la_acc)
    for axes in _TTA_FLIP_AXES:
        aug = torch.flip(img_tensor, dims=list(axes)) if axes else img_tensor
        la_p, scar_p = _run_stage2_model(model, aug, device)
        if axes:
            spatial_axes = tuple(ax - 1 for ax in axes)
            la_p = np.flip(la_p, axis=spatial_axes).copy()
            scar_p = np.flip(scar_p, axis=spatial_axes).copy()
        la_acc += la_p
        scar_acc += scar_p
    n = len(_TTA_FLIP_AXES)
    return la_acc / n, scar_acc / n


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Sample-wise z-score normalisation."""
    return (arr - arr.mean()) / (arr.std() + 1e-8)


def predict_mri_two_stage(
    img_path: Union[str, Path],
    stage1_model: torch.nn.Module,
    stage2_model: torch.nn.Module,
    device: Optional[torch.device] = None,
    use_tta: bool = True,
    apply_mclahe: Optional[bool] = None,
    centroid: Optional[Tuple[int, int, int]] = None,
    s1_threshold: float = 0.5,
    s2_threshold: float = 0.5,
    canonical_shape: Tuple[int, int, int] = MRI_CANONICAL_SHAPE,
    stage1_shape: Tuple[int, int, int] = MRI_STAGE1_SHAPE,
    stage2_crop_shape: Tuple[int, int, int] = MRI_STAGE2_CROP_SHAPE,
) -> CARE2026Outputs:
    """Two-stage MRI inference matching the training pipeline.

    Pipeline
    --------
    1. Load NIfTI → canonical → (optional CLAHE).
    2. **Stage 1**: downsample → 144×144×44 → binary VNet → LA cavity mask
       in canonical space.  Also yields the LA centroid.
    3. **Centroid crop**: 256×256×44 around centroid → resize to 128×128×44
       (training resolution) → z-score.
    4. **Stage 2**: dual-head VNet on 128×128×44 → scar prob map →
       upsample back to 256×256×44 → place in canonical → resample to native.
    5. **Post-process** (native space): keep largest LA component; constrain
       scar to *dilated* Stage-1 LA cavity (~2 mm dilation to cover the
       atrial wall).

    Parameters
    ----------
    centroid : (cx, cy, cz) or None, optional
        Pre-computed LA centroid in canonical space.  When given, Stage 1
        is *skipped* for centroid computation (but still runs to produce
        the LA cavity mask used for post-processing scar constraint).
    """
    if device is None:
        device = next(stage1_model.parameters()).device

    if apply_mclahe is None:
        apply_mclahe = _check_model_consistency(stage1_model, stage2_model)

    # ── Load & canonical ───────────────────────────────────────────────────
    nii = nib.load(str(img_path))
    image_raw = nii.get_fdata().astype(np.float32)
    orig_shape = image_raw.shape
    img_canonical = _resample_3d(image_raw, canonical_shape)
    if apply_mclahe:
        img_canonical = _mclahe(img_canonical)

    # ── Stage 1: coarse LA cavity (always run — needed for scar constraint)
    img_s1 = _resample_3d(img_canonical, stage1_shape)
    img_s1_norm = _zscore(img_s1)
    t_s1 = torch.from_numpy(img_s1_norm).unsqueeze(0).unsqueeze(0)

    stage1_model.eval()
    with torch.no_grad():
        la_prob_s1 = _stage1_tta(stage1_model, t_s1, device) if use_tta else _run_stage1_model(stage1_model, t_s1, device)
    la_mask_s1 = (la_prob_s1[1] >= s1_threshold).astype(np.uint8)

    # Stage-1 LA at canonical resolution (for scar constraint)
    la_s1_canonical = _resample_mask(la_mask_s1, canonical_shape)

    # Centroid
    if centroid is not None:
        cx, cy, cz = centroid
    else:
        fg = np.argwhere(la_s1_canonical > 0)
        cx, cy, cz = fg.mean(axis=0).round().astype(int) if len(fg) > 0 else np.array([s // 2 for s in canonical_shape])

    # ── Centroid crop → resize to training size → Stage 2 ─────────────────
    cH, cW, cD = canonical_shape
    tH, tW, tD = stage2_crop_shape

    def _crop_coords(center: int, size: int, dim_len: int) -> Tuple[int, int, int, int]:
        half = size // 2
        v_start, v_end = center - half, center - half + size
        pb, pa = max(0, -v_start), max(0, v_end - dim_len)
        return max(0, v_start), min(dim_len, v_end), pb, pa

    xs, xe, px0, px1 = _crop_coords(int(cx), tH, cH)
    ys, ye, py0, py1 = _crop_coords(int(cy), tW, cW)
    zs, ze, pz0, pz1 = _crop_coords(int(cz), tD, cD)

    crop = img_canonical[xs:xe, ys:ye, zs:ze]
    if any(p > 0 for p in [px0, px1, py0, py1, pz0, pz1]):
        crop = np.pad(crop, ((px0, px1), (py0, py1), (pz0, pz1)), mode="constant", constant_values=0.0)

    crop_h, crop_w, crop_d = crop.shape
    s2_hw = 128  # train_crop_hw from MRI_Stage2_TrainCfg
    img_s2_norm = _zscore(crop)
    img_s2_resized = _resample_3d(img_s2_norm, (s2_hw, s2_hw, crop_d))
    t_s2 = torch.from_numpy(img_s2_resized).unsqueeze(0).unsqueeze(0)

    stage2_model.eval()
    with torch.no_grad():
        _, scar_prob_s2 = _stage2_tta(stage2_model, t_s2, device) if use_tta else _run_stage2_model(stage2_model, t_s2, device)

    scar_crop = _resample_mask((scar_prob_s2[1] >= s2_threshold).astype(np.uint8), (crop_h, crop_w, crop_d))
    scar_unpad = scar_crop[px0 : tH - px1, py0 : tW - py1, pz0 : tD - pz1]
    scar_canonical = np.zeros(canonical_shape, dtype=np.uint8)
    scar_canonical[xs:xe, ys:ye, zs:ze] = scar_unpad

    # ── Resample to native ─────────────────────────────────────────────────
    la_out = _resample_mask(la_s1_canonical, orig_shape)
    scar_out = _resample_mask(scar_canonical, orig_shape)

    # ── Post-process at native resolution ───────────────────────────────────
    la_out, scar_out = postprocess_mri_masks(la_out, scar_out)

    return CARE2026Outputs(
        task="mri",
        la_mask=la_out,
        scar_mask=scar_out,
        source_affine=nii.affine,
        source_header=nii.header,
    )


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
    return prob.squeeze(0).detach().cpu().numpy()  # (n_classes, ps, ps, ps)


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

    # ── Post-processing ────────────────────────────────────────────────────
    pred_orig = postprocess_ct_mask(pred_orig, CT_NUM_CLASSES)

    return CARE2026Outputs(
        task="ct",
        ct_mask=pred_orig,
        source_affine=nii.affine,
        source_header=nii.header,
    )
