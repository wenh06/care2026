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
from torch_ecg.cfg import CFG

from const import (
    CT_HU_MAX,
    CT_HU_MIN,
    CT_NUM_CLASSES,
    CT_PATCH_SIZE,
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


def _make_gaussian_weight(patch_shape: Union[int, Sequence[int]], sigma_scale: float = 0.125) -> np.ndarray:
    """Gaussian weight map for a 3-D patch.

    Parameters
    ----------
    patch_shape : int or tuple of 3 ints
        Patch side length(s).  If int, a cubic patch is assumed.
    sigma_scale : float
        Sigma = side * sigma_scale per axis.
    """
    if isinstance(patch_shape, int):
        patch_shape = (patch_shape, patch_shape, patch_shape)
    ph, pw, pd = patch_shape
    g3d = np.ones((ph, pw, pd), dtype=np.float32)
    for axis, length in enumerate(patch_shape):
        sigma = length * sigma_scale
        coords = np.arange(length) - length / 2.0 + 0.5
        g1d = np.exp(-(coords**2) / (2 * sigma**2))
        shape = [1, 1, 1]
        shape[axis] = length
        g3d = g3d * g1d.reshape(shape)
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
    dilation_mm: float = 5.0,
    in_plane_spacing: Tuple[float, float] = (0.625, 0.625),
) -> Tuple[np.ndarray, np.ndarray]:
    """Post-process MRI segmentation outputs.

    Steps applied:

    1. **LA cavity**: keep largest connected component.
    2. **Scar**: constrain to a *dilated* LA cavity mask.  Scar is
       anatomically located in the atrial wall (~1–3 mm thick)
       surrounding the blood pool, NOT inside the cavity itself.
       A dilation of ~5 mm (≈ 8 px in-plane) captures >99 % of
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
    if dilation_mm is not None and dilation_mm <= 0:
        raise ValueError(
            f"scar_dilation must be > 0 or None (got {dilation_mm}).  Scar is in the wall, "
            f"not inside the cavity; 0 mm dilation would eliminate nearly all true scar."
        )

    la_clean = keep_largest_component(la_mask)

    if dilation_mm is not None:
        dilation_px = max(1, int(np.round(dilation_mm / max(in_plane_spacing))))
        structure = np.ones((dilation_px, dilation_px, 1), dtype=bool)
        la_dilated = binary_dilation(la_clean.astype(bool), structure=structure, iterations=1)
        scar_clean = (scar_mask.astype(bool) & la_dilated).astype(scar_mask.dtype)
    else:
        scar_clean = scar_mask

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


def _is_nnunet(model: torch.nn.Module) -> bool:
    """Return True if *model* wraps an nnUNetPredictor."""
    return hasattr(model, "_predictor")


def _check_model_consistency(
    stage1_model: torch.nn.Module,
    stage2_model: Optional[torch.nn.Module],
) -> bool:
    """Verify Stage-1 and Stage-2 models are compatible and return ``apply_mclahe``.

    When *stage2_model* is None (e.g. Task 2 cavity-only), only Stage 1's
    MCLAHE setting is used.
    """
    s1_nnunet = _is_nnunet(stage1_model)
    s2_nnunet = _is_nnunet(stage2_model) if stage2_model is not None else s1_nnunet

    if s1_nnunet:
        s1_mclahe = bool((getattr(stage1_model, "config", {}) or {}).get("apply_mclahe", False))
    else:
        s1_mclahe = bool((getattr(stage1_model, "train_config", {}) or {}).get("apply_mclahe", False))
    if stage2_model is None:
        s2_mclahe = s1_mclahe
    elif s2_nnunet:
        s2_mclahe = bool((getattr(stage2_model, "config", {}) or {}).get("apply_mclahe", False))
    else:
        s2_mclahe = bool((getattr(stage2_model, "train_config", {}) or {}).get("apply_mclahe", False))

    if s1_mclahe != s2_mclahe:
        raise ValueError(f"Stage 1 and Stage 2 models disagree on apply_mclahe: " f"Stage 1={s1_mclahe}, Stage 2={s2_mclahe}.")

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
    stage2_model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    use_tta: bool = True,
    apply_mclahe: Optional[bool] = None,
    centroid: Optional[Tuple[int, int, int]] = None,
    s1_threshold: float = 0.5,
    s2_threshold: float = 0.7,
    scar_dilation: Optional[float] = 5.0,
    canonical_shape: Optional[Tuple[int, int, int]] = None,
    stage1_shape: Optional[Tuple[int, int, int]] = None,
    stage2_crop_shape: Optional[Tuple[int, int, int]] = None,
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

    # Read shapes from model config (synced from train_config at init)
    cfg1 = getattr(stage1_model, "config", {}) or {}
    cfg2 = getattr(stage2_model, "config", {}) or {}
    if canonical_shape is None:
        canonical_shape = tuple(cfg2.get("canonical_shape", MRI_CANONICAL_SHAPE))
    if stage1_shape is None:
        stage1_shape = tuple(cfg1.get("patch_shape", MRI_STAGE1_SHAPE))
    if stage2_crop_shape is None:
        stage2_crop_shape = tuple(cfg2.get("patch_shape", MRI_STAGE2_CROP_SHAPE))
    s2_hw = int(cfg2.get("train_crop_hw", 128))

    # ── Load & canonical ───────────────────────────────────────────────────
    nii = nib.load(str(img_path))
    image_raw = nii.get_fdata().astype(np.float32)
    orig_shape = image_raw.shape
    img_canonical = _resample_3d(image_raw, canonical_shape)
    if apply_mclahe:
        img_canonical = _mclahe(img_canonical)

    # ── Stage 1: coarse LA cavity (always run — needed for scar constraint)
    mri_spacing = (0.625, 0.625, 2.5)  # Center A native spacing
    s1_nnunet = _is_nnunet(stage1_model)

    if s1_nnunet:
        stage1_model.eval()
        with torch.no_grad():
            la_s1_canonical = stage1_model.predict(img_canonical, mri_spacing, use_tta=use_tta)
            # nnUNet cavity model outputs classes {0,1} — argmax to binary
            la_s1_canonical = (la_s1_canonical > 0).astype(np.uint8)
    else:
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

    # ── If no Stage 2 model (e.g. Task 2 cavity-only), return LA mask only
    if stage2_model is None:
        la_out = _resample_mask(la_s1_canonical, orig_shape)
        la_out, _ = postprocess_mri_masks(la_out, np.zeros_like(la_out))
        return CARE2026Outputs(
            task="mri",
            la_mask=la_out,
            scar_mask=np.zeros_like(la_out),
            source_affine=nii.affine,
            source_header=nii.header,
        )

    # ── Centroid crop → Stage 2 ──────────────────────────────────────────
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
    s2_nnunet = _is_nnunet(stage2_model)

    scar_cls = getattr(stage2_model, "scar_class_index", 1)

    if s2_nnunet:
        # nnUNet Stage 2: predictor handles normalization + resampling internally
        stage2_model.eval()
        with torch.no_grad():
            pred_crop = stage2_model.predict(crop, mri_spacing, use_tta=use_tta)
        if scar_cls > 1:
            scar_crop = (pred_crop == scar_cls).astype(np.uint8)
        else:
            scar_crop = (pred_crop > 0).astype(np.uint8)
    else:
        # VNet Stage 2: z-score → resize → forward → resize back
        img_s2_norm = _zscore(crop)
        use_sdf = "2ch" in str(cfg2.get("backbone", ""))
        if use_sdf:
            from scipy.ndimage import distance_transform_edt

            la_crop = la_s1_canonical[xs:xe, ys:ye, zs:ze]
            if any(p > 0 for p in [px0, px1, py0, py1, pz0, pz1]):
                la_crop = np.pad(la_crop, ((px0, px1), (py0, py1), (pz0, pz1)), mode="constant", constant_values=0.0)
            sdf_out = distance_transform_edt(1 - la_crop)
            sdf_in = distance_transform_edt(la_crop)
            sdf = np.clip((sdf_out - sdf_in).astype(np.float32) * 0.625 / 4.0, -1.0, 1.0)
            sdf_resized = _resample_3d(sdf, (s2_hw, s2_hw, crop_d))
            img_s2_resized = _resample_3d(img_s2_norm, (s2_hw, s2_hw, crop_d))
            img_2ch = np.stack([img_s2_resized, sdf_resized], axis=0)
            t_s2 = torch.from_numpy(img_2ch).unsqueeze(0)
        else:
            img_s2_resized = _resample_3d(img_s2_norm, (s2_hw, s2_hw, crop_d))
            t_s2 = torch.from_numpy(img_s2_resized).unsqueeze(0).unsqueeze(0)

        stage2_model.eval()
        with torch.no_grad():
            _, scar_prob_s2 = (
                _stage2_tta(stage2_model, t_s2, device) if use_tta else _run_stage2_model(stage2_model, t_s2, device)
            )

        scar_crop = _resample_mask((scar_prob_s2[scar_cls] >= s2_threshold).astype(np.uint8), (crop_h, crop_w, crop_d))

    scar_unpad = scar_crop[px0 : tH - px1, py0 : tW - py1, pz0 : tD - pz1]
    scar_canonical = np.zeros(canonical_shape, dtype=np.uint8)
    scar_canonical[xs:xe, ys:ye, zs:ze] = scar_unpad

    # ── Resample to native ─────────────────────────────────────────────────
    la_out = _resample_mask(la_s1_canonical, orig_shape)
    scar_out = _resample_mask(scar_canonical, orig_shape)

    # ── Post-process at native resolution ───────────────────────────────────
    la_out, scar_out = postprocess_mri_masks(la_out, scar_out, dilation_mm=scar_dilation)

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
    patch_size: Union[int, Sequence[int]],
    stride: Union[int, Sequence[int]],
    n_classes: int,
    device: torch.device,
    use_gaussian: bool = True,
) -> np.ndarray:
    """Sliding-window inference on a 3-D volume.

    Parameters
    ----------
    volume : (H, W, D) float32 numpy array
        Pre-processed image.
    model_fn : callable
        ``(tensor: (1,1,*patch)) → softmax: (n_classes,*patch)``
    patch_size : int or (int, int, int)
        Patch shape.  If int, cubic patch is assumed.
    stride : int or (int, int, int)
        Step between patch centres.  If int, same stride on all axes.
    n_classes : int
    device : torch.device
    use_gaussian : bool

    Returns
    -------
    numpy.ndarray of shape (H, W, D)
        Predicted class label per voxel.
    """
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    ps_h, ps_w, ps_d = patch_size
    st_h, st_w, st_d = stride
    H, W, D = volume.shape

    def _pad_dim(sz: int, ps: int, st: int) -> int:
        if sz < ps:
            return ps - sz
        rem = (sz - ps) % st
        return (st - rem) % st

    pH, pW, pD = _pad_dim(H, ps_h, st_h), _pad_dim(W, ps_w, st_w), _pad_dim(D, ps_d, st_d)
    vol_pad = np.pad(volume, ((0, pH), (0, pW), (0, pD)), mode="constant", constant_values=0.0)
    H2, W2, D2 = vol_pad.shape

    weight_map = _make_gaussian_weight(patch_size) if use_gaussian else np.ones(patch_size, dtype=np.float32)

    prob_acc = np.zeros((n_classes, H2, W2, D2), dtype=np.float32)
    weight_acc = np.zeros((H2, W2, D2), dtype=np.float32)

    xs = list(range(0, H2 - ps_h + 1, st_h))
    ys = list(range(0, W2 - ps_w + 1, st_w))
    zs = list(range(0, D2 - ps_d + 1, st_d))

    for x in xs:
        for y in ys:
            for z in zs:
                patch = vol_pad[x : x + ps_h, y : y + ps_w, z : z + ps_d]
                t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
                prob = model_fn(t)  # (n_classes, ps_h, ps_w, ps_d)
                prob_acc[:, x : x + ps_h, y : y + ps_w, z : z + ps_d] += prob * weight_map
                weight_acc[x : x + ps_h, y : y + ps_w, z : z + ps_d] += weight_map

    weight_acc = np.maximum(weight_acc, 1e-8)
    prob_acc /= weight_acc
    prob_acc = prob_acc[:, :H, :W, :D]
    return prob_acc.argmax(axis=0).astype(np.uint8)


def _run_ct_model(
    model: torch.nn.Module,
    patch_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Run CT model on a single patch, return softmax probabilities.

    Handles both standard models (logits1=tensor) and nnUNet-style
    models with deep supervision (logits1=list), taking the
    full-resolution output (logits[0] for PlainConvUNet).
    """
    patch_tensor = patch_tensor.to(device, dtype=torch.float32)
    out = model.forward(patch_tensor)
    if "logits2" in out:
        logits = (out["logits1"] + out["logits2"]) / 2.0
    else:
        logits = out["logits1"]
    # Deep supervision: list → take full-resolution output
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    prob = torch.softmax(logits, dim=1)
    return prob.squeeze(0).detach().cpu().numpy()


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
    patch_size: Union[int, Sequence[int]] = 128,
    stride: Union[int, Sequence[int], None] = None,
) -> CARE2026Outputs:
    """Sliding-window CT inference.

    For nnUNet models (``CARE2026_CT_nnUNet``), delegates to the built-in
    ``nnUNetPredictor`` which handles all preprocessing, normalization,
    sliding window, and resampling.
    """
    # ── nnUNet path: use the built-in predictor ───────────────────────
    if hasattr(model, "predict") and hasattr(model, "_predictor"):
        nii = nib.load(str(img_path))
        image_raw = nii.get_fdata().astype(np.float32)
        zooms = tuple(nii.header.get_zooms()[:3])
        pred = model.predict(image_raw, zooms, use_tta=use_tta)
        # skip postprocess_ct_mask — nnUNet's Gaussian-weighted sliding
        # window produces clean predictions; connected-component filtering
        # would destroy multi-lobed structures (LAA, PV).
        return CARE2026Outputs(
            task="ct",
            ct_mask=pred,
            source_affine=nii.affine,
            source_header=nii.header,
        )

    if device is None:
        device = next(model.parameters()).device

    # Read inference parameters from model config (synced from train_config at init)
    mcfg = getattr(model, "config", {}) or {}
    # Support non-isotropic patch_shape (e.g. nnUNet [112,112,192])
    _ps = mcfg.get("patch_shape", mcfg.get("patch_size", CT_PATCH_SIZE))
    if isinstance(_ps, int):
        _ps = (_ps, _ps, _ps)
    if patch_size is None or patch_size == 128:
        patch_size = tuple(int(p) for p in _ps)
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)
    if stride is None:
        stride = tuple(max(1, p // 2) for p in patch_size)

    nii = nib.load(str(img_path))
    image_raw = nii.get_fdata().astype(np.float32)  # (H, W, D)
    orig_shape = image_raw.shape
    zooms = np.array(nii.header.get_zooms()[:3], dtype=np.float64)

    # Normalise according to the model config (synced from train_config at init)
    norm_cfg = mcfg.get("normalization", CFG(mode="minmax"))
    mode = str(norm_cfg.get("mode", "minmax"))
    if mode == "nnunet":
        image = np.clip(
            image_raw,
            float(norm_cfg["global_clip_min"]),
            float(norm_cfg["global_clip_max"]),
        )
        image = (image - float(norm_cfg["global_mean"])) / max(float(norm_cfg["global_std"]), 1e-8)
    elif mode == "percentile":
        p_low = float(norm_cfg.get("p_low", 0.5))
        p_high = float(norm_cfg.get("p_high", 99.5))
        v_low = float(np.percentile(image_raw, p_low))
        v_high = float(np.percentile(image_raw, p_high))
        if v_high > v_low:
            image = np.clip(image_raw, v_low, v_high)
            image = (image - v_low) / (v_high - v_low)
        else:
            image = np.zeros_like(image_raw, dtype=np.float32)
    elif mode == "zscore":
        mu, std = float(image_raw.mean()), float(image_raw.std())
        image = (image_raw - mu) / (std + 1e-8)
    else:
        # "minmax"
        hu_min = float(norm_cfg.get("hu_min", CT_HU_MIN))
        hu_max = float(norm_cfg.get("hu_max", CT_HU_MAX))
        image = np.clip(image_raw, hu_min, hu_max)
        image = (image - hu_min) / (hu_max - hu_min)

    # Resample to target spacing (from model config or default isotropic)
    target_spacing_cfg = mcfg.get("target_spacing", CT_TARGET_SPACING)
    target_spacing = np.array(target_spacing_cfg, dtype=np.float64)
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
