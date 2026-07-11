"""
Visualisation and evaluation utilities for the CARE 2026 challenge.

Provides standalone functions for viewing and debugging segmentation
results in Jupyter notebooks.  The slice-view helpers are also used
internally by the ``view_data`` methods of :class:`~data_reader.CARE2026_MRI`
and :class:`~data_reader.CARE2026_CT`.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from ipywidgets import Dropdown, IntSlider, interact

from const import MRI_CANONICAL_SHAPE, MRI_STAGE1_SHAPE
from data_reader import CARE2026_MRI
from predict import predict_mri_two_stage
from utils.mclahe import mclahe as _mclahe
from utils.viz_utils import _hatch_for, _is_notebook

__all__ = [
    "view_prediction",
    "evaluate_stage1",
    "evaluate_stage2",
    "evaluate_ct",
    "evaluate_training_sample",
    "debug_ct_data_flow",
]


def _is_nnunet_model(model: torch.nn.Module) -> bool:
    """Return True if the model wraps an nnUNetPredictor."""
    return hasattr(model, "_predictor") or getattr(getattr(model, "config", None), "backbone", None) == "nnunet"


def _safe_device(model: torch.nn.Module) -> torch.device:
    """Get device from model, falling back to CUDA / CPU."""
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_get_config(model: torch.nn.Module, key: str, default=None):
    """Read a config key from model.train_config or model.config."""
    for attr in ("train_config", "config"):
        cfg = getattr(model, attr, None)
        if cfg is not None and key in cfg:
            return cfg[key]
    return default


def _binary_dice_metric(pred: np.ndarray, target: np.ndarray) -> float:
    p, g = pred.astype(bool), target.astype(bool)
    inter = (p & g).sum()
    return float(2 * inter / (p.sum() + g.sum() + 1e-8))


def _multiclass_dice(mask_a: np.ndarray, mask_b: np.ndarray, cls_id: int) -> float:
    """Dice score for one foreground class."""
    a = mask_a == cls_id
    b = mask_b == cls_id
    return float(2 * (a & b).sum() / (a.sum() + b.sum() + 1e-8))


def _mask_centroid(mask: np.ndarray, cls_id: int) -> Optional[np.ndarray]:
    """Centroid of a class mask in voxel coordinates, or None if absent."""
    pts = np.argwhere(mask == cls_id)
    if len(pts) == 0:
        return None
    return pts.mean(axis=0)


def _contourf_with_hatch(ax, mask_slice, color, hatch, alpha=0.25):
    """Draw filled contour + opaque hatch overlay on top.

    Hatch lines must be rendered independently from the transparent fill,
    otherwise they inherit the fill's alpha and become invisible.
    Each class should use a distinct pattern via ``_hatch_for(cls_id)``.
    """
    ax.contourf(mask_slice, levels=[0.5, 1], colors=[color], alpha=alpha, antialiased=True)
    return ax.contourf(mask_slice, levels=[0.5, 1], colors="none", antialiased=True, hatches=[hatch])


def view_prediction(
    image: Union[str, Path, np.ndarray],
    prediction: Union[str, Path, np.ndarray],
    ground_truth: Optional[Union[str, Path, np.ndarray]] = None,
    *,
    overlay_mode: str = "filled+hatch",
    class_map: Optional[Dict[int, str]] = None,
    palette: Optional[Dict[int, str]] = None,
    slice_idx: Optional[int] = None,
) -> None:
    """Visualise prediction results alongside the original image (and optional GT).

    Designed for Jupyter notebooks — displays an interactive slider-based
    slice viewer.  Falls back to a static grid when run from a script.

    Parameters
    ----------
    image : path-like or (H, W, D) numpy array
        Original 3-D volume.
    prediction : path-like or (H, W, D) numpy array
        Predicted segmentation mask (binary or multi-class integer).
    ground_truth : path-like or (H, W, D) numpy array, optional
        Ground-truth segmentation mask for comparison.
    class_map : dict of ``int → str``, optional
        Class-id to class-name mapping (e.g. ``{0: "bg", 1: "LA"}``).
    palette : dict of ``int → colour``, optional
        Class-id to colour mapping.  Auto-generated if not provided.
    slice_idx : int, optional
        Initial slice to display (default: middle slice).

    Examples
    --------
    In a Jupyter notebook::

        from viz import view_prediction
        view_prediction("enhanced.nii.gz", "la_predict.nii.gz", "atriumSegImgMO.nii.gz")

    Or side-by-side without GT::

        view_prediction("enhanced.nii.gz", "la_predict.nii.gz")
    """

    # -- helpers ---------------------------------------------------------------
    def _load(path_or_arr, dtype=np.float32):
        if isinstance(path_or_arr, (str, Path)):
            return nib.load(str(path_or_arr)).get_fdata().astype(dtype)
        return path_or_arr.astype(dtype) if hasattr(path_or_arr, "astype") else path_or_arr

    def _load_mask(path_or_arr):
        if isinstance(path_or_arr, (str, Path)):
            return nib.load(str(path_or_arr)).get_fdata().astype(np.uint8)
        return path_or_arr.astype(np.uint8)

    # -- load data -------------------------------------------------------------
    img = _load(image)
    pred = _load_mask(prediction)
    gt = _load_mask(ground_truth) if ground_truth is not None else None

    # -- build palette and class map -------------------------------------------
    all_ids_set = set(np.unique(pred)) | (set(np.unique(gt)) if gt is not None else set())
    all_ids_set.discard(0)
    all_ids = sorted(all_ids_set)

    if class_map is None:
        class_map = {i: f"Class {i}" for i in all_ids}
    if palette is None:
        default_colors = ["#FF4444", "#4488FF", "#44FF44", "#FFAA00", "#FF44FF", "#00FFFF"]
        palette = {i: default_colors[(i - 1) % len(default_colors)] for i in all_ids}
        palette[0] = (0, 0, 0, 0)

    # -- build panel layout ----------------------------------------------------
    n_panels = 2 if gt is None else 3
    n_slices = img.shape[-1]
    mid = slice_idx if slice_idx is not None else n_slices // 2

    if _is_notebook():

        def _plot(sl: int = mid, overlay_mode: str = "filled+hatch"):
            fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
            if n_panels == 1:
                axes = [axes]

            is_filled = overlay_mode.startswith("filled")
            use_hatch = overlay_mode == "filled+hatch"

            axes[0].imshow(img[..., sl], cmap="gray", origin="lower")
            axes[0].set_title(f"Image  (slice {sl + 1}/{n_slices})")
            axes[0].axis("off")

            legend_handles = []
            for c in all_ids:
                if is_filled:
                    legend_handles.append(
                        mpatches.Patch(
                            facecolor=palette.get(c, "white"),
                            alpha=0.5,
                            edgecolor=palette.get(c, "white"),
                            label=class_map.get(c, f"Class {c}"),
                        )
                    )
                else:
                    legend_handles.append(mpatches.Patch(color=palette.get(c, "white"), label=class_map.get(c, f"Class {c}")))

            def _draw_mask(ax, mask_vol, cls_ids):
                for cls_id in cls_ids:
                    mask_slice = (mask_vol[..., sl] == cls_id).astype(np.uint8)
                    if mask_slice.max() == 0:
                        continue
                    color = palette.get(cls_id, "white")
                    if is_filled:
                        if use_hatch:
                            _contourf_with_hatch(ax, mask_slice, color, _hatch_for(cls_id))
                        else:
                            ax.contourf(
                                mask_slice,
                                levels=[0.5, 1],
                                colors=[color],
                                alpha=0.25,
                                antialiased=True,
                            )
                    else:
                        ax.contour(mask_slice, levels=[0.5], colors=[color], linewidths=1.5)

            if gt is not None:
                gt_idx = 1
                axes[gt_idx].imshow(img[..., sl], cmap="gray", origin="lower")
                _draw_mask(axes[gt_idx], gt, all_ids)
                axes[gt_idx].set_title("Ground Truth")
                axes[gt_idx].axis("off")
                if legend_handles:
                    axes[gt_idx].legend(handles=legend_handles, loc="upper right", framealpha=0.7, fontsize="x-small")
                pred_idx = 2
            else:
                pred_idx = 1

            axes[pred_idx].imshow(img[..., sl], cmap="gray", origin="lower")
            _draw_mask(axes[pred_idx], pred, all_ids)
            axes[pred_idx].set_title("Prediction")
            axes[pred_idx].axis("off")
            if legend_handles:
                axes[pred_idx].legend(handles=legend_handles, loc="upper right", framealpha=0.7, fontsize="x-small")

            fig.tight_layout()
            plt.show()

        interact(
            _plot,
            sl=IntSlider(min=0, max=n_slices - 1, step=1, value=mid, description="Slice"),
            overlay_mode=Dropdown(
                options=["contour", "filled", "filled+hatch"],
                value="filled+hatch",
                description="Overlay",
            ),
        )
    else:
        # Static fallback: show 6 evenly-spaced slices
        is_filled = overlay_mode.startswith("filled")
        use_hatch = overlay_mode == "filled+hatch"

        def _draw_static(ax, mask_vol):
            for cls_id in all_ids:
                mask_slice = (mask_vol[..., sl] == cls_id).astype(np.uint8)
                if mask_slice.max() == 0:
                    continue
                color = palette.get(cls_id, "white")
                if is_filled:
                    if use_hatch:
                        _contourf_with_hatch(ax, mask_slice, color, _hatch_for(cls_id))
                    else:
                        ax.contourf(
                            mask_slice,
                            levels=[0.5, 1],
                            colors=[color],
                            alpha=0.25,
                            antialiased=True,
                        )
                else:
                    ax.contour(mask_slice, levels=[0.5], colors=[color], linewidths=1)

        step = max(1, n_slices // 6)
        slices = list(range(0, n_slices, step))[:6]
        fig, axes = plt.subplots(n_panels, len(slices), figsize=(4 * len(slices), 4 * n_panels))
        if axes.ndim == 1:
            axes = axes[:, np.newaxis]

        for col, sl in enumerate(slices):
            axes[0, col].imshow(img[..., sl], cmap="gray", origin="lower")
            axes[0, col].set_title(f"Slice {sl}")
            axes[0, col].axis("off")

            if gt is not None:
                axes[1, col].imshow(img[..., sl], cmap="gray", origin="lower")
                _draw_static(axes[1, col], gt)
                axes[1, col].axis("off")
                pred_row = 2
            else:
                pred_row = 1

            axes[pred_row, col].imshow(img[..., sl], cmap="gray", origin="lower")
            _draw_static(axes[pred_row, col], pred)
            axes[pred_row, col].axis("off")
            axes[pred_row, col].axis("off")

        axes[0, 0].set_ylabel("Image", fontsize=12)
        if gt is not None:
            axes[1, 0].set_ylabel("GT", fontsize=12)
        axes[pred_row, 0].set_ylabel("Prediction", fontsize=12)

        # Shared legend
        legend_handles = [mpatches.Patch(color=palette.get(c, "white"), label=class_map.get(c, f"Class {c}")) for c in all_ids]
        if legend_handles:
            axes[-1, -1].legend(handles=legend_handles, loc="upper right", framealpha=0.7, fontsize="x-small")

        fig.tight_layout()
        plt.show()


def evaluate_stage1(
    rec: str,
    model: torch.nn.Module,
    db_dir: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate Stage 1 (coarse LA localisation) on a training sample.

    Runs the Stage-1 preprocessing pipeline (canonical → downsample →
    CLAHE if configured → z-score) and compares the coarse LA prediction
    with the GT LA mask downsampled to Stage-1 resolution (144×144×44).

    Parameters
    ----------
    rec : str
        Training record name, e.g. ``"train_1"``.
    model : torch.nn.Module
        CARE2026_MRI_Stage1_Model in eval mode.
    db_dir : path-like
        Root of the CARE2026 dataset.
    device : torch.device, optional

    Returns
    -------
    dict with ``la_dice`` (Stage-1 resolution).
    """

    if _is_nnunet_model(model):
        raise RuntimeError(
            "evaluate_stage1 does not support nnUNet models. "
            "Use evaluate_training_sample(rec, task=1, stage1_model=..., stage2_model=...) instead."
        )

    if device is None:
        device = _safe_device(model)

    db_dir = Path(db_dir).expanduser().resolve()
    has_scar = False
    reader = CARE2026_MRI(db_dir=db_dir, task=1, verbose=0)
    if rec in reader._all_records:
        has_scar = reader.get_scar_path(rec) is not None
    else:
        reader = CARE2026_MRI(db_dir=db_dir, task=2, verbose=0)

    apply_mclahe = bool(_safe_get_config(model, "apply_mclahe", False))

    # Preprocessing
    image = reader.load_data(rec)
    gt_la_native = reader.load_la_ann(rec)
    image_canonical = CARE2026_MRI.resample_data(image, MRI_CANONICAL_SHAPE)
    gt_la_canonical = CARE2026_MRI.resample_ann(gt_la_native, MRI_CANONICAL_SHAPE)

    if apply_mclahe:
        image_canonical = _mclahe(image_canonical)

    image_s1 = CARE2026_MRI.resample_data(image_canonical, MRI_STAGE1_SHAPE)
    gt_la_s1 = CARE2026_MRI.resample_ann(gt_la_canonical, MRI_STAGE1_SHAPE)

    # z-score
    mean, std = float(image_s1.mean()), float(image_s1.std())
    image_s1 = (image_s1 - mean) / (std + 1e-8)

    # Inference
    img_t = torch.from_numpy(image_s1).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_t)
    pred_s1 = out["la_mask"].squeeze().cpu().numpy().astype(np.uint8)

    la_dice = float(_binary_dice_metric(pred_s1, gt_la_s1))
    print(f"  Stage-1 LA Dice : {la_dice:.4f}  (Stage-1 shape {MRI_STAGE1_SHAPE})")
    print(f"  apply_mclahe    : {apply_mclahe}")

    # Interactive view
    n_slices = image_s1.shape[-1]
    mid = n_slices // 2
    PALETTE = {0: (0, 0, 0, 0), 1: "#00FFFF"}
    image_disp = image_s1  # already normalized

    def _plot(sl: int = mid, overlay_mode: str = "filled+hatch"):
        is_filled = overlay_mode.startswith("filled")
        use_hatch = overlay_mode == "filled+hatch"
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax in axes:
            ax.imshow(image_disp[..., sl], cmap="gray", origin="lower")

        def _draw(ax, mask, color):
            ms = mask[..., sl]
            if is_filled:
                if use_hatch:
                    _contourf_with_hatch(ax, ms, color, _hatch_for(1))
                else:
                    ax.contourf(ms, levels=[0.5, 1], colors=[color], alpha=0.25, antialiased=True)
            else:
                ax.contour(ms, levels=[0.5], colors=[color], linewidths=1.5)

        _draw(axes[1], gt_la_s1, PALETTE[1])
        axes[1].set_title("GT LA (Stage-1)")

        _draw(axes[2], pred_s1, PALETTE[1])
        axes[2].set_title("Pred LA (Stage-1)")

        axes[0].set_title(f"Stage-1 Image  (slice {sl + 1}/{n_slices})")
        for ax in axes:
            ax.axis("off")

        legend_h = [mpatches.Patch(facecolor=PALETTE[1], alpha=0.5, edgecolor=PALETTE[1], label="LA cavity")]
        axes[2].legend(handles=legend_h, loc="upper right", framealpha=0.7, fontsize="small")
        fig.tight_layout()
        plt.show()

    if _is_notebook():
        interact(
            _plot,
            sl=IntSlider(min=0, max=n_slices - 1, step=1, value=mid, description="Slice"),
            overlay_mode=Dropdown(options=["contour", "filled", "filled+hatch"], value="filled+hatch", description="Overlay"),
        )
    else:
        _plot()

    return {"la_dice": la_dice}


def evaluate_stage2(
    rec: str,
    stage1_model: torch.nn.Module,
    stage2_model: torch.nn.Module,
    db_dir: Union[str, Path],
    device: Optional[torch.device] = None,
    use_gt_centroid: bool = True,
) -> Dict[str, float]:
    """Evaluate Stage 2 (scar segmentation) on a training sample.

    Uses the unified :func:`predict_mri_two_stage` pipeline:
    Stage 1 provides the LA cavity mask for scar constraint; Stage 2
    predicts scar on a 128×128×44 patch at the centroid.  All
    post-processing (dilated-cavity scar constraint) is applied at
    native resolution.

    Parameters
    ----------
    rec : str, stage1_model, stage2_model, db_dir, device
    use_gt_centroid : bool, default True
        If True, use the GT LA centroid (bypasses Stage-1 localisation
        errors, isolating Stage-2 scar prediction quality).

    Returns
    -------
    dict with ``la_dice``, ``scar_dice``.
    """
    import nibabel as nib

    if device is None:
        device = _safe_device(stage1_model)

    db_dir = Path(db_dir).expanduser().resolve()
    reader = CARE2026_MRI(db_dir=db_dir, task=1, verbose=0)
    if rec not in reader._all_records:
        reader = CARE2026_MRI(db_dir=db_dir, task=2, verbose=0)
    has_scar = reader.get_scar_path(rec) is not None
    img_path = reader.get_data_path(rec)

    gt_la = reader.load_la_ann(rec)
    gt_scar = reader.load_scar_ann(rec) if has_scar else np.zeros_like(gt_la)

    # Compute GT centroid in canonical space
    centroid = None
    if use_gt_centroid:
        nii = nib.load(str(img_path))
        img_native = nii.get_fdata().astype(np.float32)
        gt_la_can = CARE2026_MRI.resample_ann(gt_la, MRI_CANONICAL_SHAPE)
        fg = np.argwhere(gt_la_can > 0)
        if len(fg) > 0:
            centroid = tuple(int(fg[:, i].mean()) for i in range(3))

    out = predict_mri_two_stage(img_path, stage1_model, stage2_model, device=device, centroid=centroid)
    pred_la = out.la_mask
    pred_scar = out.scar_mask

    la_dice = float(_binary_dice_metric(pred_la, gt_la))
    scar_dice = float(_binary_dice_metric(pred_scar, gt_scar)) if has_scar else float("nan")

    centroid_src = "GT" if centroid is not None else "Stage-1 predicted"
    apply_mclahe = bool(_safe_get_config(stage2_model, "apply_mclahe", False))
    print(f"  Centroid       : {centroid_src}")
    print(f"  apply_mclahe   : {apply_mclahe}")
    print(f"  LA Dice        : {la_dice:.4f}")
    if has_scar:
        print(f"  Scar Dice      : {scar_dice:.4f}  (GT: {gt_scar.sum()}, Pred: {pred_scar.sum()})")
    else:
        print("  (no scar GT for Task-2 records)")

    # Interactive view
    nii = nib.load(str(img_path))
    img_native = nii.get_fdata().astype(np.float32)
    n_slices_native = img_native.shape[-1]
    mid_n = n_slices_native // 2
    PALETTE = {0: (0, 0, 0, 0), 1: "#00FFFF", 2: "#FF4444"}

    def _plot(sl: int = mid_n, overlay_mode: str = "filled+hatch"):
        is_filled = overlay_mode.startswith("filled")
        use_hatch = overlay_mode == "filled+hatch"
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax in axes:
            ax.imshow(img_native[..., sl], cmap="gray", origin="lower")

        def _draw(ax, la, scar):
            for cls_id, mask, c in [(1, la, PALETTE[1]), (2, scar, PALETTE[2])]:
                if mask.max() == 0:
                    continue
                ms = mask[..., sl]
                if is_filled:
                    if use_hatch:
                        _contourf_with_hatch(ax, ms, c, _hatch_for(cls_id))
                    else:
                        ax.contourf(ms, levels=[0.5, 1], colors=[c], alpha=0.25, antialiased=True)
                else:
                    ax.contour(ms, levels=[0.5], colors=[c], linewidths=1.5)

        axes[0].set_title(f"Native Image  (slice {sl + 1}/{n_slices_native})")
        _draw(axes[1], gt_la, gt_scar)
        axes[1].set_title("Ground Truth")
        _draw(axes[2], pred_la, pred_scar)
        axes[2].set_title(f"Prediction ({centroid_src} centroid)")
        for ax in axes:
            ax.axis("off")
        legend_h = []
        for cls_id, c, label in [(1, PALETTE[1], "LA cavity"), (2, PALETTE[2], "LA scar")]:
            legend_h.append(
                mpatches.Patch(facecolor=c, alpha=0.5, edgecolor=c, label=label)
                if is_filled
                else mpatches.Patch(color=c, label=label)
            )
        axes[2].legend(handles=legend_h, loc="upper right", framealpha=0.7, fontsize="small")
        fig.tight_layout()
        plt.show()

    if _is_notebook():
        interact(
            _plot,
            sl=IntSlider(min=0, max=n_slices_native - 1, step=1, value=mid_n, description="Slice"),
            overlay_mode=Dropdown(options=["contour", "filled", "filled+hatch"], value="filled+hatch", description="Overlay"),
        )
    else:
        _plot()

    return {"la_dice": la_dice, "scar_dice": scar_dice}


def evaluate_ct(
    rec: str,
    model: torch.nn.Module,
    db_dir: Union[str, Path],
    device: Optional[torch.device] = None,
    use_tta: bool = False,
) -> Dict[str, float]:
    """Evaluate CT model (Task 3) on a training sample with GT labels.

    Displays a 3-panel interactive view (Image | GT | Prediction) and prints
    per-class Dice scores.

    Parameters
    ----------
    rec : str
        Training record name, e.g. ``"train_1"`` (must have a GT label).
    model : torch.nn.Module
        CARE2026_CT_Model or CARE2026_CT_nnUNet in eval mode.
    db_dir : path-like
        Root of the CARE2026 dataset.
    device : torch.device, optional
        Ignored for nnUNet models (device is handled internally).
    use_tta : bool, default False

    Returns
    -------
    dict with ``ct_dice_la``, ``ct_dice_pv``, ``ct_dice_laa``, ``ct_mean_dice``.
    """
    from data_reader import CARE2026_CT
    from predict import predict_ct

    if device is None and not _is_nnunet_model(model):
        device = _safe_device(model)
    db_dir = Path(db_dir).expanduser().resolve()

    reader = CARE2026_CT(db_dir=db_dir, verbose=0)
    img_path = reader.get_data_path(rec)
    gt = reader.load_ann(rec)

    out = predict_ct(img_path, model, device=device, use_tta=use_tta)
    pred = out.ct_mask

    def _dice_per_class(mask, cls_id):
        p = (mask == cls_id).astype(bool)
        g = (gt == cls_id).astype(bool)
        inter = (p & g).sum()
        return float(2 * inter / (p.sum() + g.sum() + 1e-8))

    dice_la = _dice_per_class(pred, 1)
    dice_pv = _dice_per_class(pred, 2)
    dice_laa = _dice_per_class(pred, 3)
    print(f"  LA  Dice: {dice_la:.4f}")
    print(f"  PV  Dice: {dice_pv:.4f}")
    print(f"  LAA Dice: {dice_laa:.4f}")
    print(f"  Mean Dice: {(dice_la + dice_pv + dice_laa) / 3:.4f}")

    # Interactive view
    img_native = nib.load(str(img_path)).get_fdata().astype(np.float32)
    n_slices = img_native.shape[-1]
    mid = n_slices // 2
    PALETTE = {1: "#FF4444", 2: "#4488FF", 3: "#44FF44"}
    CLASS_NAMES = {1: "LA", 2: "PV", 3: "LAA"}

    def _plot(sl: int = mid, overlay_mode: str = "filled+hatch"):
        is_filled = overlay_mode.startswith("filled")
        use_hatch = overlay_mode == "filled+hatch"
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax in axes:
            ax.imshow(img_native[..., sl], cmap="gray", origin="lower")

        def _draw(ax, mask):
            for cls_id, c in PALETTE.items():
                ms = (mask[..., sl] == cls_id).astype(np.uint8)
                if ms.max() == 0:
                    continue
                if is_filled:
                    if use_hatch:
                        _contourf_with_hatch(ax, ms, c, _hatch_for(cls_id))
                    else:
                        ax.contourf(ms, levels=[0.5, 1], colors=[c], alpha=0.25, antialiased=True)
                else:
                    ax.contour(ms, levels=[0.5], colors=[c], linewidths=1.5)

        axes[0].set_title(f"CT Image  (slice {sl + 1}/{n_slices})")
        _draw(axes[1], gt)
        axes[1].set_title("Ground Truth")
        _draw(axes[2], pred)
        axes[2].set_title("Prediction")
        for ax in axes:
            ax.axis("off")
        legend_h = [
            mpatches.Patch(facecolor=c, alpha=0.5, edgecolor=c, label=l) if is_filled else mpatches.Patch(color=c, label=l)
            for c, l in zip(PALETTE.values(), CLASS_NAMES.values())
        ]
        axes[2].legend(handles=legend_h, loc="upper right", framealpha=0.7, fontsize="small")
        fig.tight_layout()
        plt.show()

    if _is_notebook():
        interact(
            _plot,
            sl=IntSlider(min=0, max=n_slices - 1, step=1, value=mid, description="Slice"),
            overlay_mode=Dropdown(options=["contour", "filled", "filled+hatch"], value="filled+hatch", description="Overlay"),
        )
    else:
        _plot()

    return {
        "ct_dice_la": dice_la,
        "ct_dice_pv": dice_pv,
        "ct_dice_laa": dice_laa,
        "ct_mean_dice": (dice_la + dice_pv + dice_laa) / 3,
    }


def debug_ct_data_flow(
    rec: str = "train_1",
    db_dir: Union[str, Path] = None,
    *,
    pred_path: Optional[Union[str, Path]] = None,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    use_tta: bool = False,
    dataset_training: bool = False,
    dataset_index: Optional[int] = None,
) -> Dict[str, object]:
    """Inspect CT preprocessing, dataset output, and optional prediction alignment.

    This is intended for debugging apparent spatial shifts.  It shows:

    - Native CT image + native GT label.
    - Label after train-time resampling to 0.5 mm and nearest-neighbour
      roundtrip back to native shape.
    - The exact preprocessed full-volume arrays produced by
      :class:`CARE2026_CT_Dataset._preprocess_ct`.
    - A dataset ``__getitem__`` patch.
    - Optional prediction from ``pred_path`` or live ``model`` inference.

    Notebook usage::

        from viz import debug_ct_data_flow
        debug_ct_data_flow(
            rec="train_1",
            db_dir="/Data1/wenh06/CARE2026-LeftAtrium",
            pred_path="/path/to/val_or_train_pred.nii.gz",
        )

    Returns a dict of loaded arrays so the same data can be inspected manually.
    """
    if db_dir is None:
        raise ValueError("db_dir is required")
    from cfg import CT_TrainCfg
    from const import CT_NUM_CLASSES, CT_TARGET_SPACING
    from data_reader import CARE2026_CT
    from dataset import CARE2026_CT_Dataset
    from predict import _resample_mask, postprocess_ct_mask, predict_ct

    db_dir = Path(db_dir).expanduser().resolve()
    reader = CARE2026_CT(db_dir=db_dir, verbose=0)
    img_path = reader.get_data_path(rec)
    ann_path = reader.get_ann_path(rec)
    if ann_path is None:
        raise FileNotFoundError(f"{rec} has no CT ground-truth label.")

    img_nii = nib.load(str(img_path))
    gt_nii = nib.load(str(ann_path))
    img_native = img_nii.get_fdata().astype(np.float32)
    gt_native = gt_nii.get_fdata().astype(np.uint8)

    cfg = deepcopy(CT_TrainCfg)
    cfg.augmentation = None
    dataset = CARE2026_CT_Dataset(db_dir=db_dir, config=cfg, training=dataset_training, labeled=True)
    image_iso, gt_iso = dataset._preprocess_ct(rec)
    if gt_iso is None:
        raise RuntimeError(f"Dataset preprocessing returned no mask for {rec}.")
    gt_roundtrip = _resample_mask(gt_iso, gt_native.shape)

    if dataset_index is None:
        try:
            dataset_index = dataset._records.index(rec)
        except ValueError:
            dataset_index = 0
    sample = dataset[dataset_index]
    patch_img = sample["image"][0]
    patch_mask = sample.get("mask")

    pred = None
    pred_pp = None
    if pred_path is not None:
        pred = nib.load(str(pred_path)).get_fdata().astype(np.uint8)
        pred_pp = postprocess_ct_mask(pred, CT_NUM_CLASSES)
    elif model is not None:
        if device is None:
            device = next(model.parameters()).device
        out = predict_ct(img_path, model, device=device, use_tta=use_tta)
        pred = out.ct_mask
        pred_pp = pred

    print(f"Record: {rec}")
    print(f"Image path: {img_path}")
    print(f"Label path: {ann_path}")
    print(f"Native image shape / zooms: {img_native.shape} / {img_nii.header.get_zooms()[:3]}")
    print(f"Native label shape / zooms: {gt_native.shape} / {gt_nii.header.get_zooms()[:3]}")
    print(f"Target spacing: {CT_TARGET_SPACING}; preprocessed shape: {image_iso.shape}")
    print(f"Image axis codes: {nib.aff2axcodes(img_nii.affine)}")
    print(f"Label axis codes: {nib.aff2axcodes(gt_nii.affine)}")
    print(f"Max |image affine - label affine|: {float(np.max(np.abs(img_nii.affine - gt_nii.affine))):.6g}")
    print("Roundtrip label Dice (native GT vs resample-to-0.5mm-back):")
    for cls_id, name in [(1, "LA"), (2, "PV"), (3, "LAA")]:
        d = _multiclass_dice(gt_roundtrip, gt_native, cls_id)
        c_gt = _mask_centroid(gt_native, cls_id)
        c_rt = _mask_centroid(gt_roundtrip, cls_id)
        shift = None if c_gt is None or c_rt is None else c_rt - c_gt
        shift_s = "NA" if shift is None else np.array2string(shift, precision=2)
        print(f"  {name}: dice={d:.4f}, centroid_shift_vox={shift_s}")
    if pred is not None:
        print("Prediction Dice / centroid shift vs native GT:")
        for cls_id, name in [(1, "LA"), (2, "PV"), (3, "LAA")]:
            d = _multiclass_dice(pred, gt_native, cls_id)
            c_gt = _mask_centroid(gt_native, cls_id)
            c_pr = _mask_centroid(pred, cls_id)
            shift = None if c_gt is None or c_pr is None else c_pr - c_gt
            shift_s = "NA" if shift is None else np.array2string(shift, precision=2)
            print(f"  {name}: dice={d:.4f}, centroid_shift_vox={shift_s}")

    palette = {1: "#FF4444", 2: "#4488FF", 3: "#44FF44"}
    class_names = {1: "LA", 2: "PV", 3: "LAA"}

    views = {
        "native_gt": (img_native, gt_native, "Native image + native GT"),
        "roundtrip_gt": (img_native, gt_roundtrip, "Native image + GT after 0.5mm roundtrip"),
        "preprocessed_gt": (image_iso, gt_iso, "Preprocessed 0.5mm image + GT"),
    }
    if patch_mask is not None:
        views["dataset_patch"] = (patch_img, patch_mask.astype(np.uint8), f"Dataset __getitem__ patch: {sample['record']}")
    if pred is not None:
        views["prediction"] = (img_native, pred, "Native image + prediction")
    if pred_pp is not None and pred_path is not None:
        views["prediction_postprocessed"] = (img_native, pred_pp, "Native image + prediction after CT postprocess")

    def _plot(view: str = "native_gt", cls: str = "all", sl: Optional[int] = None, overlay_mode: str = "filled+hatch"):
        img, mask, title = views[view]
        if sl is None:
            sl = img.shape[-1] // 2
        sl = int(np.clip(sl, 0, img.shape[-1] - 1))
        is_filled = overlay_mode.startswith("filled")
        use_hatch = overlay_mode == "filled+hatch"
        cls_ids = [1, 2, 3] if cls == "all" else [int(cls)]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img[..., sl], cmap="gray", origin="lower")
        handles = []
        for cls_id in cls_ids:
            ms = (mask[..., sl] == cls_id).astype(np.uint8)
            if ms.max() == 0:
                continue
            color = palette[cls_id]
            if is_filled:
                if use_hatch:
                    _contourf_with_hatch(ax, ms, color, _hatch_for(cls_id))
                else:
                    ax.contourf(ms, levels=[0.5, 1], colors=[color], alpha=0.25, antialiased=True)
                handles.append(mpatches.Patch(facecolor=color, alpha=0.5, edgecolor=color, label=class_names[cls_id]))
            else:
                ax.contour(ms, levels=[0.5], colors=[color], linewidths=1.5)
                handles.append(mpatches.Patch(color=color, label=class_names[cls_id]))
        ax.set_title(f"{title}  (slice {sl + 1}/{img.shape[-1]})")
        ax.axis("off")
        if handles:
            ax.legend(handles=handles, loc="upper right", framealpha=0.7, fontsize="small")
        fig.tight_layout()
        plt.show()

    if _is_notebook():
        max_slices = max(arr[0].shape[-1] for arr in views.values())
        interact(
            _plot,
            view=Dropdown(options=[(v[2], k) for k, v in views.items()], value="native_gt", description="View"),
            cls=Dropdown(options=[("All", "all"), ("LA", "1"), ("PV", "2"), ("LAA", "3")], value="all", description="Class"),
            sl=IntSlider(min=0, max=max_slices - 1, step=1, value=img_native.shape[-1] // 2, description="Slice"),
            overlay_mode=Dropdown(options=["contour", "filled", "filled+hatch"], value="filled+hatch", description="Overlay"),
        )
    else:
        _plot()

    return {
        "image_native": img_native,
        "label_native": gt_native,
        "image_iso": image_iso,
        "label_iso": gt_iso,
        "label_roundtrip": gt_roundtrip,
        "dataset_sample": sample,
        "prediction": pred,
        "prediction_postprocessed": pred_pp,
    }


def evaluate_training_sample(
    rec: str,
    task: int,
    stage1_model: Optional[torch.nn.Module] = None,
    stage2_model: Optional[torch.nn.Module] = None,
    ct_model: Optional[torch.nn.Module] = None,
    db_dir: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    use_tta: bool = False,
) -> Dict[str, float]:
    """Run inference on a *training* sample and compare with GT labels.

    Displays a 3-panel interactive view (Image | GT | Prediction) and prints
    Dice scores.

    Parameters
    ----------
    rec : str
        Record name, e.g. ``"train_1"``.
    task : int
        Task number: 1 (LA scar), 2 (LA cavity), 3 (CT multi-structure).
    stage1_model, stage2_model : nn.Module or None
        MRI Stage-1 / Stage-2 models (required for task=1 or 2).
    ct_model : nn.Module or None
        CT model (required for task=3).
    db_dir : path-like
        Root of the CARE2026 dataset.
    device : torch.device, optional
    use_tta : bool, default False

    Returns
    -------
    dict with per-class Dice scores.
    """
    assert db_dir is not None, "db_dir is required"
    import nibabel as nib

    db_dir = Path(db_dir).expanduser().resolve()
    is_mri = task in (1, 2)

    if is_mri:
        if stage1_model is None or stage2_model is None:
            raise ValueError("stage1_model and stage2_model required for MRI evaluation.")
        if device is None:
            device = _safe_device(stage1_model)

        reader = CARE2026_MRI(db_dir=db_dir, task=task, verbose=0)
        has_scar = reader.get_scar_path(rec) is not None
        img_path = reader.get_data_path(rec)
        gt_la = reader.load_la_ann(rec)
        gt_scar = reader.load_scar_ann(rec) if has_scar else np.zeros_like(gt_la)

        out = predict_mri_two_stage(img_path, stage1_model, stage2_model, device=device, use_tta=use_tta)
        pred_a = out.la_mask
        pred_b = out.scar_mask
        dice_a = float(_binary_dice_metric(pred_a, gt_la))
        dice_b = float(_binary_dice_metric(pred_b, gt_scar)) if has_scar else float("nan")

        print(f"  LA Dice : {dice_a:.4f}")
        if has_scar:
            print(f"  Scar Dice: {dice_b:.4f}  (GT voxels: {gt_scar.sum()}, Pred voxels: {pred_b.sum()})")
        else:
            print("  (no scar GT for Task-2 records)")

        PALETTE = [(1, "#00FFFF", "LA cavity"), (2, "#FF4444", "LA scar")]
        gt_masks = [(gt_la, 1), (gt_scar, 2)]
        pred_masks = [(pred_a, 1), (pred_b, 2)]
    else:
        # CT (Task 3)
        from data_reader import CARE2026_CT

        if ct_model is None:
            raise ValueError("ct_model required for CT evaluation.")
        if device is None:
            device = _safe_device(ct_model)
        ct_reader = CARE2026_CT(db_dir=db_dir, verbose=0)
        img_path = ct_reader.get_data_path(rec)
        gt = ct_reader.load_ann(rec)

        from predict import predict_ct

        out = predict_ct(img_path, ct_model, device=device, use_tta=use_tta)
        pred = out.ct_mask
        pred_a = pred
        pred_b = None

        def _dice_cls(mask, cls_id):
            p = mask == cls_id
            g = gt == cls_id
            return float(2 * (p & g).sum() / (p.sum() + g.sum() + 1e-8))

        d_la = _dice_cls(pred, 1)
        d_pv = _dice_cls(pred, 2)
        d_laa = _dice_cls(pred, 3)
        print(f"  LA  Dice: {d_la:.4f}")
        print(f"  PV  Dice: {d_pv:.4f}")
        print(f"  LAA Dice: {d_laa:.4f}")
        print(f"  Mean Dice: {(d_la + d_pv + d_laa) / 3:.4f}")

        PALETTE = [(1, "#FF4444", "LA"), (2, "#4488FF", "PV"), (3, "#44FF44", "LAA")]
        gt_masks = [(gt, c) for c, _, _ in PALETTE]
        pred_masks = [(pred, c) for c, _, _ in PALETTE]
        has_scar = False
        dice_a = dice_b = None  # computed above, not returned the same way

    # Interactive view
    img_native = nib.load(str(img_path)).get_fdata().astype(np.float32)
    n_slices = img_native.shape[-1]
    mid = n_slices // 2

    def _plot(sl: int = mid, overlay_mode: str = "filled+hatch"):
        is_filled = overlay_mode.startswith("filled")
        use_hatch = overlay_mode == "filled+hatch"
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax in axes:
            ax.imshow(img_native[..., sl], cmap="gray", origin="lower")

        def _draw(ax, masks):
            for mask, cls_id in masks:
                if mask.max() == 0:
                    continue
                c = next(col for cid, col, _ in PALETTE if cid == cls_id)
                ms = (mask[..., sl] == cls_id).astype(np.uint8) if not is_mri else mask[..., sl]
                if is_filled:
                    if use_hatch:
                        _contourf_with_hatch(ax, ms, c, _hatch_for(cls_id))
                    else:
                        ax.contourf(ms, levels=[0.5, 1], colors=[c], alpha=0.25, antialiased=True)
                else:
                    ax.contour(ms, levels=[0.5], colors=[c], linewidths=1.5)

        axes[0].set_title(f"Image  (slice {sl + 1}/{n_slices})")
        _draw(axes[1], gt_masks)
        axes[1].set_title("Ground Truth")
        _draw(axes[2], pred_masks)
        axes[2].set_title("Prediction")
        for ax in axes:
            ax.axis("off")
        legend_h = [
            mpatches.Patch(facecolor=c, alpha=0.5, edgecolor=c, label=l) if is_filled else mpatches.Patch(color=c, label=l)
            for _, c, l in PALETTE
        ]
        axes[2].legend(handles=legend_h, loc="upper right", framealpha=0.7, fontsize="small")
        fig.tight_layout()
        plt.show()

    if _is_notebook():
        interact(
            _plot,
            sl=IntSlider(min=0, max=n_slices - 1, step=1, value=mid, description="Slice"),
            overlay_mode=Dropdown(options=["contour", "filled", "filled+hatch"], value="filled+hatch", description="Overlay"),
        )
    else:
        _plot()

    if is_mri:
        return {"la_dice": dice_a, "scar_dice": dice_b}
    return {"ct_dice_la": d_la, "ct_dice_pv": d_pv, "ct_dice_laa": d_laa, "ct_mean_dice": (d_la + d_pv + d_laa) / 3}
