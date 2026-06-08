"""Low-level visualisation helpers shared by data_reader and viz."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython
from IPython.display import display
from ipywidgets import Checkbox, Dropdown, HBox, IntSlider, Output, VBox, interactive_output
from matplotlib.colors import ListedColormap

__all__ = [
    "_is_notebook",
    "_build_seg_cmap",
    "_slice_view_interactive",
    "_slice_view_static",
]

# Distinct hatch patterns per class ID (cycled)
_HATCH_POOL = ["//", "\\\\", "..", "xx", "oo", "**"]


def _hatch_for(cls_id: int) -> str:
    return _HATCH_POOL[(cls_id - 1) % len(_HATCH_POOL)]


def _is_notebook() -> bool:
    """Return True when running inside a Jupyter notebook / IPython kernel."""
    try:
        if get_ipython() is not None:
            return True
    except Exception:
        pass
    return False


def _build_seg_cmap(palette: Dict[int, str], n_classes: int) -> ListedColormap:
    """Build a discrete colormap from a class-id->colour palette."""
    colors = [palette.get(i, (0, 0, 0, 0)) for i in range(n_classes)]
    return ListedColormap(colors)


def _slice_view_interactive(
    image: np.ndarray,
    masks: Optional[Dict[int, np.ndarray]] = None,
    palette: Optional[Dict[int, str]] = None,
    class_names: Optional[Dict[int, str]] = None,
    title: str = "",
    figsize: Tuple[int, int] = (8, 8),
) -> None:
    """Interactive single-panel slice viewer with checkboxes, overlay mode,
    and legend.

    In Jupyter notebooks, displays:
    - An integer slider to scrub through z-slices.
    - One checkbox per label class to toggle overlay on/off.
    - A dropdown to select overlay style: contour, filled (transparent),
      or filled+hatch (transparent + diagonal line texture).
    - A colour legend.

    Parameters
    ----------
    image : (H, W, D) float32 or uint8 array
    masks : dict of ``class_id → (H, W, D) uint8 array``, optional
    palette : dict of ``class_id → colour``, optional
    class_names : dict of ``class_id → str``, optional
        Human-readable names for the legend and checkbox labels.
    title : str
    figsize : (int, int)
    """

    if palette is None:
        palette = {}
    if class_names is None:
        class_names = {}
    if masks is None:
        masks = {}

    n_slices = image.shape[-1]
    mid = n_slices // 2
    mask_ids = sorted(masks.keys())  # stable checkbox / legend order

    # -- widgets ---------------------------------------------------------------
    slider = IntSlider(min=0, max=n_slices - 1, step=1, value=mid, description="Slice")
    overlay_dd = Dropdown(
        options=["contour", "filled", "filled+hatch"],
        value="filled+hatch",
        description="Overlay:",
    )
    show_cbs: Dict[int, Checkbox] = {}
    for cls_id in mask_ids:
        label = class_names.get(cls_id, f"Class {cls_id}")
        show_cbs[cls_id] = Checkbox(value=True, description=label, indent=False)

    out = Output()

    # -- plot function ---------------------------------------------------------
    def _plot(slice_idx: int, overlay_mode: str, **show: bool) -> None:
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(image[..., slice_idx], cmap="gray", origin="lower")

            is_filled = overlay_mode.startswith("filled")
            use_hatch = overlay_mode == "filled+hatch"

            legend_handles = []
            for cls_id in mask_ids:
                if not show.get(str(cls_id), True):
                    continue
                mask = masks[cls_id]
                if mask.max() == 0:
                    continue
                mask_slice = mask[..., slice_idx]
                color = palette.get(cls_id, "white")

                if is_filled:
                    # Filled overlay: single contourf with optional hatch pattern.
                    # (Two separate contourf calls would overwrite each other.)
                    ax.contourf(
                        mask_slice,
                        levels=[0.5, 1],
                        colors=[color],
                        alpha=0.25,
                        antialiased=True,
                        hatches=[_hatch_for(cls_id)] if use_hatch else [],
                    )
                    legend_handle = mpatches.Patch(
                        facecolor=color,
                        alpha=0.5,
                        edgecolor=color,
                        label=class_names.get(cls_id, f"Class {cls_id}"),
                    )
                else:
                    # Contour-only mode
                    ax.contour(mask_slice, levels=[0.5], colors=[color], linewidths=1.5)
                    legend_handle = mpatches.Patch(
                        color=color,
                        label=class_names.get(cls_id, f"Class {cls_id}"),
                    )

                legend_handles.append(legend_handle)

            if legend_handles:
                ax.legend(handles=legend_handles, loc="upper right", framealpha=0.7, fontsize="small")

            ax.set_title(f"{title}  (slice {slice_idx + 1}/{n_slices})")
            ax.axis("off")
            fig.tight_layout()
            plt.show()

    # -- wire widgets ----------------------------------------------------------
    controls: Dict = {"slice_idx": slider, "overlay_mode": overlay_dd}
    controls.update({str(cls_id): cb for cls_id, cb in show_cbs.items()})

    checkbox_row = HBox(list(show_cbs.values()))
    ui = VBox([slider, overlay_dd, checkbox_row, out])
    display(ui)

    # Hold a reference so the widget isn't garbage-collected
    _plot._widget = interactive_output(_plot, controls)


def _slice_view_static(
    image: np.ndarray,
    masks: Optional[Dict[int, np.ndarray]] = None,
    palette: Optional[Dict[int, str]] = None,
    class_names: Optional[Dict[int, str]] = None,
    channels: Optional[List[int]] = None,
    title: str = "",
    overlay_mode: str = "contour",
    max_cols: int = 4,
) -> None:
    """Static multi-slice grid view (fallback when not in a notebook).

    Parameters
    ----------
    overlay_mode : str, default "contour"
        ``"contour"``, ``"filled"``, or ``"filled+hatch"``.
    """

    n_slices = image.shape[-1]
    if channels is None:
        channels = list(range(n_slices))
    if palette is None:
        palette = {}
    if class_names is None:
        class_names = {}
    if masks is None:
        masks = {}

    mask_ids = sorted(masks.keys())
    is_filled = overlay_mode.startswith("filled")
    use_hatch = overlay_mode == "filled+hatch"

    n = len(channels)
    n_rows = int(np.ceil(n / max_cols))
    n_cols = min(max_cols, n)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes_flat = np.array(axes).ravel() if n > 1 else [axes]
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    for ax_idx, sl in enumerate(channels):
        ax = axes_flat[ax_idx]
        ax.set_axis_off()
        ax.imshow(image[..., sl], cmap="gray", origin="lower")
        ax.set_title(f"Slice {sl}")
        for cls_id in mask_ids:
            mask = masks[cls_id]
            if mask.max() == 0:
                continue
            mask_slice = mask[..., sl]
            color = palette.get(cls_id, "white")
            if is_filled:
                ax.contourf(
                    mask_slice,
                    levels=[0.5, 1],
                    colors=[color],
                    alpha=0.25,
                    antialiased=True,
                    hatches=[_hatch_for(cls_id)] if use_hatch else [],
                )
            else:
                ax.contour(mask_slice, levels=[0.5], colors=[color], linewidths=1)

    # Shared legend on the last visible axis
    legend_handles = []
    for cls_id in mask_ids:
        if masks[cls_id].max() > 0:
            color = palette.get(cls_id, "white")
            legend_handles.append(
                mpatches.Patch(
                    facecolor=color if is_filled else "none",
                    alpha=0.5 if is_filled else 1.0,
                    edgecolor=color,
                    label=class_names.get(cls_id, f"Class {cls_id}"),
                )
            )
    if legend_handles:
        axes_flat[min(n - 1, len(axes_flat) - 1)].legend(
            handles=legend_handles, loc="upper right", framealpha=0.7, fontsize="small"
        )

    for ax_idx in range(n, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)
    fig.suptitle(title)
    plt.show()
