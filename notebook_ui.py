"""
Interactive inference dashboard for the CARE 2026 challenge.

Jupyter notebook widget panel for loading models, browsing runs,
adjusting thresholds with live preview, and packaging submissions.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from IPython.display import display
from ipywidgets import HTML, Button, Dropdown, FloatSlider, HBox, IntSlider, Output, VBox

from outputs import package_submission
from predict import predict_ct, predict_mri_two_stage
from utils.viz_utils import _is_notebook

__all__ = ["InferencePanel"]


# Shared palette
_PALETTE = {"la": "#00FFFF", "scar": "#FF4444", "laa": "#44FF44", "pv": "#4488FF"}


class InferencePanel:
    """Widget-based inference dashboard.

    Notebook usage::

        panel = InferencePanel(
            checkpoints_dir="checkpoints/",
            val_data_root="/path/to/CARE2026-LeftAtrium",
            output_root="/path/to/output",
        )
        panel.show()
    """

    def __init__(
        self,
        checkpoints_dir: str,
        val_data_root: str,
        output_root: str,
    ) -> None:
        if not _is_notebook():
            raise RuntimeError("InferencePanel only works inside a Jupyter notebook.")

        self.ckpt_dir = Path(checkpoints_dir).expanduser().resolve()
        self.val_root = Path(val_data_root).expanduser().resolve()
        self.out_root = Path(output_root).expanduser().resolve()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._s1_model: Optional[torch.nn.Module] = None
        self._s2_model: Optional[torch.nn.Module] = None
        self._ct_model: Optional[torch.nn.Module] = None
        self._live_output: Optional[Dict[str, np.ndarray]] = None

        self._build()
        self._scan_checkpoints()
        self._refresh_runs()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        # -- Controls row 1: model checkpoints + load ---------------------------
        self._dd_s1 = Dropdown(options=[], description="S1:", layout={"width": "220px"})
        self._dd_s2 = Dropdown(options=[], description="S2:", layout={"width": "220px"})
        self._dd_ct = Dropdown(options=[], description="CT:", layout={"width": "220px"})
        self._btn_load = Button(description="Load", button_style="primary")
        self._btn_load.on_click(self._on_load_models)
        self._btn_refresh = Button(description="↻", layout={"width": "40px"})
        self._btn_refresh.on_click(self._on_refresh)

        row1 = HBox([self._dd_s1, self._dd_s2, self._dd_ct, self._btn_load, self._btn_refresh])

        # -- Controls row 2: run selection -------------------------------------
        self._dd_run = Dropdown(options=[], description="Run:", layout={"width": "300px"})
        self._txt_new = Dropdown(options=["(auto)", "(latest)"], value="(auto)", description="New:", layout={"width": "300px"})
        self._btn_save = Button(description="Save", button_style="success")
        self._btn_save.on_click(self._on_save)
        self._btn_pkg = Button(description="Package", button_style="warning")
        self._btn_pkg.on_click(self._on_package)
        row2 = HBox([self._dd_run, self._txt_new, self._btn_save, self._btn_pkg])

        # -- Controls row 3: task / sample -------------------------------------
        self._dd_task = Dropdown(
            options=[("Task 1 — scar", 1), ("Task 2 — cavity", 2), ("Task 3 — CT", 3)], description="Task:", value=1
        )
        self._dd_task.observe(self._on_task_change, names="value")
        self._dd_sample = Dropdown(options=[], description="Sample:", layout={"width": "180px"})
        self._dd_sample.observe(self._on_sample_change, names="value")
        self._btn_preview = Button(description="Preview", button_style="info")
        self._btn_preview.on_click(self._on_preview)
        row3 = HBox([self._dd_task, self._dd_sample, self._btn_preview])

        # -- Controls row 4: threshold sliders ---------------------------------
        self._sl_s1 = FloatSlider(
            min=0.01, max=0.99, step=0.01, value=0.5, description="S1 LA:", continuous_update=False, layout={"width": "280px"}
        )
        self._sl_s2 = FloatSlider(
            min=0.01, max=0.99, step=0.01, value=0.5, description="S2 Scar:", continuous_update=False, layout={"width": "280px"}
        )
        self._sl_ct = FloatSlider(
            min=0.01, max=0.99, step=0.01, value=0.5, description="CT:", continuous_update=False, layout={"width": "280px"}
        )
        for sl in (self._sl_s1, self._sl_s2, self._sl_ct):
            sl.observe(self._on_threshold_change, names="value")
        row4 = HBox([self._sl_s1, self._sl_s2, self._sl_ct])

        # -- Status label ------------------------------------------------------
        self._lbl_status = HTML(value="")

        # -- Two-panel output --------------------------------------------------
        self._out_live = Output()
        self._out_saved = Output()

        controls = VBox(
            [
                HTML("<b>Models</b>"),
                row1,
                HTML("<b>Run</b>"),
                row2,
                row3,
                row4,
                self._lbl_status,
            ]
        )
        panels = HBox([self._out_live, self._out_saved], layout={"border": "1px solid #ccc"})

        self._ui = VBox([controls, panels])

    # ------------------------------------------------------------------
    # Checkpoints & runs
    # ------------------------------------------------------------------

    def _scan_checkpoints(self) -> None:
        def _opts(pattern):
            files = sorted(self.ckpt_dir.glob(pattern))
            return [(p.name, str(p)) for p in files] or [("(none)", "")]

        self._dd_s1.options = _opts("*mri1*.safetensors")
        self._dd_s2.options = _opts("*mri2*.safetensors") + _opts("*scar*.safetensors")
        self._dd_ct.options = _opts("*ct*.safetensors")

    def _refresh_runs(self) -> None:
        runs = []
        if self.out_root.exists():
            runs = sorted([d.name for d in self.out_root.iterdir() if d.is_dir() and d.name.startswith("run_")], reverse=True)
        self._dd_run.options = [(r, r) for r in runs]
        self._txt_new.options = ["(auto)", "(latest)"] + runs
        if runs:
            self._dd_run.value = runs[0]

    def _on_refresh(self, _btn) -> None:
        self._scan_checkpoints()
        self._refresh_runs()
        self._on_task_change(None)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _on_load_models(self, _btn) -> None:
        from models import CARE2026_CT_Model, CARE2026_MRI_Stage1_Model, CARE2026_MRI_Stage2_Model

        loaded = []
        for name, dd, key in [
            ("S1", self._dd_s1, "_s1_model"),
            ("S2", self._dd_s2, "_s2_model"),
            ("CT", self._dd_ct, "_ct_model"),
        ]:
            path = dd.value
            if not path or Path(path).name == "(none)":
                continue
            cls_map = {"S1": CARE2026_MRI_Stage1_Model, "S2": CARE2026_MRI_Stage2_Model, "CT": CARE2026_CT_Model}
            model, aux = cls_map[name].from_checkpoint(path, device=self._device)
            model.train_config.update(aux)
            setattr(self, key, model.to(self._device).eval())
            loaded.append(f"{name}={Path(path).name}")
        self._lbl_status.value = f"<b>Loaded:</b> {', '.join(loaded)}" if loaded else "<i>No models selected.</i>"
        self._on_task_change(None)

    # ------------------------------------------------------------------
    # Task / sample
    # ------------------------------------------------------------------

    def _get_val_records(self, task: int) -> List[str]:
        val_dir = self.val_root / f"task{task}" / "val_data"
        if not val_dir.exists():
            return []
        return sorted(
            [d.name for d in val_dir.iterdir() if d.is_dir() and d.name.startswith("val_")], key=lambda r: int(r.split("_")[1])
        )

    def _pred_path(self, task: int, rec: str) -> Optional[Path]:
        run = self._dd_run.value
        if not run:
            return None
        dir_map = {1: "LA scar quantification", 2: "LA cavity segmentation", 3: "LA multi-structure segmentation"}
        p = self.out_root / run / dir_map[task] / rec / f"{rec}_pred.nii.gz"
        return p if p.exists() else None

    def _on_task_change(self, change) -> None:
        task = self._dd_task.value
        records = self._get_val_records(task)
        run = self._dd_run.value
        done = sum(1 for r in records if self._pred_path(task, r)) if run else 0
        self._dd_sample.options = records
        self._lbl_status.value = f"Task {task}: <b>{done}/{len(records)}</b> predicted" if records else ""

    def _on_sample_change(self, change) -> None:
        self._on_preview(None)

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------

    def _run_inference(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (data, live_pred, saved_pred) for current selection."""
        task = self._dd_task.value
        rec = self._dd_sample.value
        if not rec:
            return None, None, None

        t1, t2 = self._sl_s1.value, self._sl_s2.value

        if task == 3:
            img_path = self.val_root / "task3" / "val_data" / rec / f"{int(rec.split('_')[1]):04d}.nii.gz"
        else:
            img_path = self.val_root / f"task{task}" / "val_data" / rec / "enhanced.nii.gz"
        if not img_path.exists():
            return None, None, None

        data = nib.load(str(img_path)).get_fdata().astype(np.float32)

        # Live prediction
        live = None
        try:
            if task == 3 and self._ct_model is not None:
                out = predict_ct(img_path, self._ct_model, device=self._device)
                self._live_output = {"ct": out.ct_mask}
                live = out.ct_mask
            elif task in (1, 2) and self._s1_model is not None and self._s2_model is not None:
                out = predict_mri_two_stage(
                    img_path, self._s1_model, self._s2_model, device=self._device, s1_threshold=t1, s2_threshold=t2
                )
                self._live_output = {"la": out.la_mask, "scar": out.scar_mask}
                live = out.scar_mask if task == 1 else out.la_mask
        except Exception as e:
            self._lbl_status.value = f"<b style='color:red'>Inference error:</b> {e}"

        # Saved prediction (from disk)
        saved = None
        saved_path = self._pred_path(task, rec)
        if saved_path is not None:
            saved = nib.load(str(saved_path)).get_fdata().astype(np.uint8)

        return data, live, saved

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _on_preview(self, _btn) -> None:
        data, live, saved = self._run_inference()
        if data is None:
            return

        task = self._dd_task.value
        rec = self._dd_sample.value
        n_slices = data.shape[-1]
        mid = n_slices // 2

        def _draw_panel(out_widget, title, mask, show_mask=True):
            out_widget.clear_output(wait=True)
            with out_widget:

                def _plot(sl=mid):
                    fig, ax = plt.subplots(figsize=(7, 7))
                    ax.imshow(data[..., sl], cmap="gray", origin="lower")
                    if show_mask and mask is not None:
                        _overlay(ax, mask, sl, task)
                    ax.set_title(f"{title} — {rec}  (slice {sl + 1}/{n_slices})")
                    ax.axis("off")
                    _add_legend(ax, task, mask is not None and show_mask)
                    fig.tight_layout()
                    plt.show()

                from ipywidgets import interact

                interact(_plot, sl=IntSlider(min=0, max=n_slices - 1, step=1, value=mid, description="Slice"))

        # Left: live preview
        _draw_panel(self._out_live, "Live Preview", live)

        # Right: saved prediction
        if saved is not None:
            _draw_panel(self._out_saved, "Saved", saved)
        else:
            self._out_saved.clear_output(wait=True)
            with self._out_saved:
                print("(Not predicted yet — click Save)")

    # ------------------------------------------------------------------
    # Threshold change
    # ------------------------------------------------------------------

    def _on_threshold_change(self, change) -> None:
        # Re-run preview if any output already shown
        if self._out_live.outputs:
            data, live, saved = self._run_inference()
            if data is None or live is None:
                return
            # Only update live panel
            self._out_live.clear_output(wait=True)
            with self._out_live:
                self._render_slice_view(data, live, self._dd_task.value, self._dd_sample.value, "Live Preview")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _on_save(self, _btn) -> None:
        task = self._dd_task.value
        rec = self._dd_sample.value
        if not rec or self._live_output is None:
            self._lbl_status.value = "<b style='color:red'>Nothing to save. Run Preview first.</b>"
            return

        run = self._dd_run.value
        txt_val = self._txt_new.value
        if not run or txt_val == "(auto)":
            run = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        elif txt_val == "(latest)":
            run = self._dd_run.options[0][0] if self._dd_run.options else datetime.now().strftime("run_%Y%m%d_%H%M%S")
        elif txt_val != run:
            run = txt_val

        dir_map = {1: "LA scar quantification", 2: "LA cavity segmentation", 3: "LA multi-structure segmentation"}
        save_dir = self.out_root / run / dir_map[task] / rec
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{rec}_pred.nii.gz"

        masks = self._live_output
        if task == 3 and "ct" in masks:
            mask = masks["ct"]
        elif task == 1 and "scar" in masks:
            mask = masks["scar"]
        else:
            mask = masks.get("la")
        if mask is None:
            return

        nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine=np.eye(4)), str(out_path))
        self._lbl_status.value = f"<b style='color:green'>Saved:</b> {out_path}"
        self._refresh_runs()
        self._on_task_change(None)
        self._on_preview(None)

    # ------------------------------------------------------------------
    # Package
    # ------------------------------------------------------------------

    def _on_package(self, _btn) -> None:
        run = self._dd_run.value
        if not run:
            self._lbl_status.value = "<b style='color:red'>No run selected.</b>"
            return
        results_dir = self.out_root / run
        if not results_dir.exists():
            self._lbl_status.value = f"<b style='color:red'>Run dir not found:</b> {results_dir}"
            return

        dir_map = {1: "LA scar quantification", 2: "LA cavity segmentation", 3: "LA multi-structure segmentation"}
        missing = []
        for task in [1, 2, 3]:
            for rec in self._get_val_records(task):
                if not (results_dir / dir_map[task] / rec / f"{rec}_pred.nii.gz").exists():
                    missing.append((task, rec))

        if missing:
            lines = ["<b style='color:red'>Missing predictions:</b><br>"]
            for t, r in missing:
                lines.append(f"Task {t}: {r}<br>")
            self._lbl_status.value = "".join(lines)
            return

        zip_path = package_submission(results_dir=results_dir, team_name="REVENGER")
        self._lbl_status.value = f"<b style='color:green'>Package:</b> {zip_path}"

    # ------------------------------------------------------------------
    # Slice-view helpers
    # ------------------------------------------------------------------

    def _render_slice_view(self, data, mask, task, rec, title):
        n_slices = data.shape[-1]
        mid = n_slices // 2

        def _plot(sl=mid):
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.imshow(data[..., sl], cmap="gray", origin="lower")
            _overlay(ax, mask, sl, task)
            ax.set_title(f"{title} — {rec}  (slice {sl + 1}/{n_slices})")
            ax.axis("off")
            _add_legend(ax, task, mask is not None)
            fig.tight_layout()
            plt.show()

        from ipywidgets import interact

        interact(_plot, sl=IntSlider(min=0, max=n_slices - 1, step=1, value=mid, description="Slice"))

    def show(self) -> None:
        display(self._ui)


# ------------------------------------------------------------------
# Shared drawing helpers
# ------------------------------------------------------------------


def _overlay(ax, mask, sl, task):
    if task == 3:
        for cls_id, color in [(1, _PALETTE["scar"]), (2, _PALETTE["pv"]), (3, _PALETTE["laa"])]:
            m = (mask[..., sl] == cls_id).astype(np.uint8)
            if m.max() > 0:
                ax.contour(m, levels=[0.5], colors=[color], linewidths=1.5)
    elif task == 1:
        if mask.max() > 0:
            ax.contour(mask[..., sl], levels=[0.5], colors=[_PALETTE["scar"]], linewidths=1.5)
    elif task == 2:
        if mask.max() > 0:
            ax.contour(mask[..., sl], levels=[0.5], colors=[_PALETTE["la"]], linewidths=1.5)


def _add_legend(ax, task, has_mask):
    if not has_mask:
        return
    items = []
    if task == 1:
        items = [(_PALETTE["scar"], "LA scar")]
    elif task == 2:
        items = [(_PALETTE["la"], "LA cavity")]
    elif task == 3:
        items = [(_PALETTE["scar"], "LA"), (_PALETTE["pv"], "PV"), (_PALETTE["laa"], "LAA")]
    handles = [mpatches.Patch(color=c, label=l) for c, l in items]
    if handles:
        ax.legend(handles=handles, loc="upper right", framealpha=0.7, fontsize="small")
