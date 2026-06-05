"""
Data reader for the CARE 2026 Left Atrium challenge.

Covers:
- LGE-MRI data (Center A) for Task 1 (LA scar quantification) and Task 2 (LA cavity segmentation)
- CT data (Center D) for Task 3 (LA multi-structure segmentation)

Data layout on disk
-------------------
Task 1 (LA scar, MRI)::

    <db_dir>/LA scar quantification（MRI）/train_data/train_N/
        enhanced.nii.gz          -- LGE-MRI image
        atriumSegImgMO.nii.gz    -- LA cavity binary mask  (raw values 0/~420, normalised to 0/1)
        scarSegImgM.nii.gz       -- LA scar binary mask    (values 0/1)

Task 2 (LA cavity, MRI)::

    <db_dir>/LA cavity segmentation（MRI）/train_data/train_N/
        enhanced.nii.gz          -- LGE-MRI image
        atriumSegImgMO.nii.gz    -- LA cavity binary mask  (raw values 0/~420, normalised to 0/1)

Task 3 (multi-structure, CT)::

    <db_dir>/cardiac anatomy segmentation（CT）/train_data/train_N/
        XXXX.nii.gz              -- CT image  (XXXX = zero-padded 4-digit record number)
        label_XXXX.nii.gz        -- multi-class mask (0/1/2/3), only available for train_1..train_50
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_ecg.databases.base import DataBaseInfo, _DataBase
from torch_ecg.utils.misc import add_docstring

from utils.viz_utils import _is_notebook, _slice_view_interactive, _slice_view_static

__all__ = [
    "CARE2026_MRI",
    "CARE2026_CT",
]


_CARE2026_MRI_INFO = DataBaseInfo(
    title="""
    MICCAI CARE 2026 - Left Atrium LGE-MRI (Tasks 1 & 2)
    """,
    about="""
    1. Task 1 (LA scar quantification): 60 LGE-MRIs from Center A (Utah NAMIC-CARMA).
       Labels: LA cavity mask (atriumSegImgMO) + LA scar mask (scarSegImgM).
    2. Task 2 (LA cavity segmentation): 130 LGE-MRIs from Center A.
       Labels: LA cavity mask (atriumSegImgMO) only.
    3. Validation: 10 records for Task 1, 20 records for Task 2 (released May 2026).
       Test: 24 records for Task 1, 14 records (Center A) + 20 (Center B) + 10 (Center C) for Task 2.
    4. All images are in NIfTI format.

    Training-set image statistics (N = 190 LGE-MRI volumes, Center A):
    ─────────────────────────────────────────────────────────────────
    Spatial resolution:  0.625 × 0.625 × 2.5 mm  (in-plane × z)
    Image matrix:        576 × 576 × 44 voxels  (modal value; all volumes)
    Field of view:       360 × 360 × 110 mm³

    LA cavity bounding box (in canonical 576×576×44 voxel space):
      H (x):  p50 ≈ 155 px,  p75 ≈ 190 px,  p95 ≈ 230 px
      W (y):  p50 ≈ 145 px,  p75 ≈ 175 px,  p95 ≈ 210 px
      D (z):  all 44 slices  (LA spans the full z extent in Center A scans)

    → Stage-2 crop shape (256×256×44) comfortably covers p95 LA extent
      (230 px) with ~13 px margin on each side.

    Annotation:
      LA cavity mask: raw values {0, ~420}, normalised to {0, 1} on load.
      LA scar mask:   values {0, 1}; average scar-to-LA ratio ≈ 5–15 %.
      Class imbalance ratio (scar vs. non-scar within LA): ~ 1 : 10–20.
    """,
    usage=[
        "LA Scar Quantification",
        "LA Cavity Segmentation",
    ],
    references=[
        "https://www.zmic.org.cn/care_2026/track_leftatrium/",
    ],
    doi=[],
)

_CARE2026_CT_INFO = DataBaseInfo(
    title="""
    MICCAI CARE 2026 - Left Atrium CT (Task 3)
    """,
    about="""
    1. Task 3 (multi-structure segmentation): 150 CTs from Center D (Fuzhou University Hospital).
    2. Labels available only for train_1..train_50 (50 labelled, 100 unlabelled).
    3. Multi-class label values: 0=background, 1=left atrium, 2=pulmonary veins, 3=left atrial appendage.
    4. Image naming: XXXX.nii.gz; label: label_XXXX.nii.gz (XXXX = 4-digit zero-padded record number).
    5. Validation: 20 records; Test: 130 records.

    Training-set image statistics (N = 150 CT volumes, Center D):
    ─────────────────────────────────────────────────────────────
    Spatial resolution:  in-plane 0.30–0.80 mm (variable), z = 0.5 mm (fixed)
    Image matrix:        variable; typically 512×512 in-plane
    HU range:            soft-tissue window; clipped to [−200, +800] HU for training

    Annotation (50 labelled volumes):
      Class 1 (LA):               present in all 50; typical DSC among top teams ≈ 0.93–0.96
      Class 2 (pulmonary veins):  thin tubular structures; hardest class; DSC ≈ 0.82–0.89
      Class 3 (LA appendage):     small / variable shape; DSC ≈ 0.80–0.88
    Semi-supervised split: 50 labelled (train_1..50) + 100 unlabelled (train_51..150).
    """,
    usage=[
        "LA Multi-Structure Segmentation",
    ],
    references=[
        "https://www.zmic.org.cn/care_2026/track_leftatrium/",
    ],
    doi=[],
)


@add_docstring(_CARE2026_MRI_INFO.format_database_docstring(), mode="prepend")
class CARE2026_MRI(_DataBase):
    """Reader for the LGE-MRI portion of the CARE 2026 Left Atrium challenge.

    Parameters
    ----------
    db_dir : path-like, optional
        Root directory of the dataset (the folder that contains both
        the scar-quantification and cavity-segmentation MRI sub-folders).
    task : {1, 2}, default 1
        Which MRI task to load.

        * 1 -- LA scar quantification (60 training samples, scar + cavity labels)
        * 2 -- LA cavity segmentation (130 training samples, cavity label only)
    working_dir : path-like, optional
        Working directory for intermediate files and logs.
    verbose : int, default 1
        Logging verbosity level.
    kwargs : dict, optional
        Additional keyword arguments passed to the base class.

    """

    __task1_class_map__: Dict[int, str] = {
        0: "background",
        1: "LA scar",
    }
    __task2_class_map__: Dict[int, str] = {
        0: "background",
        1: "left atrium",
    }
    __palette__: Dict[int, str] = {
        0: (0, 0, 0, 0),  # background — transparent
        1: "#00FFFF",  # LA cavity — cyan
        2: "#FF4444",  # LA scar — red
    }
    __default_crop_pad__: List[int] = [7, 7, 3]

    def __init__(
        self,
        db_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        task: Literal[1, 2] = 1,
        working_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        assert task in (1, 2), f"task must be 1 or 2, got {task}"
        self.task = task
        super().__init__(
            db_name="CARE2026_MRI",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.data_ext = "nii.gz"
        self.ann_ext = "nii.gz"
        self._ls_rec()

    # ------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------

    @property
    def _task_subdir(self) -> Path:
        """Sub-directory on disk for the current task."""
        if self.task == 1:
            return self.db_dir / "LA scar quantification（MRI）" / "train_data"
        return self.db_dir / "LA cavity segmentation（MRI）" / "train_data"

    @property
    def class_map(self) -> Dict[int, str]:
        """Class-id to class-name mapping for the current task."""
        if self.task == 1:
            return self.__task1_class_map__
        return self.__task2_class_map__

    @property
    def label2id(self) -> Dict[str, int]:
        return {v: k for k, v in self.class_map.items()}

    @property
    def id2label(self) -> Dict[int, str]:
        return dict(self.class_map)

    # ------------------------------------------------------------------
    # Record listing
    # ------------------------------------------------------------------

    def _ls_rec(self) -> None:
        """Scan the task sub-directory and populate ``self._df_records``."""
        task_dir = self._task_subdir
        if not task_dir.exists():
            self.logger.warning(f"Task directory not found: {task_dir}")
            self._df_records = self._empty_mri_df()
            self._all_records = []
            return

        records = []
        for d in sorted(task_dir.iterdir(), key=lambda p: int(p.name.split("_")[1])):
            if not d.is_dir():
                continue
            img_path = d / "enhanced.nii.gz"
            la_path = d / "atriumSegImgMO.nii.gz"
            scar_path = d / "scarSegImgM.nii.gz"
            if not img_path.exists():
                continue
            records.append(
                {
                    "record": d.name,
                    "path": img_path,
                    "la_path": la_path if la_path.exists() else None,
                    "scar_path": scar_path if scar_path.exists() else None,
                }
            )

        if not records:
            self.logger.warning("No records found.")
            self._df_records = self._empty_mri_df()
            self._all_records = []
            return

        self._df_records = pd.DataFrame(records).set_index("record")
        self._all_records = self._df_records.index.tolist()

    @staticmethod
    def _empty_mri_df() -> pd.DataFrame:
        return pd.DataFrame(columns=["path", "la_path", "scar_path"])

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _resolve_rec(self, rec: Union[str, int]) -> str:
        if isinstance(rec, int):
            return self._all_records[rec]
        return rec

    def get_data_path(self, rec: Union[str, int]) -> Path:
        """Return the path to the LGE-MRI image for *rec*."""
        return self._df_records.loc[self._resolve_rec(rec), "path"]

    def get_la_path(self, rec: Union[str, int]) -> Optional[Path]:
        """Return the path to the LA cavity annotation for *rec*."""
        return self._df_records.loc[self._resolve_rec(rec), "la_path"]

    def get_scar_path(self, rec: Union[str, int]) -> Optional[Path]:
        """Return the path to the LA scar annotation for *rec* (Task 1 only)."""
        return self._df_records.loc[self._resolve_rec(rec), "scar_path"]

    def get_ann_path(self, rec: Union[str, int]) -> Optional[Path]:
        """Return the primary annotation path (scar for Task 1, LA cavity for Task 2)."""
        if self.task == 1:
            return self.get_scar_path(rec)
        return self.get_la_path(rec)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(
        self,
        rec: Union[str, int],
        output_shape: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Load the LGE-MRI image for *rec*.

        Parameters
        ----------
        rec : str or int
            Record name (e.g. ``'train_1'``) or integer index.
        output_shape : sequence of int, optional
            If given, the volume is resampled to this shape via trilinear interpolation.

        Returns
        -------
        numpy.ndarray
            3-D float32 array of shape ``(H, W, D)``.

        """
        img = nib.load(str(self.get_data_path(rec))).get_fdata().astype(np.float32)
        if output_shape is not None:
            img = self.resample_data(img, output_shape)
        return img

    def load_la_ann(
        self,
        rec: Union[str, int],
        output_shape: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Load the LA cavity binary mask for *rec*.

        The raw mask values are {0, ~420}; this method normalises them to {0, 1}.

        Parameters
        ----------
        rec : str or int
            Record name or integer index.
        output_shape : sequence of int, optional
            If given, the mask is resampled (nearest-neighbour) to this shape.

        Returns
        -------
        numpy.ndarray of dtype uint8
            Binary 3-D mask of shape ``(H, W, D)``.

        """
        path = self.get_la_path(rec)
        if path is None:
            raise FileNotFoundError(f"LA cavity annotation not found for record '{self._resolve_rec(rec)}'")
        mask = (nib.load(str(path)).get_fdata() > 0).astype(np.uint8)
        if output_shape is not None:
            mask = self.resample_ann(mask, output_shape)
        return mask

    def load_scar_ann(
        self,
        rec: Union[str, int],
        output_shape: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Load the LA scar binary mask for *rec* (Task 1 only).

        Parameters
        ----------
        rec : str or int
            Record name or integer index.
        output_shape : sequence of int, optional
            If given, the mask is resampled (nearest-neighbour) to this shape.

        Returns
        -------
        numpy.ndarray of dtype uint8
            Binary 3-D mask of shape ``(H, W, D)``.

        """
        path = self.get_scar_path(rec)
        if path is None:
            raise FileNotFoundError(f"Scar annotation not found for record '{self._resolve_rec(rec)}'")
        mask = (nib.load(str(path)).get_fdata() > 0).astype(np.uint8)
        if output_shape is not None:
            mask = self.resample_ann(mask, output_shape)
        return mask

    def load_ann(
        self,
        rec: Union[str, int],
        output_shape: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Load the primary annotation for *rec*.

        For Task 1 this returns the scar mask; for Task 2 the LA cavity mask.

        Parameters
        ----------
        rec : str or int
            Record name or integer index.
        output_shape : sequence of int, optional
            If given, the mask is resampled to this shape.

        Returns
        -------
        numpy.ndarray of dtype uint8
            Binary 3-D segmentation mask.

        """
        if self.task == 1:
            return self.load_scar_ann(rec, output_shape)
        return self.load_la_ann(rec, output_shape)

    # ------------------------------------------------------------------
    # Bounding-box helpers
    # ------------------------------------------------------------------

    def load_ann_box(
        self,
        rec: Union[str, int],
        pad: Optional[Union[int, Sequence[int]]] = None,
        ann_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the axis-aligned bounding box of the LA cavity annotation.

        The LA cavity is used as the reference region for both tasks (the scar
        lies entirely within it).

        Parameters
        ----------
        rec : str or int
            Record name or integer index.
        pad : int or sequence of int, optional
            Voxel padding in each axis.  Defaults to ``__default_crop_pad__``.
        ann_mask : numpy.ndarray, optional
            Pre-loaded LA cavity mask.  If *None* it is loaded from disk.

        Returns
        -------
        numpy.ndarray of shape (3, 2)
            ``[[x_min, x_max], [y_min, y_max], [z_min, z_max]]``.

        """
        if ann_mask is None:
            ann_mask = self.load_la_ann(rec)
        if pad is None:
            pad = self.__default_crop_pad__
        if isinstance(pad, int):
            pad = [pad] * 3
        x, y, z = np.where(ann_mask > 0)
        x_min = max(0, int(x.min()) - pad[0])
        x_max = min(ann_mask.shape[0], int(x.max()) + pad[0])
        y_min = max(0, int(y.min()) - pad[1])
        y_max = min(ann_mask.shape[1], int(y.max()) + pad[1])
        z_min = max(0, int(z.min()) - pad[2])
        z_max = min(ann_mask.shape[2], int(z.max()) + pad[2])
        return np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])

    def load_data_cropped(
        self,
        rec: Union[str, int],
        pad: Optional[Union[int, Sequence[int]]] = None,
        output_shape: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Load the LGE-MRI cropped to the LA bounding box.

        Parameters
        ----------
        rec : str or int
            Record name or integer index.
        pad : int or sequence of int, optional
            Voxel padding around the bounding box.
        output_shape : sequence of int, optional
            If given, the cropped volume is resampled to this shape.

        Returns
        -------
        numpy.ndarray
            Cropped 3-D float image.

        """
        data = self.load_data(rec)
        (x0, x1), (y0, y1), (z0, z1) = self.load_ann_box(rec, pad=pad)
        cropped = data[x0:x1, y0:y1, z0:z1]
        if output_shape is not None:
            cropped = self.resample_data(cropped, output_shape)
        return cropped

    def load_ann_cropped(
        self,
        rec: Union[str, int],
        pad: Optional[Union[int, Sequence[int]]] = None,
        output_shape: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Load the primary annotation cropped to the LA bounding box.

        Parameters
        ----------
        rec : str or int
            Record name or integer index.
        pad : int or sequence of int, optional
            Voxel padding around the bounding box.
        output_shape : sequence of int, optional
            If given, the cropped mask is resampled to this shape.

        Returns
        -------
        numpy.ndarray of dtype uint8
            Cropped segmentation mask.

        """
        ann = self.load_ann(rec)
        (x0, x1), (y0, y1), (z0, z1) = self.load_ann_box(rec, pad=pad)
        cropped = ann[x0:x1, y0:y1, z0:z1]
        if output_shape is not None:
            cropped = self.resample_ann(cropped, output_shape)
        return cropped

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def view_data(
        self,
        rec: Union[str, int],
        channels: Optional[Union[int, Sequence[int]]] = None,
        with_ann: bool = True,
        orthoview: bool = False,
        output_shape: Optional[Sequence[int]] = None,
        crop: bool = False,
        crop_pad: Optional[Union[int, Sequence[int]]] = None,
        data: Optional[np.ndarray] = None,
        interactive: Optional[bool] = None,
        overlay_mode: str = "filled+hatch",
    ) -> None:
        """Visualise slices of the LGE-MRI, optionally overlaid with the annotation.

        In Jupyter notebooks the default is an interactive slider-based view
        (one slice at a time).  Outside notebooks the default is a static
        grid of all (or selected) slices.

        Parameters
        ----------
        rec : str or int
            Record name or integer index.
        channels : int or sequence of int, optional
            Slice indices to display (static mode only).  If *None*, all slices.
        with_ann : bool, default True
            Overlay the annotation contours on the image.
        orthoview : bool, default False
            Use nibabel's orthoview (ignores most other arguments).
        output_shape : sequence of int, optional
            Resample to this shape before displaying.
        crop : bool, default False
            Crop to the LA bounding box before displaying.
        crop_pad : int or sequence of int, optional
            Bounding-box padding (only used when *crop* is True).
        data : numpy.ndarray, optional
            Pre-loaded image array; avoids re-reading from disk.
        interactive : bool, optional
            Force interactive (``True``) or static (``False``) mode.
            Defaults to auto-detection based on the runtime environment.
        overlay_mode : str, default ``"filled+hatch"``
            Overlay style: ``"contour"``, ``"filled"``, or ``"filled+hatch"``.
            In interactive mode this is controlled by a dropdown; this value
            sets the initial selection.

        """
        rec = self._resolve_rec(rec)
        if data is None:
            if crop:
                data = self.load_data_cropped(rec, pad=crop_pad, output_shape=output_shape)
            else:
                data = self.load_data(rec, output_shape=output_shape)

        if orthoview:
            nib.load(str(self.get_data_path(rec))).orthoview()
            return

        # Build mask dict for overlay (keys = class IDs matching __palette__)
        masks: Dict[int, np.ndarray] = {}
        title = f"MRI — {rec}"
        if with_ann:
            ann = (
                self.load_ann_cropped(rec, pad=crop_pad, output_shape=output_shape)
                if crop
                else self.load_ann(rec, output_shape=output_shape)
            )
            if self.task == 1:
                # Task 1: ann = scar mask; also load LA cavity for context
                masks[2] = (ann > 0).astype(np.uint8)  # scar → red
                try:
                    la = self.load_la_ann(rec, output_shape=output_shape)
                    if crop:
                        (x0, x1), (y0, y1), (z0, z1) = self.load_ann_box(rec, pad=crop_pad)
                        la = la[x0:x1, y0:y1, z0:z1]
                    masks[1] = la.astype(np.uint8)  # LA → cyan
                except Exception:
                    pass
            else:
                # Task 2: ann = LA cavity mask
                masks[1] = ann.astype(np.uint8)  # LA → cyan
            title += " + annotation"

        # Human-readable names matching the palette keys used above
        if self.task == 1:
            _class_names = {1: "LA cavity", 2: "LA scar"}
        else:
            _class_names = {1: "LA cavity"}

        # Choose interactive vs static
        if interactive is None:
            interactive = _is_notebook()

        if interactive:
            _slice_view_interactive(data, masks, self.__palette__, _class_names, title=title)
        else:
            _slice_view_static(
                data, masks, self.__palette__, _class_names, channels=channels, title=title, overlay_mode=overlay_mode
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def database_info(self) -> DataBaseInfo:
        return _CARE2026_MRI_INFO

    @property
    def url(self) -> str:
        return "https://www.zmic.org.cn/care_2026/track_leftatrium/"

    @property
    def webpage(self) -> str:
        return self.url

    # ------------------------------------------------------------------
    # Resampling utilities
    # ------------------------------------------------------------------

    @staticmethod
    def resample_data(data: np.ndarray, shape: Sequence[int]) -> np.ndarray:
        """Trilinear resampling of a float volumetric image.

        Parameters
        ----------
        data : numpy.ndarray
            3-D float image.
        shape : sequence of int
            Target spatial shape ``(H, W, D)``.

        Returns
        -------
        numpy.ndarray
            Resampled image preserving the input dtype.

        """
        t = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        out = F.interpolate(t, size=tuple(shape), mode="trilinear", align_corners=True)
        return out.squeeze().numpy().astype(data.dtype)

    @staticmethod
    def resample_ann(mask: np.ndarray, shape: Sequence[int]) -> np.ndarray:
        """Nearest-neighbour resampling of a segmentation mask.

        Parameters
        ----------
        mask : numpy.ndarray
            3-D integer segmentation mask.
        shape : sequence of int
            Target spatial shape.

        Returns
        -------
        numpy.ndarray of dtype uint8
            Resampled mask.

        """
        t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        out = F.interpolate(t, size=tuple(shape), mode="nearest")
        return out.squeeze().numpy().astype(np.uint8)


# ---------------------------------------------------------------------------
# CT reader (Task 3)
# ---------------------------------------------------------------------------


@add_docstring(_CARE2026_CT_INFO.format_database_docstring(), mode="prepend")
class CARE2026_CT(_DataBase):
    """Reader for the CT portion of the CARE 2026 Left Atrium challenge (Task 3).

    Parameters
    ----------
    db_dir : path-like, optional
        Root directory of the dataset (the folder containing the
        cardiac anatomy segmentation CT sub-folder).
    working_dir : path-like, optional
        Working directory for intermediate files and logs.
    verbose : int, default 1
        Logging verbosity level.
    kwargs : dict, optional
        Additional keyword arguments passed to the base class.

    """

    __class_map__: Dict[int, str] = {
        0: "background",
        1: "left atrium",
        2: "pulmonary veins",
        3: "left atrial appendage",
    }
    __palette__: Dict[int, str] = {
        0: (0, 0, 0, 0),  # background — transparent
        1: "#FF4444",  # left atrium — red
        2: "#4488FF",  # pulmonary veins — blue
        3: "#44FF44",  # left atrial appendage — green
    }
    __default_crop_pad__: List[int] = [5, 5, 5]

    def __init__(
        self,
        db_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        working_dir: Optional[Union[str, bytes, os.PathLike]] = None,
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            db_name="CARE2026_CT",
            db_dir=db_dir,
            working_dir=working_dir,
            verbose=verbose,
            **kwargs,
        )
        self.data_ext = "nii.gz"
        self.ann_ext = "nii.gz"
        self._ls_rec()

    # ------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------

    @property
    def _task_subdir(self) -> Path:
        return self.db_dir / "cardiac anatomy segmentation（CT）" / "train_data"

    @property
    def label2id(self) -> Dict[str, int]:
        return {v: k for k, v in self.__class_map__.items()}

    @property
    def id2label(self) -> Dict[int, str]:
        return dict(self.__class_map__)

    # ------------------------------------------------------------------
    # Record listing
    # ------------------------------------------------------------------

    def _ls_rec(self) -> None:
        """Scan the task sub-directory and populate ``self._df_records``."""
        task_dir = self._task_subdir
        if not task_dir.exists():
            self.logger.warning(f"Task directory not found: {task_dir}")
            self._df_records = self._empty_ct_df()
            self._df_records_labeled = self._empty_ct_df()
            self._all_records = []
            return

        records = []
        for d in sorted(task_dir.iterdir(), key=lambda p: int(p.name.split("_")[1])):
            if not d.is_dir():
                continue
            rec_num = int(d.name.split("_")[1])
            num_str = str(rec_num).zfill(4)
            img_path = d / f"{num_str}.nii.gz"
            label_path = d / f"label_{num_str}.nii.gz"
            if not img_path.exists():
                continue
            records.append(
                {
                    "record": d.name,
                    "path": img_path,
                    "ann_path": label_path if label_path.exists() else None,
                }
            )

        if not records:
            self.logger.warning("No records found.")
            self._df_records = self._empty_ct_df()
            self._df_records_labeled = self._empty_ct_df()
            self._all_records = []
            return

        self._df_records = pd.DataFrame(records).set_index("record")
        self._df_records_labeled = self._df_records[self._df_records["ann_path"].notna()]
        self._all_records = self._df_records.index.tolist()

    @staticmethod
    def _empty_ct_df() -> pd.DataFrame:
        return pd.DataFrame(columns=["path", "ann_path"])

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _resolve_rec(self, rec: Union[str, int]) -> str:
        if isinstance(rec, int):
            return self._all_records[rec]
        return rec

    def get_data_path(self, rec: Union[str, int]) -> Path:
        """Return the path to the CT image for *rec*."""
        return self._df_records.loc[self._resolve_rec(rec), "path"]

    def get_ann_path(self, rec: Union[str, int]) -> Optional[Path]:
        """Return the path to the multi-class segmentation mask for *rec*, or *None* if unlabelled."""
        return self._df_records.loc[self._resolve_rec(rec), "ann_path"]

    @property
    def labeled_records(self) -> List[str]:
        """Record IDs that have ground-truth labels (train_1..train_50)."""
        return self._df_records_labeled.index.tolist()

    @property
    def unlabeled_records(self) -> List[str]:
        """Record IDs without ground-truth labels (train_51..train_150)."""
        return self._df_records[self._df_records["ann_path"].isna()].index.tolist()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(
        self,
        rec: Union[str, int],
        output_shape: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Load the CT image for *rec*.

        Parameters
        ----------
        rec : str or int
            Record name or integer index.
        output_shape : sequence of int, optional
            If given, the volume is resampled to this shape via trilinear interpolation.

        Returns
        -------
        numpy.ndarray
            3-D float32 array of shape ``(H, W, D)``.

        """
        img = nib.load(str(self.get_data_path(rec))).get_fdata().astype(np.float32)
        if output_shape is not None:
            img = self.resample_data(img, output_shape)
        return img

    def load_ann(
        self,
        rec: Union[str, int],
        output_shape: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Load the multi-class segmentation mask for *rec*.

        Parameters
        ----------
        rec : str or int
            Record name or integer index.
        output_shape : sequence of int, optional
            If given, the mask is resampled (nearest-neighbour) to this shape.

        Returns
        -------
        numpy.ndarray of dtype uint8
            Multi-class 3-D mask with values in {0, 1, 2, 3}.

        Raises
        ------
        FileNotFoundError
            If the record has no ground-truth label.

        """
        path = self.get_ann_path(rec)
        if path is None:
            raise FileNotFoundError(f"No label available for record '{self._resolve_rec(rec)}'")
        mask = nib.load(str(path)).get_fdata().astype(np.uint8)
        if output_shape is not None:
            mask = self.resample_ann(mask, output_shape)
        return mask

    # ------------------------------------------------------------------
    # Bounding-box helpers
    # ------------------------------------------------------------------

    def load_ann_box(
        self,
        rec: Union[str, int],
        pad: Optional[Union[int, Sequence[int]]] = None,
        ann_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the axis-aligned bounding box of all foreground structures.

        Parameters
        ----------
        rec : str or int
            Record name or integer index.
        pad : int or sequence of int, optional
            Voxel padding in each axis.  Defaults to ``__default_crop_pad__``.
        ann_mask : numpy.ndarray, optional
            Pre-loaded mask.  If *None*, it is loaded from disk.

        Returns
        -------
        numpy.ndarray of shape (3, 2)
            ``[[x_min, x_max], [y_min, y_max], [z_min, z_max]]``.

        """
        if ann_mask is None:
            ann_mask = self.load_ann(rec)
        if pad is None:
            pad = self.__default_crop_pad__
        if isinstance(pad, int):
            pad = [pad] * 3
        x, y, z = np.where(ann_mask > 0)
        x_min = max(0, int(x.min()) - pad[0])
        x_max = min(ann_mask.shape[0], int(x.max()) + pad[0])
        y_min = max(0, int(y.min()) - pad[1])
        y_max = min(ann_mask.shape[1], int(y.max()) + pad[1])
        z_min = max(0, int(z.min()) - pad[2])
        z_max = min(ann_mask.shape[2], int(z.max()) + pad[2])
        return np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])

    def load_data_cropped(
        self,
        rec: Union[str, int],
        pad: Optional[Union[int, Sequence[int]]] = None,
        output_shape: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Load the CT image cropped to the foreground bounding box.

        Parameters
        ----------
        rec : str or int
            Record name or integer index.
        pad : int or sequence of int, optional
            Voxel padding around the bounding box.
        output_shape : sequence of int, optional
            If given, the cropped volume is resampled to this shape.

        Returns
        -------
        numpy.ndarray
            Cropped 3-D float image.

        """
        data = self.load_data(rec)
        (x0, x1), (y0, y1), (z0, z1) = self.load_ann_box(rec, pad=pad)
        cropped = data[x0:x1, y0:y1, z0:z1]
        if output_shape is not None:
            cropped = self.resample_data(cropped, output_shape)
        return cropped

    def load_ann_cropped(
        self,
        rec: Union[str, int],
        pad: Optional[Union[int, Sequence[int]]] = None,
        output_shape: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Load the segmentation mask cropped to the foreground bounding box.

        Parameters
        ----------
        rec : str or int
            Record name or integer index.
        pad : int or sequence of int, optional
            Voxel padding around the bounding box.
        output_shape : sequence of int, optional
            If given, the cropped mask is resampled to this shape.

        Returns
        -------
        numpy.ndarray of dtype uint8
            Cropped multi-class segmentation mask.

        """
        ann = self.load_ann(rec)
        (x0, x1), (y0, y1), (z0, z1) = self.load_ann_box(rec, pad=pad, ann_mask=ann)
        cropped = ann[x0:x1, y0:y1, z0:z1]
        if output_shape is not None:
            cropped = self.resample_ann(cropped, output_shape)
        return cropped

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def view_data(
        self,
        rec: Union[str, int],
        channels: Optional[Union[int, Sequence[int]]] = None,
        with_ann: bool = True,
        output_shape: Optional[Sequence[int]] = None,
        crop: bool = False,
        crop_pad: Optional[Union[int, Sequence[int]]] = None,
        data: Optional[np.ndarray] = None,
        interactive: Optional[bool] = None,
        overlay_mode: str = "filled+hatch",
    ) -> None:
        """Visualise slices of the CT volume with multi-class annotation overlay.

        In Jupyter notebooks the default is an interactive slider-based view.
        Outside notebooks the default is a static grid.

        Parameters
        ----------
        rec : str or int
            Record name or integer index.
        channels : int or sequence of int, optional
            Slice indices to display (static mode only).
        with_ann : bool, default True
            Overlay the segmentation contours on the image.
        output_shape : sequence of int, optional
            Resample to this shape before displaying.
        crop : bool, default False
            Crop to the foreground bounding box before displaying.
        crop_pad : int or sequence of int, optional
            Bounding-box padding (only used when *crop* is True).
        data : numpy.ndarray, optional
            Pre-loaded image array; avoids re-reading from disk.
        interactive : bool, optional
            Force interactive (``True``) or static (``False``) mode.
        overlay_mode : str, default ``"filled+hatch"``
            Overlay style.  In interactive mode this is controlled by a
            dropdown; this value sets the initial selection.

        """
        rec = self._resolve_rec(rec)
        if data is None:
            if crop:
                data = self.load_data_cropped(rec, pad=crop_pad, output_shape=output_shape)
            else:
                data = self.load_data(rec, output_shape=output_shape)

        masks: Dict[int, np.ndarray] = {}
        title = f"CT — {rec}"
        if with_ann:
            try:
                ann = (
                    self.load_ann_cropped(rec, pad=crop_pad, output_shape=output_shape)
                    if crop
                    else self.load_ann(rec, output_shape=output_shape)
                )
            except FileNotFoundError:
                ann = np.zeros(data.shape[:3], dtype=np.uint8)
            for cls_id in [1, 2, 3]:
                m = (ann == cls_id).astype(np.uint8)
                if m.max() > 0:
                    masks[cls_id] = m
            title += " + annotation"

        _class_names = {1: "left atrium", 2: "pulmonary veins", 3: "left atrial appendage"}

        if interactive is None:
            interactive = _is_notebook()

        if interactive:
            _slice_view_interactive(data, masks, self.__palette__, _class_names, title=title)
        else:
            _slice_view_static(
                data, masks, self.__palette__, _class_names, channels=channels, title=title, overlay_mode=overlay_mode
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def database_info(self) -> DataBaseInfo:
        return _CARE2026_CT_INFO

    @property
    def url(self) -> str:
        return "https://www.zmic.org.cn/care_2026/track_leftatrium/"

    @property
    def webpage(self) -> str:
        return self.url

    # ------------------------------------------------------------------
    # Resampling utilities
    # ------------------------------------------------------------------

    @staticmethod
    def resample_data(data: np.ndarray, shape: Sequence[int]) -> np.ndarray:
        """Trilinear resampling of a float volumetric CT image.

        Parameters
        ----------
        data : numpy.ndarray
            3-D float image.
        shape : sequence of int
            Target spatial shape.

        Returns
        -------
        numpy.ndarray
            Resampled image preserving the input dtype.

        """
        t = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        out = F.interpolate(t, size=tuple(shape), mode="trilinear", align_corners=True)
        return out.squeeze().numpy().astype(data.dtype)

    @staticmethod
    def resample_ann(mask: np.ndarray, shape: Sequence[int]) -> np.ndarray:
        """Nearest-neighbour resampling of a multi-class segmentation mask.

        Parameters
        ----------
        mask : numpy.ndarray
            3-D integer segmentation mask.
        shape : sequence of int
            Target spatial shape.

        Returns
        -------
        numpy.ndarray of dtype uint8
            Resampled mask.

        """
        t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        out = F.interpolate(t, size=tuple(shape), mode="nearest")
        return out.squeeze().numpy().astype(np.uint8)
