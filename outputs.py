"""
Output container definitions for model predictions.
"""

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import nibabel as nib
import numpy as np

__all__ = ["CARE2026Outputs"]

# Challenge-compliant submission directory names
_TASK_DIRNAME = {
    1: "LA scar quantification",
    2: "LA cavity segmentation",
    3: "LA multi-structure segmentation",
}


@dataclass
class CARE2026Outputs:
    """Container for CARE2026 model predictions.

    Parameters
    ----------
    task : str
        Either ``"mri"`` or ``"ct"``.
    la_mask : np.ndarray, optional
        LA cavity segmentation, shape ``(H, W, D)``, dtype uint8.
        Present for ``task="mri"``.
    scar_mask : np.ndarray, optional
        LA scar segmentation, shape ``(H, W, D)``, dtype uint8.
        Present for ``task="mri"``.
    ct_mask : np.ndarray, optional
        CT multi-structure segmentation, shape ``(H, W, D)``, dtype uint8.
        Present for ``task="ct"``.
    source_affine : np.ndarray, optional
        4×4 affine matrix of the source NIfTI image.  Copied to the
        output file so predictions are aligned in the original physical space.
    source_header : nibabel header, optional
        Original NIfTI header (used to preserve spacing metadata).
    """

    task: str
    la_mask: Optional[np.ndarray] = None
    scar_mask: Optional[np.ndarray] = None
    ct_mask: Optional[np.ndarray] = None
    source_affine: Optional[np.ndarray] = None
    source_header: Optional[Any] = None

    # ------------------------------------------------------------------
    # Saving helpers
    # ------------------------------------------------------------------

    def save_as_nifti(
        self,
        output_dir: Union[str, Path],
        record_id: str,
        task_num: int,
    ) -> Path:
        """Save the prediction mask as a challenge-compliant NIfTI file.

        The file is written to::

            <output_dir>/<task_dirname>/<record_id>/<record_id>_pred.nii.gz

        Parameters
        ----------
        output_dir : path-like
            Base results directory (e.g. ``results/``).
        record_id : str
            Validation record identifier, e.g. ``"val_1"``.
        task_num : {1, 2, 3}
            Challenge task number:
            1 → scar mask (binary), 2 → LA cavity mask (binary),
            3 → multi-structure mask (0–3).

        Returns
        -------
        pathlib.Path
            Path to the saved file.
        """
        if task_num not in _TASK_DIRNAME:
            raise ValueError(f"task_num must be 1, 2, or 3, got {task_num!r}")

        mask = self._select_mask(task_num)
        if mask is None:
            raise ValueError(f"No mask available for task {task_num} " f"(task={self.task!r}).  Did you run the correct model?")

        if record_id.startswith("test_"):
            # Test phase: test_01 → 01_pred.nii.gz
            out_name = f"{record_id.split('_')[1]}_pred.nii.gz"
        else:
            # Validation phase: val_1 → val_1_pred.nii.gz
            out_name = f"{record_id}_pred.nii.gz"
        save_dir = Path(output_dir) / _TASK_DIRNAME[task_num] / record_id
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / out_name

        affine = self.source_affine if self.source_affine is not None else np.eye(4)
        nii_img = nib.Nifti1Image(mask.astype(np.uint8), affine=affine, header=self.source_header)
        nib.save(nii_img, str(out_path))
        return out_path

    def _select_mask(self, task_num: int) -> Optional[np.ndarray]:
        """Return the appropriate mask array for *task_num*."""
        if task_num == 1:
            return self.scar_mask
        if task_num == 2:
            return self.la_mask
        if task_num == 3:
            return self.ct_mask
        return None


# ---------------------------------------------------------------------------
# Submission packaging
# ---------------------------------------------------------------------------


def package_submission(
    results_dir: Union[str, Path],
    team_name: str,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Package the prediction results into a challenge-submission zip.

    The zip layout follows the official specification::

        CARE-Leftatrium-<team_name>.zip
        ├── LA scar quantification/
        │   └── val_*/
        │       └── *_pred.nii.gz
        ├── LA cavity segmentation/
        │   └── val_*/
        │       └── *_pred.nii.gz
        └── LA multi-structure segmentation/
            └── val_*/
                └── *_pred.nii.gz

    Parameters
    ----------
    results_dir : path-like
        Directory that contains ``LA scar quantification/``,
        ``LA cavity segmentation/``, and
        ``LA multi-structure segmentation/`` sub-folders.
    team_name : str
        Team name used to name the zip file.
    output_path : path-like, optional
        Destination path for the zip file.  Defaults to
        ``<results_dir>/CARE-Leftatrium-<team_name>.zip``.

    Returns
    -------
    pathlib.Path
        Path to the created zip file.
    """
    results_dir = Path(results_dir).expanduser().resolve()
    if output_path is None:
        output_path = results_dir / f"CARE-Leftatrium-{team_name}.zip"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(str(output_path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task_dir_name in _TASK_DIRNAME.values():
            task_dir = results_dir / task_dir_name
            if not task_dir.exists():
                continue
            for pred_file in sorted(task_dir.rglob("*_pred.nii.gz")):
                # arc name relative to results_dir so the zip root contains the task dirs
                arc_name = pred_file.relative_to(results_dir)
                zf.write(str(pred_file), str(arc_name))

    return output_path
