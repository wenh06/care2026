"""
PyTorch Dataset classes for the CARE 2026 Left Atrium challenge.

Provides dataset implementations for all three tasks:
- Task 1: LA scar quantification (LGE-MRI)
- Task 2: LA cavity segmentation (LGE-MRI)
- Task 3: LA multi-structure segmentation (CT)
"""

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch_ecg.cfg import CFG
from torch_ecg.utils.misc import ReprMixin
from tqdm.auto import tqdm

from cfg import CT_TrainCfg, MRI_TrainCfg
from const import CT_HU_MAX, CT_HU_MIN, CT_TARGET_SPACING, DEFAULT_VAL_RATIO
from data_reader import CARE2026_CT, CARE2026_MRI

__all__ = [
    "CARE2026_MRI_Dataset",
    "CARE2026_CT_Dataset",
    "collate_fn_mri",
    "collate_fn_ct",
]


# ---------------------------------------------------------------------------
# MRI Dataset (Tasks 1 & 2 — dual-head, in-memory)
# ---------------------------------------------------------------------------


class CARE2026_MRI_Dataset(Dataset, ReprMixin):
    """In-memory dataset covering both Task 1 (scar) and Task 2 (cavity) MRI data.

    Combines all 190 LGE-MRI records (60 from Task 1 + 130 from Task 2) into a
    single dataset.  Task-1 records carry both ``la_mask`` and ``scar_mask``;
    Task-2 records only carry ``la_mask`` (``has_scar = False``).

    Each volume is bbox-cropped to the LA region, resized to ``MRI_PATCH_SHAPE``
    (256 × 256 × 44), and z-score normalised before being stored in RAM.

    Parameters
    ----------
    db_dir : path-like
        Root directory of the CARE 2026 dataset.
    config : CFG, optional
        Training configuration (defaults to ``MRI_TrainCfg``).
    training : bool, default True
        Whether this is the training split (enables augmentation).
    val_ratio : float, default 0.1
        Fraction of samples held out for validation.
    random_seed : int, default 42
        Reproducible train/val split seed.

    """

    def __init__(
        self,
        db_dir: Union[str, Path],
        config: Optional[CFG] = None,
        training: bool = True,
        val_ratio: float = DEFAULT_VAL_RATIO,
        random_seed: int = 42,
    ) -> None:
        super().__init__()
        self.config = CFG(deepcopy(MRI_TrainCfg))
        if config is not None:
            self.config.update(deepcopy(config))
        self.training = training
        self.db_dir = Path(db_dir).expanduser().resolve()

        self._reader_t1 = CARE2026_MRI(db_dir=self.db_dir, task=1, verbose=0)
        self._reader_t2 = CARE2026_MRI(db_dir=self.db_dir, task=2, verbose=0)

        # Unified record index: (reader, record_name, task_id, has_scar)
        self._index: List[Tuple] = []
        for rec in self._reader_t1.all_records:
            self._index.append((self._reader_t1, rec, 1, True))
        for rec in self._reader_t2.all_records:
            self._index.append((self._reader_t2, rec, 2, False))

        # Stratified train/val split (stratify by task to preserve proportions)
        task_labels = [item[2] for item in self._index]
        idx_all = list(range(len(self._index)))
        idx_train, idx_val = train_test_split(
            idx_all,
            test_size=val_ratio,
            stratify=task_labels,
            random_state=random_seed,
        )
        self._indices = idx_train if training else idx_val

        # Pre-allocate in-memory cache arrays
        patch_shape = tuple(self.config.patch_shape)  # (H, W, D)
        n = len(self._indices)
        self._cache_image = np.zeros((n, 1, *patch_shape), dtype=np.float32)
        self._cache_la_mask = np.zeros((n, *patch_shape), dtype=np.uint8)
        self._cache_scar_mask = np.zeros((n, *patch_shape), dtype=np.uint8)
        self._cache_has_scar = np.zeros(n, dtype=bool)
        self._cache_task = np.zeros(n, dtype=np.int64)
        self._cache_records: List[str] = [""] * n

        self._load_all()

    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """Crop, resize, z-score normalise, and cache every volume."""
        patch_shape = tuple(self.config.patch_shape)
        split_name = "train" if self.training else "val"
        with tqdm(
            enumerate(self._indices),
            total=len(self._indices),
            desc=f"Loading MRI ({split_name})",
            unit="vol",
            dynamic_ncols=True,
        ) as pbar:
            for cache_idx, data_idx in pbar:
                reader, rec, task, has_scar = self._index[data_idx]

                # LA mask is always available (used for bounding box in both tasks)
                la_mask = reader.load_la_ann(rec)
                box = reader.load_ann_box(rec, ann_mask=la_mask)
                (x0, x1), (y0, y1), (z0, z1) = box

                # Crop image and LA mask to the LA bounding box
                image = reader.load_data(rec)[x0:x1, y0:y1, z0:z1]
                la_crop = la_mask[x0:x1, y0:y1, z0:z1]

                # Resize to canonical patch shape
                image = CARE2026_MRI.resample_data(image, patch_shape)
                la_crop = CARE2026_MRI.resample_ann(la_crop, patch_shape)

                # Z-score normalise
                mean, std = float(image.mean()), float(image.std())
                image = (image - mean) / (std + 1e-8)

                self._cache_image[cache_idx, 0] = image
                self._cache_la_mask[cache_idx] = la_crop
                self._cache_task[cache_idx] = task
                self._cache_has_scar[cache_idx] = has_scar
                self._cache_records[cache_idx] = rec

                if has_scar:
                    scar_mask = reader.load_scar_ann(rec)
                    scar_crop = scar_mask[x0:x1, y0:y1, z0:z1]
                    scar_crop = CARE2026_MRI.resample_ann(scar_crop, patch_shape)
                    self._cache_scar_mask[cache_idx] = scar_crop

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict:
        image = self._cache_image[idx].copy()  # (1, H, W, D)
        la_mask = self._cache_la_mask[idx].copy()  # (H, W, D)
        scar_mask = self._cache_scar_mask[idx].copy()
        has_scar = bool(self._cache_has_scar[idx])
        task = int(self._cache_task[idx])
        record = self._cache_records[idx]

        if self.training:
            aug_prob = float(self.config.get("aug_prob", 0.5))
            image, la_mask, scar_mask = _augment_mri(image, la_mask, scar_mask, p=aug_prob)

        return {
            "image": image,
            "la_mask": la_mask,
            "scar_mask": scar_mask,
            "has_scar": has_scar,
            "task": task,
            "record": record,
        }

    @property
    def extra_repr_keys(self) -> List[str]:
        return ["training", "db_dir"]


# ---------------------------------------------------------------------------
# CT Dataset (Task 3 — lazy-load, patch-based, semi-supervised)
# ---------------------------------------------------------------------------


class CARE2026_CT_Dataset(Dataset, ReprMixin):
    """Lazy-loading patch-based dataset for CT Task 3 (semi-supervised segmentation).

    Labeled and unlabeled records can be mixed.  During training a random
    foreground-biased 128³ patch is extracted; during validation the centre
    crop is returned.

    Parameters
    ----------
    db_dir : path-like
        Root directory of the CARE 2026 dataset.
    config : CFG, optional
        Training configuration (defaults to ``CT_TrainCfg``).
    training : bool, default True
        Whether this is the training split (enables augmentation + random patches).
    labeled : bool or None, default None
        ``True`` → labeled records only.  ``False`` → unlabeled records only.
        ``None`` → all records (labeled split applies only to labeled records).
    val_ratio : float, default 0.1
        Fraction of **labeled** samples held out for validation.
    random_seed : int, default 42
        Reproducible seed for train/val split and patch sampling.

    """

    def __init__(
        self,
        db_dir: Union[str, Path],
        config: Optional[CFG] = None,
        training: bool = True,
        labeled: Optional[bool] = None,
        val_ratio: float = DEFAULT_VAL_RATIO,
        random_seed: int = 42,
    ) -> None:
        super().__init__()
        self.config = CFG(deepcopy(CT_TrainCfg))
        if config is not None:
            self.config.update(deepcopy(config))
        self.training = training
        self.labeled = labeled
        self.db_dir = Path(db_dir).expanduser().resolve()

        self._reader = CARE2026_CT(db_dir=self.db_dir, verbose=0)

        labeled_recs = self._reader.labeled_records
        unlabeled_recs = self._reader.unlabeled_records

        if labeled is True:
            idx_train, idx_val = train_test_split(
                labeled_recs,
                test_size=val_ratio,
                random_state=random_seed,
            )
            self._records: List[str] = idx_train if training else idx_val
            self._is_labeled_map: Dict[str, bool] = {r: True for r in labeled_recs}
        elif labeled is False:
            # Unlabeled records are used for training only (no validation split)
            self._records = unlabeled_recs
            self._is_labeled_map = {r: False for r in unlabeled_recs}
        else:
            # All records: split labeled ones, include all unlabeled in training
            idx_train_l, idx_val_l = train_test_split(
                labeled_recs,
                test_size=val_ratio,
                random_state=random_seed,
            )
            if training:
                self._records = idx_train_l + unlabeled_recs
            else:
                self._records = idx_val_l
            self._is_labeled_map = {
                **{r: True for r in labeled_recs},
                **{r: False for r in unlabeled_recs},
            }

        self._patch_size: int = int(self.config.patch_size)

        # Per-record lazy cache: stores (image, mask_or_None) after preprocessing
        self._cache: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict:
        rec = self._records[idx]
        is_labeled = self._is_labeled_map[rec]

        image, mask = self._get_preprocessed(rec)

        ps = self._patch_size
        if self.training:
            aug_prob = float(self.config.get("aug_prob", 0.5))
            image_patch, mask_patch = _random_patch(image, mask, ps)
            image_patch, mask_patch = _augment_ct(image_patch, mask_patch, p=aug_prob)
        else:
            image_patch, mask_patch = _center_patch(image, mask, ps)

        out: Dict = {
            "image": image_patch[np.newaxis].astype(np.float32),  # (1, ps, ps, ps)
            "is_labeled": is_labeled,
            "record": rec,
        }
        if mask_patch is not None:
            out["mask"] = mask_patch  # (ps, ps, ps)
        return out

    # ------------------------------------------------------------------

    def _get_preprocessed(self, rec: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (image, mask) from in-memory cache, preprocessing on first access."""
        if rec not in self._cache:
            self._cache[rec] = self._preprocess_ct(rec)
        return self._cache[rec]

    def _preprocess_ct(self, rec: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """HU clip, normalise, and resample CT volume to isotropic 0.5 mm voxels."""
        nii = nib.load(str(self._reader.get_data_path(rec)))
        image = nii.get_fdata().astype(np.float32)
        zooms = np.array(nii.header.get_zooms()[:3], dtype=np.float64)  # (sx, sy, sz) mm

        # HU clip then normalise to [0, 1]
        image = np.clip(image, CT_HU_MIN, CT_HU_MAX)
        image = (image - CT_HU_MIN) / (CT_HU_MAX - CT_HU_MIN)

        # Compute target shape for isotropic resampling
        native_shape = np.array(image.shape[:3], dtype=np.float64)
        target_spacing = np.array(CT_TARGET_SPACING, dtype=np.float64)
        target_shape = tuple(int(np.round(native_shape[i] * zooms[i] / target_spacing[i])) for i in range(3))

        image = CARE2026_CT.resample_data(image, target_shape)

        mask: Optional[np.ndarray] = None
        if self._is_labeled_map.get(rec, False):
            mask = self._reader.load_ann(rec)
            mask = CARE2026_CT.resample_ann(mask, target_shape)

        return image, mask

    @property
    def extra_repr_keys(self) -> List[str]:
        return ["training", "labeled", "db_dir"]


# ---------------------------------------------------------------------------
# Spatial helpers (module-level, used by both dataset classes)
# ---------------------------------------------------------------------------


def _pad_to_size(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    ps: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Zero-pad *image* (and *mask*) to ``ps × ps × ps`` if any dim is smaller."""
    H, W, D = image.shape[:3]
    pad = [(0, max(ps - H, 0)), (0, max(ps - W, 0)), (0, max(ps - D, 0))]
    if any(p[1] > 0 for p in pad):
        image = np.pad(image, pad, mode="constant", constant_values=0.0)
        if mask is not None:
            mask = np.pad(mask, pad, mode="constant", constant_values=0)
    return image, mask


def _random_patch(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    ps: int,
    fg_bias: float = 0.5,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract a random ``ps³`` patch with foreground-biased centre sampling."""
    H, W, D = image.shape[:3]
    rng = np.random.default_rng()

    if mask is not None and rng.random() < fg_bias and mask.max() > 0:
        fg_coords = np.argwhere(mask > 0)
        centre = fg_coords[rng.integers(len(fg_coords))]
    else:
        centre = np.array([rng.integers(max(H, 1)), rng.integers(max(W, 1)), rng.integers(max(D, 1))])

    starts = np.array([np.clip(int(centre[i]) - ps // 2, 0, max(image.shape[i] - ps, 0)) for i in range(3)])
    x0, y0, z0 = starts
    img_patch = image[x0 : x0 + ps, y0 : y0 + ps, z0 : z0 + ps]
    msk_patch = mask[x0 : x0 + ps, y0 : y0 + ps, z0 : z0 + ps] if mask is not None else None

    img_patch, msk_patch = _pad_to_size(img_patch, msk_patch, ps)
    return img_patch, msk_patch


def _center_patch(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    ps: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract the centre crop of size ``ps³`` (used for validation)."""
    H, W, D = image.shape[:3]
    x0, y0, z0 = max((H - ps) // 2, 0), max((W - ps) // 2, 0), max((D - ps) // 2, 0)
    img_patch = image[x0 : x0 + ps, y0 : y0 + ps, z0 : z0 + ps]
    msk_patch = mask[x0 : x0 + ps, y0 : y0 + ps, z0 : z0 + ps] if mask is not None else None
    img_patch, msk_patch = _pad_to_size(img_patch, msk_patch, ps)
    return img_patch, msk_patch


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------


def _augment_mri(
    image: np.ndarray,
    la_mask: np.ndarray,
    scar_mask: np.ndarray,
    p: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random augmentation for a single MRI sample.

    Applied independently with probability *p*:
    axis flips, 90° in-plane rotations, gamma correction, additive noise.
    """
    rng = np.random.default_rng()

    # Random flips along each spatial axis (axis 0-2 of mask → axis 1-3 of image)
    for ax in range(3):
        if rng.random() < p:
            image = np.flip(image, axis=ax + 1)
            la_mask = np.flip(la_mask, axis=ax)
            scar_mask = np.flip(scar_mask, axis=ax)

    # Random 90° rotation in the axial (x–y) plane
    if rng.random() < p:
        k = int(rng.integers(1, 4))
        image = np.rot90(image, k=k, axes=(1, 2))
        la_mask = np.rot90(la_mask, k=k, axes=(0, 1))
        scar_mask = np.rot90(scar_mask, k=k, axes=(0, 1))

    # Gamma correction (intensity augmentation on image channel only)
    if rng.random() < p:
        gamma = float(rng.uniform(0.7, 1.5))
        img_min = float(image.min())
        img_range = float(image.max()) - img_min
        if img_range > 0:
            image_norm = (image - img_min) / img_range
            image = (image_norm**gamma) * img_range + img_min

    # Additive Gaussian noise
    if rng.random() < p:
        noise_std = float(rng.uniform(0.0, 0.1))
        image = image + rng.standard_normal(image.shape).astype(np.float32) * noise_std

    # np.flip / np.rot90 return views; make contiguous for DataLoader safety
    image = np.ascontiguousarray(image)
    la_mask = np.ascontiguousarray(la_mask)
    scar_mask = np.ascontiguousarray(scar_mask)

    return image, la_mask, scar_mask


def _augment_ct(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    p: float = 0.5,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Random augmentation for a single CT patch."""
    rng = np.random.default_rng()

    # Random flips
    for ax in range(3):
        if rng.random() < p:
            image = np.flip(image, axis=ax)
            if mask is not None:
                mask = np.flip(mask, axis=ax)

    # Random 90° rotation in the axial plane
    if rng.random() < p:
        k = int(rng.integers(1, 4))
        image = np.rot90(image, k=k, axes=(0, 1))
        if mask is not None:
            mask = np.rot90(mask, k=k, axes=(0, 1))

    # Intensity scaling (keep within [0, 1])
    if rng.random() < p:
        scale = float(rng.uniform(0.9, 1.1))
        image = np.clip(image * scale, 0.0, 1.0)

    # Gaussian noise
    if rng.random() < p:
        noise_std = float(rng.uniform(0.0, 0.05))
        image = np.clip(
            image + rng.standard_normal(image.shape).astype(np.float32) * noise_std,
            0.0,
            1.0,
        )

    image = np.ascontiguousarray(image)
    if mask is not None:
        mask = np.ascontiguousarray(mask)
    return image, mask


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------


def collate_fn_mri(batch: List[Dict]) -> Dict:
    """Collate MRI dataset samples into a batch.

    Returns
    -------
    dict
        ``image``     : float32 tensor ``(B, 1, H, W, D)``
        ``la_mask``   : int64 tensor   ``(B, H, W, D)``
        ``scar_mask`` : int64 tensor   ``(B, H, W, D)``
        ``has_scar``  : bool tensor    ``(B,)``
        ``task``      : int64 tensor   ``(B,)``
        ``record``    : list of str, length ``B``
    """
    images = torch.from_numpy(np.stack([s["image"] for s in batch]))
    la_masks = torch.from_numpy(np.stack([s["la_mask"] for s in batch])).long()
    scar_masks = torch.from_numpy(np.stack([s["scar_mask"] for s in batch])).long()
    has_scar = torch.tensor([s["has_scar"] for s in batch], dtype=torch.bool)
    tasks = torch.tensor([s["task"] for s in batch], dtype=torch.int64)
    records = [s["record"] for s in batch]
    return {
        "image": images,
        "la_mask": la_masks,
        "scar_mask": scar_masks,
        "has_scar": has_scar,
        "task": tasks,
        "record": records,
    }


def collate_fn_ct(batch: List[Dict]) -> Dict:
    """Collate CT dataset samples into a batch.

    Unlabeled samples carry no ``mask`` key; they are padded with zeros so
    the batch tensor is always present (avoids branching in the CPS trainer).

    Returns
    -------
    dict
        ``image``      : float32 tensor ``(B, 1, ps, ps, ps)``
        ``mask``       : int64 tensor   ``(B, ps, ps, ps)`` — zeros for unlabeled
        ``is_labeled`` : bool tensor    ``(B,)``
        ``record``     : list of str, length ``B``
    """
    images = torch.from_numpy(np.stack([s["image"] for s in batch]))
    is_labeled = torch.tensor([s["is_labeled"] for s in batch], dtype=torch.bool)
    records = [s["record"] for s in batch]

    ps = images.shape[-1]
    masks_np = []
    for s in batch:
        masks_np.append(s["mask"] if "mask" in s else np.zeros((ps, ps, ps), dtype=np.uint8))
    masks = torch.from_numpy(np.stack(masks_np)).long()

    return {
        "image": images,
        "mask": masks,
        "is_labeled": is_labeled,
        "record": records,
    }
