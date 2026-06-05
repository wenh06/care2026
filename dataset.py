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

from cfg import CT_TrainCfg, MRI_Stage1_TrainCfg, MRI_Stage2_TrainCfg
from const import (
    CT_HU_MAX,
    CT_HU_MIN,
    CT_TARGET_SPACING,
    DEFAULT_VAL_RATIO,
    MRI_CANONICAL_SHAPE,
    MRI_STAGE1_SHAPE,
    MRI_STAGE2_CACHE_SHAPE,
    MRI_STAGE2_CENTROID_JITTER,
    MRI_STAGE2_CROP_SHAPE,
)
from data_reader import CARE2026_CT, CARE2026_MRI
from utils.mclahe import mclahe as _mclahe

__all__ = [
    "CARE2026_MRI_Stage1_Dataset",
    "CARE2026_MRI_Stage2_Dataset",
    "CARE2026_CT_Dataset",
    "collate_fn_mri_stage1",
    "collate_fn_mri",
    "collate_fn_ct",
]


# ---------------------------------------------------------------------------
# Shared helpers for MRI datasets
# ---------------------------------------------------------------------------


def _build_mri_index(reader_t1, reader_t2):
    """Build unified record index: list of (reader, rec, task_id, has_scar)."""
    index = []
    for rec in reader_t1.all_records:
        index.append((reader_t1, rec, 1, True))
    for rec in reader_t2.all_records:
        index.append((reader_t2, rec, 2, False))
    return index


def _mri_train_val_split(index, val_ratio, random_seed):
    """Stratified train/val split on task label."""
    task_labels = [item[2] for item in index]
    idx_all = list(range(len(index)))
    idx_train, idx_val = train_test_split(
        idx_all,
        test_size=val_ratio,
        stratify=task_labels,
        random_state=random_seed,
    )
    return idx_train, idx_val


# ---------------------------------------------------------------------------
# MRI Stage 1 Dataset — coarse LA localisation (Tasks 1 & 2)
# ---------------------------------------------------------------------------


class CARE2026_MRI_Stage1_Dataset(Dataset, ReprMixin):
    """In-memory dataset for Stage 1 coarse LA localisation.

    All 190 LGE-MRI records are:
    1. Loaded at native resolution.
    2. Resampled to ``MRI_CANONICAL_SHAPE`` (576 × 576 × 44).
    3. Downsampled to ``MRI_STAGE1_SHAPE`` (144 × 144 × 44).
    4. Z-score normalised, then cached in RAM as ``(1, 144, 144, 44)`` float32.

    Labels: binary LA mask at Stage 1 resolution.

    Parameters
    ----------
    db_dir : path-like
        Root directory of the CARE 2026 dataset.
    config : CFG, optional
        Training configuration (defaults to ``MRI_Stage1_TrainCfg``).
    training : bool, default True
        Whether this is the training split (enables augmentation).
    val_ratio : float, default DEFAULT_VAL_RATIO
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
        self.config = CFG(deepcopy(MRI_Stage1_TrainCfg))
        if config is not None:
            self.config.update(deepcopy(config))
        self.training = training
        self.db_dir = Path(db_dir).expanduser().resolve()

        self._reader_t1 = CARE2026_MRI(db_dir=self.db_dir, task=1, verbose=0)
        self._reader_t2 = CARE2026_MRI(db_dir=self.db_dir, task=2, verbose=0)
        self._index = _build_mri_index(self._reader_t1, self._reader_t2)

        idx_train, idx_val = _mri_train_val_split(self._index, val_ratio, random_seed)
        self._indices = idx_train if training else idx_val

        canonical_shape = tuple(self.config.get("canonical_shape", MRI_CANONICAL_SHAPE))
        stage1_shape = tuple(self.config.get("patch_shape", MRI_STAGE1_SHAPE))
        n = len(self._indices)
        self._cache_image = np.zeros((n, 1, *stage1_shape), dtype=np.float32)
        self._cache_la_mask = np.zeros((n, *stage1_shape), dtype=np.uint8)
        self._cache_records: List[str] = [""] * n

        self._canonical_shape = canonical_shape
        self._stage1_shape = stage1_shape
        self._load_all()

    def _load_all(self) -> None:
        split_name = "train" if self.training else "val"
        with tqdm(
            enumerate(self._indices),
            total=len(self._indices),
            desc=f"Loading MRI Stage1 ({split_name})",
            unit="vol",
            dynamic_ncols=True,
        ) as pbar:
            for cache_idx, data_idx in pbar:
                reader, rec, _task, _has_scar = self._index[data_idx]
                image = reader.load_data(rec)
                la_mask = reader.load_la_ann(rec)

                # 1. Resample to canonical shape
                image = CARE2026_MRI.resample_data(image, self._canonical_shape)
                la_mask = CARE2026_MRI.resample_ann(la_mask, self._canonical_shape)

                # 1.5. CLAHE on the full canonical image (before any downsampling)
                if self.config.get("apply_mclahe", False):
                    image = _mclahe(image)

                # 2. Downsample to Stage 1 shape
                image = CARE2026_MRI.resample_data(image, self._stage1_shape)
                la_mask = CARE2026_MRI.resample_ann(la_mask, self._stage1_shape)

                # 3. Z-score normalise
                mean, std = float(image.mean()), float(image.std())
                image = (image - mean) / (std + 1e-8)

                self._cache_image[cache_idx, 0] = image
                self._cache_la_mask[cache_idx] = la_mask
                self._cache_records[cache_idx] = rec

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict:
        image = self._cache_image[idx].copy()  # (1, H, W, D)
        la_mask = self._cache_la_mask[idx].copy()
        record = self._cache_records[idx]

        if self.training:
            p = float(self.config.get("aug_prob", 0.5))
            image, la_mask, _ = _augment_mri(image, la_mask, np.zeros_like(la_mask), p=p)

        return {"image": image, "la_mask": la_mask, "record": record}

    @property
    def extra_repr_keys(self) -> List[str]:
        return ["training", "db_dir"]


# ---------------------------------------------------------------------------
# MRI Stage 2 Dataset — fine LA+scar segmentation (Tasks 1 & 2)
# ---------------------------------------------------------------------------


class CARE2026_MRI_Stage2_Dataset(Dataset, ReprMixin):
    """In-memory dataset for Stage 2 fine segmentation.

    Each volume is:
    1. Resampled to ``MRI_CANONICAL_SHAPE`` (576 × 576 × 44).
    2. The GT LA centroid is computed in canonical space.
    3. A generous region ``MRI_STAGE2_CACHE_SHAPE`` (320 × 320 × 44) centred on
       the centroid is cropped and cached (with zero-padding at image borders).
    4. At ``__getitem__`` time a random spatial jitter ``MRI_STAGE2_CENTROID_JITTER``
       is applied to simulate Stage 1 prediction noise, then a sub-crop of size
       ``MRI_STAGE2_CROP_SHAPE`` (256 × 256 × 44) is returned.

    The jitter margin = (cache_H - crop_H) / 2 = (320 - 256) / 2 = 32, which
    equals ``MRI_STAGE2_CENTROID_JITTER[0]``.

    Task-1 records carry both ``la_mask`` and ``scar_mask``;
    Task-2 records only carry ``la_mask`` (``has_scar = False``).

    Parameters
    ----------
    db_dir : path-like
        Root directory of the CARE 2026 dataset.
    config : CFG, optional
        Training configuration (defaults to ``MRI_Stage2_TrainCfg``).
    training : bool, default True
        Whether this is the training split (enables augmentation + jitter).
    val_ratio : float, default DEFAULT_VAL_RATIO
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
        no_scar_proportion: float = 0.3,
    ) -> None:
        super().__init__()
        self.config = CFG(deepcopy(MRI_Stage2_TrainCfg))
        if config is not None:
            self.config.update(deepcopy(config))
        self.training = training
        self.db_dir = Path(db_dir).expanduser().resolve()

        self._reader_t1 = CARE2026_MRI(db_dir=self.db_dir, task=1, verbose=0)
        self._reader_t2 = CARE2026_MRI(db_dir=self.db_dir, task=2, verbose=0)
        self._index = _build_mri_index(self._reader_t1, self._reader_t2)

        idx_train, idx_val = _mri_train_val_split(self._index, val_ratio, random_seed)
        self._indices = idx_train if training else idx_val

        # Subsample no-scar (Task 2) records in the training split.
        # Scar is only present in Task 1 records (~32 % of all samples).
        # Including all 130 no-scar samples dilutes the scar signal; we
        # keep a small fraction as hard negatives to teach the model
        # what healthy tissue looks like.
        if training:
            rng = np.random.default_rng(random_seed)
            scar_idx = [i for i in self._indices if self._index[i][3]]  # has_scar
            no_scar_idx = [i for i in self._indices if not self._index[i][3]]
            n_keep = max(0, int(len(no_scar_idx) * no_scar_proportion))
            no_scar_kept = sorted(rng.choice(no_scar_idx, size=n_keep, replace=False).tolist()) if n_keep > 0 else []
            self._indices = scar_idx + no_scar_kept

        self._canonical_shape = tuple(self.config.get("canonical_shape", MRI_CANONICAL_SHAPE))
        self._cache_shape = tuple(self.config.get("cache_shape", MRI_STAGE2_CACHE_SHAPE))
        self._crop_shape = tuple(self.config.get("patch_shape", MRI_STAGE2_CROP_SHAPE))
        self._jitter = tuple(self.config.get("centroid_jitter", MRI_STAGE2_CENTROID_JITTER))

        n = len(self._indices)
        self._cache_image = np.zeros((n, 1, *self._cache_shape), dtype=np.float32)
        self._cache_la_mask = np.zeros((n, *self._cache_shape), dtype=np.uint8)
        self._cache_scar_mask = np.zeros((n, *self._cache_shape), dtype=np.uint8)
        self._cache_has_scar = np.zeros(n, dtype=bool)
        self._cache_task = np.zeros(n, dtype=np.int64)
        self._cache_records: List[str] = [""] * n

        self._load_all()

    def _centroid(self, mask: np.ndarray) -> Tuple[int, int, int]:
        """Return the integer centroid of foreground voxels.  Falls back to centre."""
        fg = np.argwhere(mask > 0)
        if len(fg) == 0:
            return tuple(s // 2 for s in mask.shape)
        return tuple(int(fg[:, i].mean()) for i in range(3))

    def _load_all(self) -> None:
        split_name = "train" if self.training else "val"
        with tqdm(
            enumerate(self._indices),
            total=len(self._indices),
            desc=f"Loading MRI Stage2 ({split_name})",
            unit="vol",
            dynamic_ncols=True,
        ) as pbar:
            for cache_idx, data_idx in pbar:
                reader, rec, task, has_scar = self._index[data_idx]

                image = reader.load_data(rec)
                la_mask_native = reader.load_la_ann(rec)

                # 1. Resample to canonical shape
                image = CARE2026_MRI.resample_data(image, self._canonical_shape)
                la_mask = CARE2026_MRI.resample_ann(la_mask_native, self._canonical_shape)

                if has_scar:
                    scar_mask_native = reader.load_scar_ann(rec)
                    scar_mask = CARE2026_MRI.resample_ann(scar_mask_native, self._canonical_shape)
                else:
                    scar_mask = np.zeros_like(la_mask)

                # 1.5. CLAHE on the full canonical image (before centroid crop)
                if self.config.get("apply_mclahe", False):
                    image = _mclahe(image)

                # 2. Find LA centroid in canonical space
                cx, cy, cz = self._centroid(la_mask)

                # 3. Crop generous cache region around centroid
                img_cache, la_cache, scar_cache = _centroid_crop(
                    image,
                    la_mask,
                    scar_mask,
                    centroid=(cx, cy, cz),
                    crop_shape=self._cache_shape,
                )

                # 4. Z-score normalise the image cache
                mean, std = float(img_cache.mean()), float(img_cache.std())
                img_cache = (img_cache - mean) / (std + 1e-8)

                self._cache_image[cache_idx, 0] = img_cache
                self._cache_la_mask[cache_idx] = la_cache
                self._cache_scar_mask[cache_idx] = scar_cache
                self._cache_has_scar[cache_idx] = has_scar
                self._cache_task[cache_idx] = task
                self._cache_records[cache_idx] = rec

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Dict:
        image = self._cache_image[idx].copy()  # (1, cH, cW, cD)
        la_mask = self._cache_la_mask[idx].copy()
        scar_mask = self._cache_scar_mask[idx].copy()
        has_scar = bool(self._cache_has_scar[idx])
        task = int(self._cache_task[idx])
        record = self._cache_records[idx]

        if self.training:
            # Simulate Stage 1 prediction error: apply random jitter then sub-crop
            jitter = [0, 0, 0]
            for i in range(3):
                max_j = int(self._jitter[i])
                jitter[i] = int(np.random.randint(-max_j, max_j + 1)) if max_j > 0 else 0

            image, la_mask, scar_mask = _jitter_and_crop(
                image,
                la_mask,
                scar_mask,
                jitter=jitter,
                crop_shape=self._crop_shape,
            )
            p = float(self.config.get("aug_prob", 0.5))
            image, la_mask, scar_mask = _augment_mri(image, la_mask, scar_mask, p=p)
            crop_hw = int(self.config.get("train_crop_hw", 0))
            if crop_hw > 0:
                image, la_mask, scar_mask = _crop_hw_train(image, la_mask, scar_mask, crop_hw)
        else:
            # Validation: take the exact centre sub-crop (no jitter)
            image, la_mask, scar_mask = _jitter_and_crop(
                image,
                la_mask,
                scar_mask,
                jitter=[0, 0, 0],
                crop_shape=self._crop_shape,
            )

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


# Backward-compatibility alias


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


def _centroid_crop(
    image: np.ndarray,
    la_mask: np.ndarray,
    scar_mask: np.ndarray,
    centroid: Tuple[int, int, int],
    crop_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop a fixed-size region centred on *centroid* from the canonical volume.

    Parameters
    ----------
    image : (H, W, D) float32 array (canonical space, not yet normalised)
    la_mask : (H, W, D) uint8 array
    scar_mask : (H, W, D) uint8 array
    centroid : (cx, cy, cz) voxel coordinates in the canonical volume
    crop_shape : (cH, cW, cD) target crop size

    Returns
    -------
    Tuple of (image_crop, la_crop, scar_crop), each of shape *crop_shape*.
    """
    H, W, D = image.shape
    cH, cW, cD = crop_shape
    cx, cy, cz = centroid

    def _clamp(start, size, max_dim):
        return int(np.clip(start, 0, max(max_dim - size, 0)))

    x0 = _clamp(cx - cH // 2, cH, H)
    y0 = _clamp(cy - cW // 2, cW, W)
    z0 = _clamp(cz - cD // 2, cD, D)

    img_crop = image[x0 : x0 + cH, y0 : y0 + cW, z0 : z0 + cD]
    la_crop = la_mask[x0 : x0 + cH, y0 : y0 + cW, z0 : z0 + cD]
    scar_crop = scar_mask[x0 : x0 + cH, y0 : y0 + cW, z0 : z0 + cD]

    # Pad to exact crop_shape if the centroid is too close to a border
    for i, (arr, sz) in enumerate(zip([img_crop, la_crop, scar_crop], [cH, cW, cD])):
        pass  # padding handled below
    pad_x = max(0, cH - img_crop.shape[0])
    pad_y = max(0, cW - img_crop.shape[1])
    pad_z = max(0, cD - img_crop.shape[2])
    if pad_x > 0 or pad_y > 0 or pad_z > 0:
        img_crop = np.pad(img_crop, [(0, pad_x), (0, pad_y), (0, pad_z)])
        la_crop = np.pad(la_crop, [(0, pad_x), (0, pad_y), (0, pad_z)])
        scar_crop = np.pad(scar_crop, [(0, pad_x), (0, pad_y), (0, pad_z)])

    return img_crop, la_crop, scar_crop


def _jitter_and_crop(
    image: np.ndarray,
    la_mask: np.ndarray,
    scar_mask: np.ndarray,
    jitter: List[int],
    crop_shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply jitter offset and extract a sub-crop from the cache arrays.

    The cache arrays have shape ``(1, cH, cW, cD)`` (image) / ``(cH, cW, cD)``
    (masks).  The crop is taken from the centre of the cache with an added jitter.

    Parameters
    ----------
    image : (1, cH, cW, cD) float32
    la_mask : (cH, cW, cD) uint8
    scar_mask : (cH, cW, cD) uint8
    jitter : [dx, dy, dz] integers (may be negative)
    crop_shape : (H, W, D) target output size

    Returns
    -------
    Tuple (image_crop, la_crop, scar_crop) with image shape (1, H, W, D).
    """
    _, cH, cW, cD = image.shape
    tH, tW, tD = crop_shape

    def _clamp(offset, cache_dim, target_dim):
        start = (cache_dim - target_dim) // 2 + offset
        return int(np.clip(start, 0, max(cache_dim - target_dim, 0)))

    x0 = _clamp(jitter[0], cH, tH)
    y0 = _clamp(jitter[1], cW, tW)
    z0 = _clamp(jitter[2], cD, tD)

    img_out = image[:, x0 : x0 + tH, y0 : y0 + tW, z0 : z0 + tD]
    la_out = la_mask[x0 : x0 + tH, y0 : y0 + tW, z0 : z0 + tD]
    scar_out = scar_mask[x0 : x0 + tH, y0 : y0 + tW, z0 : z0 + tD]

    # Pad to exact shape if needed (e.g. near boundaries)
    px = max(0, tH - img_out.shape[1])
    py = max(0, tW - img_out.shape[2])
    pz = max(0, tD - img_out.shape[3])
    if px > 0 or py > 0 or pz > 0:
        img_out = np.pad(img_out, [(0, 0), (0, px), (0, py), (0, pz)])
        la_out = np.pad(la_out, [(0, px), (0, py), (0, pz)])
        scar_out = np.pad(scar_out, [(0, px), (0, py), (0, pz)])

    return img_out, la_out, scar_out


def _crop_hw_train(
    image: np.ndarray,
    la_mask: np.ndarray,
    scar_mask: np.ndarray,
    crop_hw: int,
    fg_bias: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random H×W crop with foreground-biased centre sampling for MRI training.

    Parameters
    ----------
    image : (1, H, W, D) float32 array
    la_mask : (H, W, D) uint8 array
    scar_mask : (H, W, D) uint8 array
    crop_hw : target crop size (same for H and W)
    fg_bias : probability of centering the crop on a foreground voxel
    """
    _, H, W, D = image.shape
    rng = np.random.default_rng()

    if la_mask.max() > 0 and rng.random() < fg_bias:
        fg = np.argwhere(la_mask > 0)
        c = fg[rng.integers(len(fg))]
        ch, cw = int(c[0]), int(c[1])
    else:
        ch = rng.integers(max(H, 1))
        cw = rng.integers(max(W, 1))

    h0 = int(np.clip(ch - crop_hw // 2, 0, max(H - crop_hw, 0)))
    w0 = int(np.clip(cw - crop_hw // 2, 0, max(W - crop_hw, 0)))

    img_crop = image[:, h0 : h0 + crop_hw, w0 : w0 + crop_hw, :]
    la_crop = la_mask[h0 : h0 + crop_hw, w0 : w0 + crop_hw, :]
    scar_crop = scar_mask[h0 : h0 + crop_hw, w0 : w0 + crop_hw, :]

    # Pad if any dimension is smaller than crop_hw
    ph = max(0, crop_hw - img_crop.shape[1])
    pw = max(0, crop_hw - img_crop.shape[2])
    if ph > 0 or pw > 0:
        img_crop = np.pad(img_crop, [(0, 0), (0, ph), (0, pw), (0, 0)])
        la_crop = np.pad(la_crop, [(0, ph), (0, pw), (0, 0)])
        scar_crop = np.pad(scar_crop, [(0, ph), (0, pw), (0, 0)])

    return img_crop, la_crop, scar_crop


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


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------


def collate_fn_mri_stage1(batch: List[Dict]) -> Dict:
    """Collate Stage 1 MRI dataset samples into a batch.

    Returns
    -------
    dict
        ``image``   : float32 tensor ``(B, 1, H, W, D)``
        ``la_mask`` : int64 tensor   ``(B, H, W, D)``
        ``record``  : list of str, length ``B``
    """
    images = torch.from_numpy(np.stack([s["image"] for s in batch]))
    la_masks = torch.from_numpy(np.stack([s["la_mask"] for s in batch])).long()
    records = [s["record"] for s in batch]
    return {"image": images, "la_mask": la_masks, "record": records}


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
