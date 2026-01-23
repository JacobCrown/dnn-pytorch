"""Utilities for segmentation mask preparation."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

_RGB_TO_INT_COEFFS = np.array([256**2, 256, 1], dtype=np.int32)

_WORKER_LOOKUP_TABLE: np.ndarray | None = None
_WORKER_MASK_DTYPE: np.dtype = np.uint8


def find_mask_file(
    masks_dir: Path, image_file: Path, mask_extensions: list[str]
) -> Path | None:
    """Find the corresponding mask file for an image file."""
    for mask_ext in mask_extensions:
        mask_file = masks_dir / f"{image_file.stem}{mask_ext}"
        if mask_file.exists():
            return mask_file

    logger.warning("No mask found for image with exact name match: %s", image_file.stem)
    return None


def build_lookup_table(rgb_to_class_idx: dict[tuple[int, int, int], int]) -> np.ndarray:
    """Create a lookup table for efficient RGB->class conversion."""
    max_class = max(rgb_to_class_idx.values(), default=0)
    if max_class >= 255:
        raise ValueError(
            "Segmentation LUT uses uint8; number of classes must be <= 255."
        )

    lut = np.full(256**3, 255, dtype=np.uint8)
    for color, class_id in rgb_to_class_idx.items():
        color_id = color[0] * 256**2 + color[1] * 256 + color[2]
        lut[color_id] = np.uint8(class_id)
    return lut


def convert_rgb_mask_to_class_mask(
    mask_rgb: np.ndarray, lookup_table: np.ndarray
) -> np.ndarray:
    """Convert an RGB mask to a class index mask using a lookup table."""
    if mask_rgb.ndim != 3 or mask_rgb.shape[-1] != 3:
        raise ValueError(
            "Mask must be an RGB array with shape (H, W, 3) for conversion."
        )
    mask_int32 = mask_rgb.astype(np.int32)
    integer_mask = mask_int32.dot(_RGB_TO_INT_COEFFS)
    return lookup_table[integer_mask].astype(np.int32)


def prepare_masks(
    image_files: list[Path],
    masks_dir: Path,
    mask_extensions: list[str],
    lookup_table: np.ndarray,
    num_workers: int | None = None,
) -> None:
    """Prepare .npy masks for faster I/O."""
    if not masks_dir.exists():
        logger.warning("Masks directory does not exist: %s", masks_dir)
        return

    max_class_idx = int(lookup_table.max())
    mask_dtype = np.uint8 if max_class_idx <= 255 else np.uint16

    use_multiprocessing = len(image_files) > 1000
    if num_workers is not None:
        use_multiprocessing = num_workers > 1

    if not use_multiprocessing:
        for image_file in tqdm(
            image_files,
            desc=f"Preparing mask files for {masks_dir.parent}",
            total=len(image_files),
        ):
            _process_mask(
                image_file,
                masks_dir,
                mask_extensions,
                lookup_table,
                mask_dtype,
            )
        return

    if num_workers is None:
        if len(image_files) > 10000:
            num_workers = min(mp.cpu_count(), 8)
        else:
            num_workers = min(4, mp.cpu_count() // 2)

    num_workers = max(1, num_workers)
    chunksize = max(1, len(image_files) // (num_workers * 4))

    process_args = (
        (image_file, masks_dir, mask_extensions) for image_file in image_files
    )

    with mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(lookup_table, mask_dtype),
    ) as pool:
        iterator = pool.imap_unordered(_process_mask_worker, process_args, chunksize)
        successful = 0
        total = 0
        for _, success in tqdm(
            iterator,
            total=len(image_files),
            desc=f"Preparing mask files for {masks_dir.parent}",
        ):
            total += 1
            if success:
                successful += 1

    logger.info("Prepared %d/%d mask files", successful, total)


def _load_mask_rgb(mask_file: Path) -> np.ndarray:
    mask_rgb = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
    if mask_rgb is None:
        raise RuntimeError(f"Failed to read mask: {mask_file}")

    if mask_rgb.ndim == 3 and mask_rgb.shape[-1] == 3:
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
    elif mask_rgb.ndim == 3 and mask_rgb.shape[-1] == 4:
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError(f"Mask is not RGB: {mask_file} shape={mask_rgb.shape}")

    return mask_rgb


def _process_mask(
    image_file: Path,
    masks_dir: Path,
    mask_extensions: list[str],
    lookup_table: np.ndarray,
    mask_dtype: np.dtype,
) -> bool:
    mask_file = find_mask_file(masks_dir, image_file, mask_extensions)
    if mask_file is None:
        return False

    mask_file_npy = mask_file.with_suffix(".npy")
    if mask_file_npy.exists():
        return True

    try:
        mask_rgb = _load_mask_rgb(mask_file)
        class_mask = convert_rgb_mask_to_class_mask(mask_rgb, lookup_table)
        np.save(mask_file_npy, class_mask.astype(mask_dtype), allow_pickle=False)
        return True
    except Exception as exc:
        logger.warning("Error processing %s: %s", mask_file, exc)
        return False


def _init_worker(lookup_table: np.ndarray, mask_dtype: np.dtype) -> None:
    global _WORKER_LOOKUP_TABLE
    global _WORKER_MASK_DTYPE

    _WORKER_LOOKUP_TABLE = lookup_table
    _WORKER_MASK_DTYPE = mask_dtype
    cv2.setNumThreads(1)
    os.environ["OMP_NUM_THREADS"] = "1"


def _process_mask_worker(args: tuple[Path, Path, list[str]]) -> tuple[str, bool]:
    image_file, masks_dir, mask_extensions = args
    if _WORKER_LOOKUP_TABLE is None:
        return str(image_file), False

    success = _process_mask(
        image_file,
        masks_dir,
        mask_extensions,
        _WORKER_LOOKUP_TABLE,
        _WORKER_MASK_DTYPE,
    )
    return str(image_file), success
