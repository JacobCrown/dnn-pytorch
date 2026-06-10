"""Shared helpers for unit and integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tests.helpers import TEST_ASSETS_DIR

__all__ = [
    "TEST_ASSETS_DIR",
    "create_tiny_classification_dataset",
    "create_tiny_segmentation_dataset",
]


def create_tiny_classification_dataset(
    root: Path,
    *,
    class_names: tuple[str, ...] = ("cat", "dog"),
    samples_per_class: int = 2,
    image_size: tuple[int, int] = (32, 32),
) -> Path:
    """Create a folder-per-class dataset compatible with ``ClassificationDataset``."""
    dataset_root = root / "tiny-cls"
    for split in ("train", "val", "test"):
        for index, class_name in enumerate(class_names):
            class_dir = dataset_root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for sample_index in range(samples_per_class):
                color = (index * 80 + sample_index * 20) % 256
                image = Image.new("RGB", image_size, (color, 128, 255 - color))
                image.save(class_dir / f"{class_name}_{sample_index}.png")
    return dataset_root


def create_tiny_segmentation_dataset(
    root: Path,
    *,
    samples: int = 2,
    image_size: tuple[int, int] = (64, 64),
) -> Path:
    """Create a tiny segmentation dataset with two RGB mask classes."""
    dataset_root = root / "tiny-seg"
    background = np.array([0, 0, 0], dtype=np.uint8)
    foreground = np.array([128, 64, 128], dtype=np.uint8)

    for split in ("train", "val", "test"):
        images_dir = dataset_root / split / "images"
        masks_dir = dataset_root / split / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        for sample_index in range(samples):
            image = Image.new(
                "RGB",
                image_size,
                (sample_index * 40, 90, 180),
            )
            image.save(images_dir / f"sample_{sample_index}.png")

            mask_array = np.zeros((*image_size, 3), dtype=np.uint8)
            mask_array[:, : image_size[1] // 2] = background
            mask_array[:, image_size[1] // 2 :] = foreground
            Image.fromarray(mask_array).save(masks_dir / f"sample_{sample_index}.png")

    return dataset_root
