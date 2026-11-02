"""Tests for SegmentationDataset helpers."""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image

from src.segmentation.config import ClassConfig
from src.segmentation.dataset import SegmentationDataset


def _write_rgb_mask(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8)).save(path)


def test_get_data_distribution_ignores_unmapped_pixels(tmp_path: Path) -> None:
    """Pixels outside class_config become 255 and must not break distribution stats."""
    split = tmp_path / "train"
    images_dir = split / "images"
    masks_dir = split / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    Image.new("RGB", (4, 4), (10, 20, 30)).save(images_dir / "sample_0.png")

    mask = np.zeros((4, 4, 3), dtype=np.uint8)
    mask[:, :2] = (0, 0, 0)  # background
    mask[:, 2:] = (128, 64, 128)  # road
    mask[0, 0] = (1, 2, 3)  # unmapped -> ignore 255
    _write_rgb_mask(masks_dir / "sample_0.png", mask)

    dataset = SegmentationDataset(
        data_root=split,
        transforms=A.Compose([]),
        class_config=[
            ClassConfig(name="background", pixel_value=[0, 0, 0]),
            ClassConfig(name="road", pixel_value=[128, 64, 128]),
        ],
        extensions=[".png"],
        mask_extensions=[".png"],
    )

    distribution = dataset.get_data_distribution(use_cache=False)

    assert [item.class_name for item in distribution] == ["background", "road"]
    assert distribution[0].count == 7  # half of 16 minus the ignored corner
    assert distribution[1].count == 8
    assert distribution[0].count + distribution[1].count == 15
    assert abs(distribution[0].percentage + distribution[1].percentage - 100.0) < 1e-6
