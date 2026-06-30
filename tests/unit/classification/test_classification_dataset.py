"""Unit tests for classification dataset loading."""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import pytest

from src.classification.dataset import ClassificationDataset


def _write_images(class_dir: Path, prefix: str, count: int) -> None:
    class_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (class_dir / f"{prefix}_{index:03d}.png").write_bytes(b"png")


@pytest.fixture
def noop_transforms() -> A.Compose:
    return A.Compose([])


def test_classification_dataset_loads_folder_per_class(
    tmp_path: Path, noop_transforms: A.Compose
) -> None:
    data_root = tmp_path / "train"
    _write_images(data_root / "cat", "cat", 3)
    _write_images(data_root / "dog", "dog", 2)

    dataset = ClassificationDataset(
        data_root=data_root,
        transforms=noop_transforms,
        extensions=[".png"],
    )

    assert dataset.get_class_names() == ["cat", "dog"]
    assert len(dataset) == 5
