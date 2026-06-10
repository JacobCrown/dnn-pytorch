"""Unit tests for classification dataset loading and mixed sampling."""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import pytest

from src.classification.config import ClassificationDataConfig
from src.classification.dataset import (
    ClassificationDataset,
    build_mixed_train_dataset,
)
from src.classification.training_pipeline import resolve_eval_data_root


def _write_images(class_dir: Path, prefix: str, count: int) -> None:
    class_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (class_dir / f"{prefix}_{index:03d}.png").write_bytes(b"png")


def _count_by_label(dataset: ClassificationDataset) -> dict[int, int]:
    counts: dict[int, int] = {}
    for _, label in dataset.samples:
        counts[label] = counts.get(label, 0) + 1
    return counts


def _class_counts(dataset: ClassificationDataset) -> dict[str, int]:
    idx_to_class = dataset.get_idx_to_class()
    by_label = _count_by_label(dataset)
    return {idx_to_class[idx]: count for idx, count in by_label.items()}


@pytest.fixture
def noop_transforms() -> A.Compose:
    return A.Compose([])


def test_mixed_train_real_first_then_synthetic_fill(
    tmp_path: Path, noop_transforms: A.Compose
) -> None:
    real_root = tmp_path / "real"
    synthetic_root = tmp_path / "synthetic"
    _write_images(real_root / "A", "real", 3)
    _write_images(synthetic_root / "A", "syn", 10)

    dataset = build_mixed_train_dataset(
        real_root=real_root,
        synthetic_root=synthetic_root,
        transforms=noop_transforms,
        extensions=[".png"],
        max_samples=5,
        class_mapping=None,
        seed=42,
    )

    assert _class_counts(dataset) == {"A": 5}
    assert dataset.mixed_train_class_counts is not None
    assert dataset.mixed_train_class_counts[0].real_count == 3
    assert dataset.mixed_train_class_counts[0].synthetic_count == 2


def test_mixed_train_keeps_all_real_when_above_max_samples(
    tmp_path: Path, noop_transforms: A.Compose
) -> None:
    real_root = tmp_path / "real"
    synthetic_root = tmp_path / "synthetic"
    _write_images(real_root / "A", "real", 10)
    _write_images(synthetic_root / "A", "syn", 10)

    dataset = build_mixed_train_dataset(
        real_root=real_root,
        synthetic_root=synthetic_root,
        transforms=noop_transforms,
        extensions=[".png"],
        max_samples=5,
        class_mapping=None,
        seed=42,
    )

    assert _class_counts(dataset) == {"A": 10}
    assert dataset.mixed_train_class_counts is not None
    assert dataset.mixed_train_class_counts[0].real_count == 10
    assert dataset.mixed_train_class_counts[0].synthetic_count == 0


def test_mixed_train_without_max_samples_uses_all_sources(
    tmp_path: Path, noop_transforms: A.Compose
) -> None:
    real_root = tmp_path / "real"
    synthetic_root = tmp_path / "synthetic"
    _write_images(real_root / "A", "real", 3)
    _write_images(synthetic_root / "A", "syn", 4)

    dataset = build_mixed_train_dataset(
        real_root=real_root,
        synthetic_root=synthetic_root,
        transforms=noop_transforms,
        extensions=[".png"],
        max_samples=None,
        class_mapping=None,
        seed=42,
    )

    assert _class_counts(dataset) == {"A": 7}
    assert dataset.mixed_train_class_counts is not None
    assert dataset.mixed_train_class_counts[0].real_count == 3
    assert dataset.mixed_train_class_counts[0].synthetic_count == 4


def test_mixed_train_uses_synthetic_when_no_real(
    tmp_path: Path, noop_transforms: A.Compose
) -> None:
    real_root = tmp_path / "real"
    synthetic_root = tmp_path / "synthetic"
    _write_images(real_root / "A", "real", 2)
    _write_images(synthetic_root / "B", "syn", 5)

    dataset = build_mixed_train_dataset(
        real_root=real_root,
        synthetic_root=synthetic_root,
        transforms=noop_transforms,
        extensions=[".png"],
        max_samples=5,
        class_mapping=None,
        seed=42,
    )

    assert dataset.get_class_names() == ["A", "B"]
    assert _class_counts(dataset) == {"A": 2, "B": 5}
    assert dataset.mixed_train_class_counts is not None
    b_counts = next(c for c in dataset.mixed_train_class_counts if c.class_name == "B")
    assert b_counts.real_count == 0
    assert b_counts.synthetic_count == 5


def test_mixed_train_sampling_is_reproducible_with_seed(
    tmp_path: Path, noop_transforms: A.Compose
) -> None:
    real_root = tmp_path / "real"
    synthetic_root = tmp_path / "synthetic"
    _write_images(real_root / "A", "real", 1)
    _write_images(synthetic_root / "A", "syn", 10)

    kwargs = {
        "real_root": real_root,
        "synthetic_root": synthetic_root,
        "transforms": noop_transforms,
        "extensions": [".png"],
        "max_samples": 4,
        "class_mapping": None,
    }
    first = build_mixed_train_dataset(**kwargs, seed=7)
    second = build_mixed_train_dataset(**kwargs, seed=7)
    third = build_mixed_train_dataset(**kwargs, seed=8)

    assert [path for path, _ in first.samples] == [path for path, _ in second.samples]
    assert [path for path, _ in first.samples] != [path for path, _ in third.samples]


def test_val_dataset_uses_real_root_when_configured(
    tmp_path: Path, noop_transforms: A.Compose
) -> None:
    real_root = tmp_path / "real_merged"
    synthetic_root = tmp_path / "synthetic"
    _write_images(real_root / "train" / "A", "train", 2)
    _write_images(real_root / "val" / "A", "val", 4)
    _write_images(synthetic_root / "A", "syn", 10)

    train_dataset = build_mixed_train_dataset(
        real_root=real_root / "train",
        synthetic_root=synthetic_root,
        transforms=noop_transforms,
        extensions=[".png"],
        max_samples=3,
        class_mapping=None,
        seed=42,
    )

    data_config = ClassificationDataConfig(
        root=str(synthetic_root),
        train_dir=".",
        real_root=str(real_root),
        real_train_dir="train",
        val_dir="val",
        max_samples=3,
    )
    val_dataset = ClassificationDataset(
        data_root=resolve_eval_data_root(data_config) / data_config.val_dir,
        transforms=noop_transforms,
        extensions=[".png"],
        max_samples=data_config.max_samples,
        canonical_class_to_idx=train_dataset.get_class_to_idx(),
    )

    assert _class_counts(val_dataset) == {"A": 3}
    assert all("val" in path for path, _ in val_dataset.samples)
