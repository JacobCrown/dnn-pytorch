"""Segmentation console UX should match classification (Rich, no tqdm, class list once)."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import albumentations as A
import pytest
from PIL import Image

from src.segmentation.config import ClassConfig
from src.segmentation.dataset import DataDistribution, SegmentationDataset
from src.segmentation.training_pipeline import (
    SegmentationDataPipeline,
    log_data_pipeline_summary,
)

_SEGMENTATION_SRC = Path(__file__).resolve().parents[3] / "src" / "segmentation"


def test_segmentation_modules_do_not_import_tqdm() -> None:
    offenders: list[str] = []
    for path in sorted(_SEGMENTATION_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name == "tqdm" or alias.name.startswith("tqdm.") for alias in node.names):
                    offenders.append(str(path.relative_to(_SEGMENTATION_SRC.parent.parent)))
            elif isinstance(node, ast.ImportFrom) and node.module == "tqdm":
                offenders.append(str(path.relative_to(_SEGMENTATION_SRC.parent.parent)))
    assert offenders == []


def test_dataset_does_not_log_class_list_at_info(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    split = tmp_path / "train"
    (split / "images").mkdir(parents=True)
    (split / "masks").mkdir(parents=True)
    Image.new("RGB", (4, 4), (10, 20, 30)).save(split / "images" / "sample_0.png")
    Image.new("RGB", (4, 4), (0, 0, 0)).save(split / "masks" / "sample_0.png")

    with caplog.at_level(logging.INFO, logger="src.segmentation.dataset"):
        SegmentationDataset(
            data_root=split,
            transforms=A.Compose([]),
            class_config=[
                ClassConfig(name="background", pixel_value=[0, 0, 0]),
                ClassConfig(name="road", pixel_value=[128, 64, 128]),
            ],
            extensions=[".png"],
            mask_extensions=[".png"],
        )

    assert not any("Final classes" in record.getMessage() for record in caplog.records)


def test_log_data_pipeline_summary_logs_class_names_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _FakeDataset:
        def __init__(self, size: int) -> None:
            self._size = size

        def get_class_names(self) -> list[str]:
            return ["background", "road"]

        def get_data_distribution(self) -> list[DataDistribution]:
            return [
                DataDistribution(class_name="background", count=10, percentage=50.0),
                DataDistribution(class_name="road", count=10, percentage=50.0),
            ]

        def __len__(self) -> int:
            return self._size

    train_dataset = _FakeDataset(4)
    pipeline = SegmentationDataPipeline(
        train_dataset=train_dataset,  # type: ignore[arg-type]
        val_dataset=_FakeDataset(2),  # type: ignore[arg-type]
        test_dataset=_FakeDataset(1),  # type: ignore[arg-type]
        train_loader=None,  # type: ignore[arg-type]
        val_loader=None,  # type: ignore[arg-type]
        test_loader=object(),  # type: ignore[arg-type]
    )

    with caplog.at_level(logging.INFO):
        log_data_pipeline_summary(pipeline)

    class_name_logs = [
        record.getMessage()
        for record in caplog.records
        if "Final class names" in record.getMessage()
    ]
    assert class_name_logs == ["Final class names: 2 classes (full list at DEBUG)"]
