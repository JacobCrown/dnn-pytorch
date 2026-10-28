"""Single source of truth for experiment output dirs (mirror *_Trainer._get_root_dir_path logic)."""

from __future__ import annotations

from pathlib import Path

from src.classification.config import ClassificationConfig
from src.segmentation.config import SegmentationConfig


def classification_experiment_root(
    config: ClassificationConfig,
) -> Path:
    dataset_name = Path(config.data.root).name
    leaf = f"{config.model.height}x{config.model.width}_{config.train.epochs}epochs_{config.train.batch_size}bs"
    return Path(config.train.root_dir) / "classification" / dataset_name / config.model.name / leaf


def segmentation_experiment_root(
    config: SegmentationConfig,
) -> Path:
    dataset_name = Path(config.data.root).name
    leaf = (
        f"{config.model.height}hx{config.model.width}w_{config.train.epochs}epochs_{config.train.batch_size}bs"
    )
    return Path(config.train.root_dir) / "segmentation" / dataset_name / f"{config.model.architecture}_{config.model.encoder_name}" / leaf
