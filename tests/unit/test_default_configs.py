"""Ensure tracked default Hydra YAML groups load into dataclass configs."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.classification.config import ClassificationConfig
from src.segmentation.config import SegmentationConfig

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_classification_default_config_groups_load() -> None:
    base = CONFIG_ROOT / "classification"
    config = ClassificationConfig.from_dict(
        {
            "data": _load_yaml(base / "data/default.yaml"),
            "model": _load_yaml(base / "model/default.yaml"),
            "train": _load_yaml(base / "train/default.yaml"),
        }
    )

    assert config.data.root == "data/classification/cifar100"
    assert config.data.train_dir == "train"
    assert config.data.max_samples is None
    assert config.model.height == 32
    assert config.train.epochs == 5
    assert config.train.scheduler.params["T_max"] == config.train.epochs
    assert config.train.tracking.local_enabled is True
    assert config.train.distributed.strategy == "single"


def test_segmentation_default_config_groups_load() -> None:
    base = CONFIG_ROOT / "segmentation"
    config = SegmentationConfig.from_dict(
        {
            "data": _load_yaml(base / "data/default.yaml"),
            "model": _load_yaml(base / "model/default.yaml"),
            "train": _load_yaml(base / "train/default.yaml"),
        }
    )

    assert config.data.root == "data/segmentation/CamVid"
    assert config.data.max_samples is None
    assert config.train.epochs == 50
    assert config.train.scheduler.params["T_max"] == config.train.epochs
    assert config.train.tracking.local_enabled is True
    assert config.model.preprocessing[0].params["interpolation"] == 2
