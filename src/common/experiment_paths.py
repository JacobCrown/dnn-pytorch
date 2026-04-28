"""Single source of truth for experiment output dirs (mirror *_Trainer._get_root_dir_path logic)."""

from __future__ import annotations

from pathlib import Path


def classification_experiment_root(
    *,
    train_root_dir: str | Path,
    data_root: str | Path,
    model_name: str,
    height: int,
    width: int,
    epochs: int,
    batch_size: int,
) -> Path:
    dataset_name = Path(data_root).name
    leaf = f"{model_name}_{height}x{width}_{epochs}epochs_{batch_size}bs"
    return Path(train_root_dir) / "classification" / dataset_name / leaf


def segmentation_experiment_root(
    *,
    train_root_dir: str | Path,
    data_root: str | Path,
    architecture: str,
    encoder_name: str,
    height: int,
    width: int,
    epochs: int,
    batch_size: int,
) -> Path:
    dataset_name = Path(data_root).name
    leaf = (
        f"{architecture}_{encoder_name}_{height}x{width}_{epochs}epochs_{batch_size}bs"
    )
    return Path(train_root_dir) / "segmentation" / dataset_name / leaf


def road_sign_classification_experiment_root(
    *,
    train_root_dir: str | Path,
    data_root: str | Path,
    model_name: str,
    height: int,
    width: int,
    epochs: int,
    batch_size: int,
) -> Path:
    dataset_name = Path(data_root).name
    leaf = f"{model_name}_{height}x{width}_{epochs}epochs_{batch_size}bs"
    return Path(train_root_dir) / "road_sign_classification" / dataset_name / leaf
