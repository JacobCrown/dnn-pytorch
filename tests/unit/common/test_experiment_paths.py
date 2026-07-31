"""Experiment output paths must match Hydra interpolations / *_Trainer._get_root_dir_path."""

from pathlib import Path

from src.common.experiment_paths import (
    classification_experiment_root, road_sign_classification_experiment_root,
    segmentation_experiment_root)


def test_classification_experiment_root() -> None:
    p = classification_experiment_root(
        train_root_dir="output",
        data_root="data/classification/classification-merged",
        model_name="convnext_tiny",
        height=32,
        width=32,
        epochs=5,
        batch_size=256,
    )
    expected = Path(
        "output/classification/classification-merged/convnext_tiny_32x32_5epochs_256bs"
    )
    assert p == expected


def test_segmentation_experiment_root() -> None:
    p = segmentation_experiment_root(
        train_root_dir="output",
        data_root="/data/seg/MyDataset",
        architecture="DeepLabV3Plus",
        encoder_name="resnet18",
        height=368,
        width=480,
        epochs=10,
        batch_size=12,
    )
    expected = Path(
        "output/segmentation/MyDataset/DeepLabV3Plus_resnet18_368x480_10epochs_12bs"
    )
    assert p == expected


def test_road_sign_classification_experiment_root() -> None:
    p = road_sign_classification_experiment_root(
        train_root_dir="output",
        data_root="/data/rs/foo",
        model_name="resnet34",
        height=128,
        width=128,
        epochs=100,
        batch_size=128,
    )
    expected = Path(
        "output/road_sign_classification/foo/resnet34_128x128_100epochs_128bs"
    )
    assert p == expected
