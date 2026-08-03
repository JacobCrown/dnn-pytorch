"""End-to-end training pipeline tests with tiny synthetic datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.classification.config import (
    ClassificationConfig,
    ClassificationDataConfig,
    ClassificationModelConfig,
)
from src.classification.training_pipeline import run_classification_training
from src.common.config import (
    AugmentationConfig,
    BaseTrainingConfig,
    LossFunctionConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from src.common.experiment_registry import (
    RUN_RECORD_NAME,
    collect_run_records,
    format_leaderboard,
)
from src.common.experiment_studio import build_studio_html
from src.segmentation.config import (
    ClassConfig,
    SegmentationConfig,
    SegmentationDataConfig,
    SegmentationModelConfig,
)
from src.segmentation.training_pipeline import run_segmentation_training
from tests.fixtures.datasets import (
    create_tiny_classification_dataset,
    create_tiny_segmentation_dataset,
)


def _minimal_classification_config(dataset_root: Path, output_root: Path) -> ClassificationConfig:
    return ClassificationConfig(
        model=ClassificationModelConfig(
            name="mobilenetv3_small_075",
            pretrained=False,
            height=32,
            width=32,
            dropout=0.0,
            preprocessing=[
                AugmentationConfig(name="Resize", params={"height": 32, "width": 32}),
                AugmentationConfig(
                    name="Normalize",
                    params={"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
                ),
                AugmentationConfig(name="ToTensorV2", params={}),
            ],
        ),
        data=ClassificationDataConfig(
            root=str(dataset_root),
            train_dir="train",
            val_dir="val",
            test_dir="test",
            max_samples=4,
        ),
        train=BaseTrainingConfig(
            epochs=1,
            batch_size=2,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
            use_amp=False,
            root_dir=str(output_root),
            visualize_pre_train=False,
            visualize_post_train=False,
            optimizer=OptimizerConfig(name="Adam", params={"lr": 0.01}),
            scheduler=SchedulerConfig(name="StepLR", params={"step_size": 1, "gamma": 1.0}),
            loss_function=[
                LossFunctionConfig(
                    name="FocalLoss",
                    params={"mode": "multiclass", "gamma": 2.0},
                )
            ],
        ),
    )


def _minimal_segmentation_config(dataset_root: Path, output_root: Path) -> SegmentationConfig:
    return SegmentationConfig(
        model=SegmentationModelConfig(
            architecture="Unet",
            encoder_name="resnet18",
            encoder_weights=None,
            height=64,
            width=64,
            classes=2,
            preprocessing=[
                AugmentationConfig(name="Resize", params={"height": 64, "width": 64}),
                AugmentationConfig(
                    name="Normalize",
                    params={"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
                ),
                AugmentationConfig(name="ToTensorV2", params={"transpose_mask": True}),
            ],
        ),
        data=SegmentationDataConfig(
            root=str(dataset_root),
            train_dir="train",
            val_dir="val",
            test_dir="test",
            max_samples=2,
            class_config=[
                ClassConfig(name="background", pixel_value=[0, 0, 0]),
                ClassConfig(name="road", pixel_value=[128, 64, 128]),
            ],
            class_mapping=[],
        ),
        train=BaseTrainingConfig(
            epochs=1,
            batch_size=1,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
            use_amp=False,
            root_dir=str(output_root),
            visualize_pre_train=False,
            visualize_post_train=False,
            resume_from_checkpoint=None,
            optimizer=OptimizerConfig(name="Adam", params={"lr": 0.01}),
            scheduler=SchedulerConfig(name="StepLR", params={"step_size": 1, "gamma": 1.0}),
            loss_function=[
                LossFunctionConfig(
                    name="DiceLoss",
                    params={"mode": "multiclass", "ignore_index": 255, "smooth": 1e-6},
                )
            ],
        ),
    )


@pytest.mark.integration
def test_classification_training_writes_run_record_and_artifacts(tmp_path: Path) -> None:
    dataset_root = create_tiny_classification_dataset(tmp_path / "data")
    config = _minimal_classification_config(dataset_root, tmp_path / "output")

    outcome = run_classification_training(config)

    run_record = outcome.artifacts.root_dir / RUN_RECORD_NAME
    assert run_record.exists()
    payload = json.loads(run_record.read_text())
    assert payload["task_name"] == "classification"
    assert payload["used_validation"] is True
    assert payload["config"]["model"]["name"] == "mobilenetv3_small_075"
    assert payload["metrics"]["best_val_loss"] is not None
    assert outcome.best_checkpoint_path is not None
    assert outcome.best_checkpoint_path.exists()
    assert outcome.predictor_checkpoint_path is not None
    assert outcome.predictor_checkpoint_path.exists()
    assert (outcome.artifacts.logs_dir / "main.log").exists()
    assert (outcome.artifacts.root_dir / "tracking" / "events.jsonl").exists()
    events = [
        json.loads(line)
        for line in (outcome.artifacts.root_dir / "tracking" / "events.jsonl")
        .read_text()
        .splitlines()
    ]
    epoch_events = [event for event in events if event.get("type") == "metrics" and event.get("step")]
    assert len(epoch_events) >= 1


@pytest.mark.integration
def test_segmentation_training_writes_run_record_and_artifacts(tmp_path: Path) -> None:
    dataset_root = create_tiny_segmentation_dataset(tmp_path / "data")
    config = _minimal_segmentation_config(dataset_root, tmp_path / "output")

    outcome = run_segmentation_training(config)

    run_record = outcome.artifacts.root_dir / RUN_RECORD_NAME
    assert run_record.exists()
    payload = json.loads(run_record.read_text())
    assert payload["task_name"] == "segmentation"
    assert payload["metrics"]["best_val_iou"] is not None
    assert outcome.best_checkpoint_path is not None
    assert outcome.best_checkpoint_path.exists()
    assert (outcome.artifacts.logs_dir / "main.log").exists()
    assert (outcome.artifacts.root_dir / "tracking" / "events.jsonl").exists()
    events = [
        json.loads(line)
        for line in (outcome.artifacts.root_dir / "tracking" / "events.jsonl")
        .read_text()
        .splitlines()
    ]
    epoch_events = [event for event in events if event.get("type") == "metrics" and event.get("step")]
    assert len(epoch_events) >= 1


@pytest.mark.integration
def test_experiment_tooling_reads_completed_runs(tmp_path: Path) -> None:
    dataset_root = create_tiny_classification_dataset(tmp_path / "data")
    config = _minimal_classification_config(dataset_root, tmp_path / "output")
    run_classification_training(config)

    records = collect_run_records([tmp_path / "output"])
    assert len(records) == 1

    leaderboard = format_leaderboard(records)
    assert "classification" in leaderboard
    assert str(records[0].run_dir) in leaderboard

    html = build_studio_html(records)
    assert "Experiment Studio" in html
    assert "classification" in html
