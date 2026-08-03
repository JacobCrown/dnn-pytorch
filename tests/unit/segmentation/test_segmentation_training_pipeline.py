from pathlib import Path

from src.common.training_contracts import TrainingArtifacts
from src.segmentation.config import SegmentationConfig
from src.segmentation.training_pipeline import (
    SegmentationTrainingMetrics,
    SegmentationTrainingOutcome,
    apply_classes_from_train_dataset,
)


class FakeSegmentationDataset:
    def __init__(self, class_names: list[str]) -> None:
        self._class_names = class_names

    def get_class_names(self) -> list[str]:
        return self._class_names


def test_apply_classes_from_train_dataset_sets_model_class_count() -> None:
    config = SegmentationConfig()
    config.model.classes = 1
    dataset = FakeSegmentationDataset(["road", "sky", "vehicle"])

    derived = apply_classes_from_train_dataset(config, dataset)

    assert derived == 3
    assert config.model.classes == 3


def test_segmentation_training_outcome_exposes_common_artifact_paths(
    tmp_path: Path,
) -> None:
    metrics = SegmentationTrainingMetrics(
        losses={"train": [1.0], "val": [0.5], "test": []},
        ious={"train": [0.4], "val": [0.6], "test": []},
        best_val_loss=0.5,
        best_val_iou=0.6,
        final_train_loss=1.0,
        final_train_iou=0.4,
        final_val_loss=0.5,
        final_val_iou=0.6,
        test_loss=None,
        test_iou=None,
    )
    outcome = SegmentationTrainingOutcome(
        artifacts=TrainingArtifacts.from_completed_run(tmp_path),
        metrics=metrics,
        used_validation=True,
    )

    assert outcome.checkpoints_dir == tmp_path / "checkpoints"
    assert outcome.best_checkpoint_path is None
