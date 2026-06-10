import json
from pathlib import Path

from src.common.experiment_registry import (
    RunRecord,
    collect_run_records,
    format_leaderboard,
    rank_run_records,
    write_run_record,
)
from src.common.training_contracts import TrainingArtifacts


def test_write_run_record_serializes_training_outcome(tmp_path: Path) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    best_checkpoint = checkpoints_dir / "best.pth"
    best_checkpoint.touch()

    record_path = write_run_record(
        task_name="classification",
        artifacts=TrainingArtifacts.from_completed_run(tmp_path),
        metrics={"best_val_accuracy": 91.5, "best_val_loss": 0.2},
        config={"model": {"name": "resnet18"}},
        used_validation=True,
    )

    payload = json.loads(record_path.read_text())
    assert payload["task_name"] == "classification"
    assert payload["metrics"]["best_val_accuracy"] == 91.5
    assert payload["artifacts"]["best_checkpoint_path"] == str(best_checkpoint)
    assert payload["config"]["model"]["name"] == "resnet18"


def test_collect_and_rank_run_records_across_tasks(tmp_path: Path) -> None:
    first = tmp_path / "cls_a"
    second = tmp_path / "cls_b"
    third = tmp_path / "seg"
    first.mkdir()
    second.mkdir()
    third.mkdir()

    for run_dir, task_name, metrics in [
        (first, "classification", {"best_val_accuracy": 90.0, "best_val_loss": 0.4}),
        (second, "classification", {"best_val_accuracy": 95.0, "best_val_loss": 0.3}),
        (third, "segmentation", {"best_val_iou": 0.7, "best_val_loss": 0.6}),
    ]:
        write_run_record(
            task_name=task_name,
            artifacts=TrainingArtifacts.from_completed_run(run_dir),
            metrics=metrics,
            config={},
            used_validation=True,
        )

    records = collect_run_records([tmp_path])
    ranked = rank_run_records(records)

    assert [record.run_dir.name for record in ranked] == ["cls_b", "cls_a", "seg"]


def test_format_leaderboard_includes_score_and_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "best.pth"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    record = RunRecord(
        task_name="classification",
        run_dir=tmp_path,
        metrics={"best_val_accuracy": 88.0},
        artifacts={"best_checkpoint_path": str(checkpoint)},
        config={},
        used_validation=True,
    )

    leaderboard = format_leaderboard([record])

    assert "classification" in leaderboard
    assert "88.0000" in leaderboard
    assert str(checkpoint) in leaderboard
