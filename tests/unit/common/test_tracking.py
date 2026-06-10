import json
from pathlib import Path

from src.common.config import TrackingConfig
from src.common.tracking import LocalJsonTracker, build_tracker


def test_local_json_tracker_writes_metric_and_artifact_events(tmp_path: Path) -> None:
    tracker = LocalJsonTracker(run_dir=tmp_path, task_name="classification")

    tracker.log_metrics({"best_val_accuracy": 93.2}, step=5)
    tracker.log_artifact("best_checkpoint", tmp_path / "checkpoints" / "best.pth")
    tracker.close()

    events = [
        json.loads(line)
        for line in (tmp_path / "tracking" / "events.jsonl").read_text().splitlines()
    ]
    assert events[0]["type"] == "metrics"
    assert events[0]["task_name"] == "classification"
    assert events[0]["step"] == 5
    assert events[0]["metrics"]["best_val_accuracy"] == 93.2
    assert events[1]["type"] == "artifact"
    assert events[1]["name"] == "best_checkpoint"


def test_build_tracker_uses_local_tracker_by_default(tmp_path: Path) -> None:
    tracker = build_tracker(
        config=TrackingConfig(),
        run_dir=tmp_path,
        task_name="segmentation",
    )

    tracker.log_metrics({"best_val_iou": 0.7})
    tracker.close()

    assert (tmp_path / "tracking" / "events.jsonl").exists()
