from pathlib import Path

from src.common.experiment_registry import RunRecord
from src.common.experiment_studio import build_studio_html, write_studio


def test_build_studio_html_includes_runs_and_commands(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "checkpoints" / "best.pth"
    record = RunRecord(
        task_name="classification",
        run_dir=tmp_path / "run",
        metrics={"best_val_accuracy": 97.0, "best_val_loss": 0.1},
        artifacts={"best_checkpoint_path": str(checkpoint)},
        config={"model": {"name": "resnet18"}},
        used_validation=True,
    )

    html = build_studio_html([record])

    assert "Experiment Studio" in html
    assert "classification" in html
    assert "97.0000" in html
    assert "dnn-export-classification" in html
    assert "dnn-serve-classification" in html


def test_write_studio_writes_static_html(tmp_path: Path) -> None:
    output_path = write_studio([], tmp_path / "studio.html")

    assert output_path.exists()
    assert "No runs found" in output_path.read_text()
