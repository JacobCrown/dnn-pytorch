from pathlib import Path

from src.common.training_contracts import TrainingArtifacts, TrainingPipelineResult


def test_training_artifacts_resolves_completed_run_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "output" / "classification" / "demo"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    best_checkpoint = checkpoints_dir / "best.pth"
    predictor_checkpoint = checkpoints_dir / "predictor.pth"
    best_checkpoint.touch()
    predictor_checkpoint.touch()

    artifacts = TrainingArtifacts.from_completed_run(run_dir)

    assert artifacts.root_dir == run_dir
    assert artifacts.checkpoints_dir == checkpoints_dir
    assert artifacts.logs_dir == run_dir / "logs"
    assert artifacts.visualizations_dir == run_dir / "visualizations"
    assert artifacts.best_checkpoint_path == best_checkpoint
    assert artifacts.last_checkpoint_path is None
    assert artifacts.predictor_checkpoint_path == predictor_checkpoint


def test_training_pipeline_result_exposes_common_artifact_paths(tmp_path: Path) -> None:
    artifacts = TrainingArtifacts.from_completed_run(tmp_path)
    result = TrainingPipelineResult(
        task_name="classification",
        artifacts=artifacts,
        metrics={"accuracy": [95.0]},
        used_validation=True,
    )

    assert result.checkpoints_dir == tmp_path / "checkpoints"
    assert result.best_checkpoint_path is None
    assert result.predictor_checkpoint_path is None
