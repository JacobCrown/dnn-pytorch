"""Shared contracts for task training pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _existing_path(path: Path) -> Path | None:
    return path if path.exists() else None


@dataclass(frozen=True)
class TrainingArtifacts:
    """Filesystem artifacts produced by a completed training run."""

    root_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    visualizations_dir: Path
    best_checkpoint_path: Path | None = None
    last_checkpoint_path: Path | None = None
    predictor_checkpoint_path: Path | None = None

    @classmethod
    def from_completed_run(
        cls,
        root_dir: str | Path,
        *,
        checkpoints_dir_name: str = "checkpoints",
        logs_dir_name: str = "logs",
        visualizations_dir_name: str = "visualizations",
    ) -> TrainingArtifacts:
        root = Path(root_dir)
        checkpoints_dir = root / checkpoints_dir_name
        return cls(
            root_dir=root,
            checkpoints_dir=checkpoints_dir,
            logs_dir=root / logs_dir_name,
            visualizations_dir=root / visualizations_dir_name,
            best_checkpoint_path=_existing_path(checkpoints_dir / "best.pth"),
            last_checkpoint_path=_existing_path(checkpoints_dir / "last.pth"),
            predictor_checkpoint_path=_existing_path(checkpoints_dir / "predictor.pth"),
        )


@dataclass(frozen=True)
class TrainingPipelineResult:
    """Task-agnostic result returned by Hydra-free training pipelines."""

    task_name: str
    artifacts: TrainingArtifacts
    metrics: Any
    used_validation: bool

    @property
    def checkpoints_dir(self) -> Path:
        return self.artifacts.checkpoints_dir

    @property
    def best_checkpoint_path(self) -> Path | None:
        return self.artifacts.best_checkpoint_path

    @property
    def predictor_checkpoint_path(self) -> Path | None:
        return self.artifacts.predictor_checkpoint_path
