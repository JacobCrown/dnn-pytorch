from abc import ABC, abstractmethod
from pathlib import Path

from torch import nn


class BaseTracker(ABC):
    """Base tracker class for all tasks."""

    @abstractmethod
    def init_tracker(self, experiment_name: Path | str, run_name: str) -> None:
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        pass

    @abstractmethod
    def log_artifacts(self, local_dir: Path | str) -> None:
        pass

    @abstractmethod
    def log_model(self, pytorch_model: nn.Module, model_path: Path | str) -> None:
        pass

    @abstractmethod
    def close_tracker(self) -> None:
        pass

