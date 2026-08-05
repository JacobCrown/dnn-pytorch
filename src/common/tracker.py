from pathlib import Path

import mlflow
import torch
from torch import nn

from src.core.tracker import BaseTracker


class MLFlowTracker(BaseTracker):
    def init_tracker(self, experiment_name: Path | str, run_name: str) -> None:
        if isinstance(experiment_name, Path):
            experiment_name = str(experiment_name)
        mlflow.enable_system_metrics_logging()
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        mlflow.log_metrics(metrics=metrics, step=step)

    def log_artifacts(self, local_dir: Path | str) -> None:
        if isinstance(local_dir, Path):
            local_dir = str(local_dir)
        mlflow.log_artifacts(local_dir=local_dir)

    def log_model(self, pytorch_model: nn.Module, name: str, input_example: torch.Tensor) -> None:
        mlflow.pytorch.log_model(pytorch_model=pytorch_model, name=name, input_example=input_example, serialization_format="pickle")

    def close_tracker(self) -> None:
        mlflow.end_run()
