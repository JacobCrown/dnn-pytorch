"""Run one classification training job with the best Optuna trial params."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
from omegaconf import OmegaConf

from src.classification.config import ClassificationConfig
from src.classification.training_pipeline import run_classification_training
from src.common.experiment_paths import classification_experiment_root
from src.common.hpo import (
    apply_params_to_config,
    final_root_dir,
    load_study,
    resolve_study_name,
)
from src.common.hydra_resolvers import register_hydra_resolvers
from src.common.logging import configure_run_logging, reset_run_logging_configuration
from src.common.utils import DEVICE, set_seed

register_hydra_resolvers()

logger = logging.getLogger(__name__)


def _classification_config_path() -> str:
    for root in (Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[2]):
        candidate = root / "configs" / "classification"
        if candidate.exists():
            return str(candidate.resolve())
    return str(
        (Path(__file__).resolve().parents[2] / "configs" / "classification").resolve()
    )


def _experiment_dir(config: ClassificationConfig) -> Path:
    return classification_experiment_root(
        train_root_dir=config.train.root_dir,
        data_root=config.data.root,
        model_name=config.model.name,
        height=config.model.height,
        width=config.model.width,
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
    ).resolve()


def _config_from_hydra(cfg: Any) -> ClassificationConfig:
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise TypeError("Hydra configuration must resolve to a dictionary")
    return ClassificationConfig.from_dict(data)


def run_best_classification_training(config: ClassificationConfig) -> None:
    dataset_name = Path(config.data.root).name
    study_name = resolve_study_name(
        config.optuna,
        task_name="classification",
        dataset_name=dataset_name,
        model_name=config.model.name,
    )
    if config.optuna.study_name is None:
        raise ValueError(
            "optuna.study_name must be set when training from an existing study"
        )

    study = load_study(config.optuna, study_name)
    try:
        best_trial = study.best_trial
    except ValueError as exc:
        raise ValueError(f"Study {study_name!r} has no completed trials") from exc

    final_config = apply_params_to_config(config, best_trial.params)
    final_config.optuna.study_name = study_name
    final_config.train.root_dir = str(final_root_dir(config.optuna, study_name))
    final_config.train.resume_from_checkpoint = None

    reset_run_logging_configuration()
    run_log_path = configure_run_logging(
        experiment_directory=_experiment_dir(final_config),
        use_rich_console=True,
        quiet_third_party_loggers=True,
    )
    set_seed(config.optuna.seed or 42)
    logger.info("Final classification training from Optuna study=%s", study_name)
    logger.info("Best trial: #%d value=%s", best_trial.number, best_trial.value)
    logger.info("Best params: %s", best_trial.params)
    logger.info("Final run log: %s", run_log_path)
    logger.info("Device: %s", DEVICE)

    run_classification_training(final_config)


@hydra.main(
    version_base=None,
    config_path=_classification_config_path(),
    config_name="main.yaml",
)
def main(cfg: Any) -> None:
    config = _config_from_hydra(cfg)
    logger.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    run_best_classification_training(config)


if __name__ == "__main__":
    main()
