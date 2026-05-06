"""Optuna entry point for classification hyperparameter search."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import optuna
from omegaconf import OmegaConf

from src.classification.config import ClassificationConfig
from src.classification.training_pipeline import (
    ClassificationTrainingMetrics,
    ClassificationTrainingOutcome,
    run_classification_training,
)
from src.common.experiment_paths import classification_experiment_root
from src.common.hpo import (
    apply_params_to_config,
    create_study,
    resolve_study_name,
    study_storage_url,
    suggest_search_space,
    trial_root_dir,
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


def _objective_value(
    metrics: ClassificationTrainingMetrics,
    objective_metric: str,
) -> float:
    value = getattr(metrics, objective_metric, None)
    if value is None:
        raise ValueError(
            f"Objective metric {objective_metric!r} is unavailable for this trial"
        )
    return float(value)


def _log_wandb_history(
    outcome: ClassificationTrainingOutcome,
    *,
    objective_metric: str,
    objective_value: float,
) -> None:
    import wandb

    losses = outcome.metrics.losses
    accuracies = outcome.metrics.accuracies
    epoch_count = max(
        len(losses.get("train", [])),
        len(losses.get("val", [])),
        len(accuracies.get("train", [])),
        len(accuracies.get("val", [])),
    )
    for idx in range(epoch_count):
        payload: dict[str, float | int] = {"epoch": idx + 1}
        if idx < len(losses.get("train", [])):
            payload["train/loss"] = losses["train"][idx]
        if idx < len(accuracies.get("train", [])):
            payload["train/accuracy"] = accuracies["train"][idx]
        if idx < len(losses.get("val", [])):
            payload["val/loss"] = losses["val"][idx]
        if idx < len(accuracies.get("val", [])):
            payload["val/accuracy"] = accuracies["val"][idx]
        wandb.log(payload, step=idx + 1)

    wandb.summary[objective_metric] = objective_value
    if outcome.metrics.test_loss is not None:
        wandb.summary["test/loss"] = outcome.metrics.test_loss
    if outcome.metrics.test_accuracy is not None:
        wandb.summary["test/accuracy"] = outcome.metrics.test_accuracy


def _start_wandb_run(
    config: ClassificationConfig,
    *,
    study_name: str,
    trial: optuna.Trial,
    params: dict[str, Any],
) -> Any | None:
    wandb_cfg = config.optuna.wandb
    if not wandb_cfg.enabled:
        return None

    import wandb

    return wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity,
        group=wandb_cfg.group or study_name,
        name=f"{study_name}-trial-{trial.number:04d}",
        tags=[*wandb_cfg.tags, "optuna", "classification"],
        mode=wandb_cfg.mode,
        config={
            "study_name": study_name,
            "trial_number": trial.number,
            "optuna_params": params,
            "classification_config": config.to_dict(),
        },
        reinit=True,
    )


def _finish_wandb_run(run: Any | None, *, failed: bool = False) -> None:
    if run is None:
        return
    import wandb

    wandb.finish(exit_code=1 if failed else 0)


def run_classification_optuna_search(config: ClassificationConfig) -> optuna.Study:
    dataset_name = Path(config.data.root).name
    study_name = resolve_study_name(
        config.optuna,
        task_name="classification",
        dataset_name=dataset_name,
        model_name=config.model.name,
    )
    study = create_study(config.optuna, study_name)
    study_root = trial_root_dir(config.optuna, study_name, 0).parents[1]
    study_root.mkdir(parents=True, exist_ok=True)

    reset_run_logging_configuration()
    run_log_path = configure_run_logging(
        experiment_directory=study_root,
        use_rich_console=True,
        quiet_third_party_loggers=True,
    )
    logger.info("Optuna study log: %s", run_log_path)
    logger.info(
        "Study %s uses storage %s",
        study_name,
        study_storage_url(config.optuna, study_name),
    )
    logger.info("Device: %s", DEVICE)
    if config.optuna.n_jobs != 1:
        logger.warning(
            "Parallel Optuna workers are not enabled by this entrypoint; using n_jobs=1"
        )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_search_space(trial, config.optuna.search_space)
        trial_config = apply_params_to_config(config, params)
        trial_config.optuna.study_name = study_name
        trial_config.train.root_dir = str(
            trial_root_dir(config.optuna, study_name, trial.number)
        )
        trial_config.train.resume_from_checkpoint = None

        if config.optuna.seed is not None:
            set_seed(config.optuna.seed + trial.number)
        else:
            set_seed()

        reset_run_logging_configuration()
        trial_log_path = configure_run_logging(
            experiment_directory=_experiment_dir(trial_config),
            use_rich_console=True,
            quiet_third_party_loggers=True,
        )
        logger.info("Optuna study=%s trial=%d", study_name, trial.number)
        logger.info("Trial log: %s", trial_log_path)
        logger.info("Trial parameters: %s", params)

        wandb_run = _start_wandb_run(
            trial_config,
            study_name=study_name,
            trial=trial,
            params=params,
        )
        failed = False
        try:
            outcome = run_classification_training(trial_config)
            if not outcome.used_validation:
                logger.warning(
                    "Trial %d has no validation loader; objective uses train metrics "
                    "mirrored as validation metrics",
                    trial.number,
                )
            value = _objective_value(
                outcome.metrics,
                trial_config.optuna.objective_metric,
            )
            if wandb_run is not None:
                _log_wandb_history(
                    outcome,
                    objective_metric=trial_config.optuna.objective_metric,
                    objective_value=value,
                )
            trial.report(value, step=len(outcome.metrics.losses.get("val", [])))
            trial.set_user_attr("study_name", study_name)
            trial.set_user_attr("trial_root_dir", trial_config.train.root_dir)
            trial.set_user_attr("checkpoints_dir", str(outcome.checkpoints_dir))
            trial.set_user_attr(
                "best_checkpoint_path",
                str(outcome.best_checkpoint_path)
                if outcome.best_checkpoint_path is not None
                else None,
            )
            trial.set_user_attr(
                "objective_metric", trial_config.optuna.objective_metric
            )
            return value
        except Exception:
            failed = True
            raise
        finally:
            _finish_wandb_run(wandb_run, failed=failed)

    try:
        study.optimize(
            objective,
            n_trials=config.optuna.n_trials,
            timeout=config.optuna.timeout,
            n_jobs=1,
        )
    except KeyboardInterrupt:
        logger.warning("Optuna search interrupted; completed trials remain in storage")
    try:
        logger.info(
            "Best trial: #%d value=%s", study.best_trial.number, study.best_value
        )
        logger.info("Best params: %s", study.best_trial.params)
    except ValueError:
        logger.warning("No completed Optuna trials available yet")
    return study


@hydra.main(
    version_base=None,
    config_path=_classification_config_path(),
    config_name="main.yaml",
)
def main(cfg: Any) -> None:
    config = _config_from_hydra(cfg)
    logger.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    run_classification_optuna_search(config)


if __name__ == "__main__":
    main()
