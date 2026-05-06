"""Shared Optuna helpers for task-level hyperparameter search."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna

from src.common.config import BaseConfig


@dataclass
class WandbConfig(BaseConfig):
    """Weights & Biases settings used by HPO runs."""

    enabled: bool = False
    project: str = "dnn-pytorch"
    entity: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    mode: str | None = None


@dataclass
class SearchParameterConfig(BaseConfig):
    """One Optuna-suggested config value."""

    name: str = ""
    type: str = "float"
    low: int | float | None = None
    high: int | float | None = None
    step: int | float | None = None
    log: bool = False
    choices: list[Any] = field(default_factory=list)


@dataclass
class OptunaConfig(BaseConfig):
    """Configuration for an Optuna study."""

    n_trials: int = 20
    timeout: int | None = None
    direction: str = "minimize"
    objective_metric: str = "best_val_loss"
    study_name: str | None = None
    storage_dir: str = "output/optuna/classification"
    load_if_exists: bool = True
    seed: int | None = None
    n_jobs: int = 1
    final_root_dir: str | None = None
    search_space: list[SearchParameterConfig] = field(default_factory=list)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def slugify(value: str) -> str:
    """Return a filesystem- and Optuna-friendly name fragment."""
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "study"


def default_study_name(
    *,
    task_name: str,
    dataset_name: str,
    model_name: str,
) -> str:
    return slugify(f"{task_name}_{dataset_name}_{model_name}")


def resolve_study_name(
    optuna_cfg: OptunaConfig,
    *,
    task_name: str,
    dataset_name: str,
    model_name: str,
) -> str:
    if optuna_cfg.study_name:
        return slugify(optuna_cfg.study_name)
    prefix = default_study_name(
        task_name=task_name,
        dataset_name=dataset_name,
        model_name=model_name,
    )
    return slugify(f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


def study_storage_path(optuna_cfg: OptunaConfig, study_name: str) -> Path:
    return Path(optuna_cfg.storage_dir).expanduser() / f"{slugify(study_name)}.sqlite"


def study_storage_url(optuna_cfg: OptunaConfig, study_name: str) -> str:
    path = study_storage_path(optuna_cfg, study_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.resolve()}"


def study_run_root(optuna_cfg: OptunaConfig, study_name: str) -> Path:
    return Path(optuna_cfg.storage_dir).expanduser() / slugify(study_name)


def trial_root_dir(
    optuna_cfg: OptunaConfig, study_name: str, trial_number: int
) -> Path:
    return (
        study_run_root(optuna_cfg, study_name) / "trials" / f"trial_{trial_number:04d}"
    )


def final_root_dir(optuna_cfg: OptunaConfig, study_name: str) -> Path:
    if optuna_cfg.final_root_dir:
        return Path(optuna_cfg.final_root_dir).expanduser()
    return study_run_root(optuna_cfg, study_name) / "final"


def create_study(optuna_cfg: OptunaConfig, study_name: str) -> optuna.Study:
    sampler = (
        optuna.samplers.TPESampler(seed=optuna_cfg.seed)
        if optuna_cfg.seed is not None
        else None
    )
    return optuna.create_study(
        study_name=study_name,
        storage=study_storage_url(optuna_cfg, study_name),
        direction=optuna_cfg.direction,
        load_if_exists=optuna_cfg.load_if_exists,
        sampler=sampler,
    )


def load_study(optuna_cfg: OptunaConfig, study_name: str) -> optuna.Study:
    return optuna.load_study(
        study_name=study_name,
        storage=study_storage_url(optuna_cfg, study_name),
    )


def suggest_search_space(
    trial: optuna.Trial,
    search_space: list[SearchParameterConfig],
) -> dict[str, Any]:
    """Ask Optuna for all configured parameters."""
    params: dict[str, Any] = {}
    for spec in search_space:
        params[spec.name] = _suggest_one(trial, spec)
    return params


def apply_params_to_config[T](config: T, params: dict[str, Any]) -> T:
    """Return a deep-copied config with dotted-path params applied."""
    updated = copy.deepcopy(config)
    for path, value in params.items():
        set_config_value(updated, path, value)
    return updated


def set_config_value(config: Any, dotted_path: str, value: Any) -> None:
    target = config
    parts = dotted_path.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"Invalid config path: {dotted_path!r}")
    for part in parts[:-1]:
        target = target[part] if isinstance(target, dict) else getattr(target, part)
    last = parts[-1]
    if isinstance(target, dict):
        target[last] = value
    else:
        setattr(target, last, value)


def _suggest_one(trial: optuna.Trial, spec: SearchParameterConfig) -> Any:
    if not spec.name:
        raise ValueError("Search parameter name cannot be empty")

    kind = spec.type.lower()
    if kind == "float":
        if spec.low is None or spec.high is None:
            raise ValueError(f"{spec.name}: float search parameter requires low/high")
        return trial.suggest_float(
            spec.name,
            float(spec.low),
            float(spec.high),
            step=float(spec.step) if spec.step is not None else None,
            log=spec.log,
        )
    if kind == "int":
        if spec.low is None or spec.high is None:
            raise ValueError(f"{spec.name}: int search parameter requires low/high")
        return trial.suggest_int(
            spec.name,
            int(spec.low),
            int(spec.high),
            step=int(spec.step) if spec.step is not None else 1,
            log=spec.log,
        )
    if kind == "categorical":
        if not spec.choices:
            raise ValueError(f"{spec.name}: categorical parameter requires choices")
        return trial.suggest_categorical(spec.name, spec.choices)

    raise ValueError(
        f"{spec.name}: unsupported search parameter type {spec.type!r}; "
        "use float, int or categorical"
    )
