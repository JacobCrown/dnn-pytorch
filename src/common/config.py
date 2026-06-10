"""
Common configuration module providing a shared base and utilities for all
project configuration files. It defines the core logic for loading from YAML
and serializing to dicts, as well as common configuration blocks like
optimizer and scheduler settings.
"""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T", bound="BaseConfig")

# ============================================================================
# 1. BASE CONFIGURATION LOGIC
# ============================================================================


class BaseConfig:
    """Base class providing (de)serialisation helpers for nested configs."""

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain Python dict."""
        return asdict(self)  # type: ignore[arg-type]

    def to_yaml(self, yaml_path: str | Path) -> None:
        """Serialise this configuration object to a YAML file."""
        with open(Path(yaml_path), "w", encoding="utf-8") as fp:
            yaml.dump(self.to_dict(), fp, default_flow_style=False)

    @classmethod
    def from_yaml(cls: type[T], yaml_path: str | Path) -> T:
        """Instantiate this config class from a YAML file."""
        with open(Path(yaml_path), encoding="utf-8") as fp:
            data: dict[str, Any] = yaml.safe_load(fp)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Recursively instantiate this config class from a plain dict."""
        return _from_dict_recursive(cls, data)


# --- Private recursive helper ---


def _from_dict_recursive[T: "BaseConfig"](cls: type[T], data: dict[str, Any]) -> T:
    """Recursive builder that correctly handles nested dataclasses."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls} must be a dataclass")

    if data is None:
        return None  # type: ignore

    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        raw_value = data[f.name]

        field_type = type_hints.get(f.name, f.type)
        target_type = _resolve_optional_type(field_type)

        if raw_value is None:
            kwargs[f.name] = None
            continue

        if is_dataclass(target_type):
            kwargs[f.name] = _from_dict_recursive(target_type, raw_value)
            continue

        origin = get_origin(target_type)
        if origin in (list, list):
            item_type = _resolve_optional_type(get_args(target_type)[0])
            if is_dataclass(item_type):
                kwargs[f.name] = [
                    _from_dict_recursive(item_type, item) for item in raw_value
                ]
            else:
                kwargs[f.name] = raw_value
            continue

        kwargs[f.name] = raw_value

    return cls(**kwargs)


def _resolve_optional_type(tp):
    """Resolves Optional[T] to T for convenient runtime checks."""
    origin = get_origin(tp)
    if origin in (Union, UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        return args[0] if len(args) == 1 else tp
    return tp


# ============================================================================
# 2. COMMON CONFIGURATION BLOCKS
# ============================================================================


@dataclass
class AugmentationConfig(BaseConfig):
    """Collection of Albumentations transforms."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig(BaseConfig):
    """Wrapper around torch.optim selection parameters."""

    name: str = "Adam"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerConfig(BaseConfig):
    """Wrapper around torch.optim.lr_scheduler parameters."""

    name: str = "cosine"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossFunctionConfig(BaseConfig):
    """Wrapper around torch.nn.functional parameters."""

    name: str = "cross_entropy"
    params: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0


@dataclass
class EMAConfig(BaseConfig):
    """Exponential moving average settings for model weights."""

    enabled: bool = False
    decay: float = 0.9999
    update_after_step: int = 0
    update_every: int = 1


@dataclass
class TrackingConfig(BaseConfig):
    """Training run tracking settings."""

    local_enabled: bool = True
    wandb_enabled: bool = False
    wandb_project: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class BaseDataConfig(BaseConfig):
    """Configuration for data loading and preprocessing."""

    root: str = "data"
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str | None = None
    max_samples: int | None = None

    # Supported image extensions
    extensions: list[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    )


@dataclass
class BaseTrainingConfig(BaseConfig):
    """Configuration for training."""

    epochs: int = 100
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True

    # DataLoader performance knobs
    persistent_workers: bool = True
    prefetch_factor: int | None = 2
    drop_last: bool = False

    # Optimizer
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Scheduler
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Loss function(s) - can be a single config or list of configs
    loss_function: list[LossFunctionConfig] = field(
        default_factory=lambda: [LossFunctionConfig()]
    )

    # Early stopping
    early_stopping_patience: int | None = 10

    # Resume from checkpoint
    resume_from_checkpoint: str | None = None

    # Mixed precision
    use_amp: bool = False

    # Gradient clipping
    gradient_clip: float | None = None

    # Exponential moving average of model weights
    ema: EMAConfig = field(default_factory=EMAConfig)

    # Tracking
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    # Root directory for output files
    root_dir: str = "output"
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"
    visualize_dir: str = "visualizations"

    # visualization flags
    visualize_pre_train: bool = False
    visualize_post_train: bool = False
