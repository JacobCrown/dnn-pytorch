"""
Common configuration module providing a shared base and utilities for all
project configuration files. It defines the core logic for loading from YAML
and serializing to dicts, as well as common configuration blocks like
optimizer and scheduler settings.
"""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml

T = TypeVar("T", bound="BaseConfig")

# ============================================================================
# 1. BASE CONFIGURATION LOGIC
# ============================================================================


class BaseConfig:
    """Base class providing (de)serialisation helpers for nested configs."""

    def to_dict(self) -> Dict[str, Any]:
        """Return the config as a plain Python dict."""
        return asdict(self)  # type: ignore[arg-type]

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Serialise this configuration object to a YAML file."""
        with open(Path(yaml_path), "w", encoding="utf-8") as fp:
            yaml.dump(self.to_dict(), fp, default_flow_style=False)

    @classmethod
    def from_yaml(cls: Type[T], yaml_path: Union[str, Path]) -> T:
        """Instantiate this config class from a YAML file."""
        with open(Path(yaml_path), "r", encoding="utf-8") as fp:
            data: Dict[str, Any] = yaml.safe_load(fp)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Recursively instantiate this config class from a plain dict."""
        return _from_dict_recursive(cls, data)


# --- Private recursive helper ---


def _from_dict_recursive(cls: Type[T], data: Dict[str, Any]) -> T:
    """Recursive builder that correctly handles nested dataclasses."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls} must be a dataclass")

    if data is None:
        return None  # type: ignore

    type_hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
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
        if origin in (list, List):
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
    if origin is Union:
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
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig(BaseConfig):
    """Wrapper around torch.optim selection parameters."""

    name: str = "Adam"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerConfig(BaseConfig):
    """Wrapper around torch.optim.lr_scheduler parameters."""

    name: str = "cosine"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LossFunctionConfig(BaseConfig):
    """Wrapper around torch.nn.functional parameters."""

    name: str = "cross_entropy"
    params: Dict[str, Any] = field(default_factory=dict)
