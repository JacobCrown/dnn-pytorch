"""Convert runtime objects into JSON-safe plain Python values."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def to_plain_python(value: Any) -> Any:
    """Recursively convert configs, paths, and dataclasses into JSON-safe values."""
    if OmegaConf.is_config(value):
        container = OmegaConf.to_container(value, resolve=True, throw_on_missing=False)
        return to_plain_python(container)

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value) and not isinstance(value, type):
        return to_plain_python(asdict(value))

    if isinstance(value, dict):
        return {str(key): to_plain_python(item) for key, item in value.items()}

    if isinstance(value, list | tuple):
        return [to_plain_python(item) for item in value]

    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return value
