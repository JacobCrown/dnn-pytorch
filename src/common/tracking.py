"""Training run tracking backends."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Protocol

from src.common.config import TrackingConfig

logger = logging.getLogger(__name__)


class Tracker(Protocol):
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None: ...

    def log_artifact(self, name: str, path: str | Path) -> None: ...

    def close(self) -> None: ...


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


class LocalJsonTracker:
    """Append-only local tracker stored inside the experiment directory."""

    def __init__(self, *, run_dir: str | Path, task_name: str) -> None:
        self.run_dir = Path(run_dir)
        self.task_name = task_name
        self.tracking_dir = self.run_dir / "tracking"
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.tracking_dir / "events.jsonl"

    def _write_event(self, payload: dict[str, Any]) -> None:
        payload["task_name"] = self.task_name
        with self.events_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._write_event({"type": "metrics", "step": step, "metrics": metrics})

    def log_artifact(self, name: str, path: str | Path) -> None:
        self._write_event({"type": "artifact", "name": name, "path": Path(path)})

    def close(self) -> None:
        return None


class NullTracker:
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        return None

    def log_artifact(self, name: str, path: str | Path) -> None:
        return None

    def close(self) -> None:
        return None


def build_tracker(
    *,
    config: TrackingConfig,
    run_dir: str | Path,
    task_name: str,
) -> Tracker:
    if config.wandb_enabled:
        logger.warning("W&B tracking is configured but not initialized yet; using local tracking.")
    if config.local_enabled:
        return LocalJsonTracker(run_dir=run_dir, task_name=task_name)
    return NullTracker()
