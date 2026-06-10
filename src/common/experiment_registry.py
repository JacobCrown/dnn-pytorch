"""Local experiment registry and comparison helpers."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from src.common.training_contracts import TrainingArtifacts

RUN_RECORD_NAME = "run.json"


@dataclass(frozen=True)
class RunRecord:
    task_name: str
    run_dir: Path
    metrics: dict[str, Any]
    artifacts: dict[str, Any]
    config: dict[str, Any]
    used_validation: bool


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


def write_run_record(
    *,
    task_name: str,
    artifacts: TrainingArtifacts,
    metrics: Any,
    config: dict[str, Any],
    used_validation: bool,
) -> Path:
    """Write a machine-readable record into an experiment directory."""
    record = {
        "task_name": task_name,
        "run_dir": str(artifacts.root_dir),
        "metrics": _jsonable(metrics),
        "artifacts": _jsonable(artifacts),
        "config": _jsonable(config),
        "used_validation": used_validation,
    }
    record_path = artifacts.root_dir / RUN_RECORD_NAME
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record_path


def _load_run_record(path: Path) -> RunRecord:
    payload = json.loads(path.read_text())
    return RunRecord(
        task_name=payload["task_name"],
        run_dir=Path(payload["run_dir"]),
        metrics=payload.get("metrics", {}),
        artifacts=payload.get("artifacts", {}),
        config=payload.get("config", {}),
        used_validation=bool(payload.get("used_validation", False)),
    )


def collect_run_records(paths: list[str | Path]) -> list[RunRecord]:
    """Collect run records from files, run directories, or parent directories."""
    records: list[RunRecord] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file() and path.name == RUN_RECORD_NAME:
            records.append(_load_run_record(path))
            continue
        if path.is_dir() and (path / RUN_RECORD_NAME).exists():
            records.append(_load_run_record(path / RUN_RECORD_NAME))
            continue
        if path.is_dir():
            records.extend(_load_run_record(record) for record in path.rglob(RUN_RECORD_NAME))
    return records


def _score(record: RunRecord) -> float:
    metrics = record.metrics
    if record.task_name == "classification" and metrics.get("best_val_accuracy") is not None:
        return float(metrics["best_val_accuracy"])
    if record.task_name == "segmentation" and metrics.get("best_val_iou") is not None:
        return float(metrics["best_val_iou"]) * 100.0
    if metrics.get("best_val_loss") is not None:
        return -float(metrics["best_val_loss"])
    return float("-inf")


def rank_run_records(records: list[RunRecord]) -> list[RunRecord]:
    return sorted(records, key=_score, reverse=True)


def format_leaderboard(records: list[RunRecord]) -> str:
    if not records:
        return "No run records found."

    lines = ["rank | task | score | run_dir | best_checkpoint", "-" * 64]
    for index, record in enumerate(rank_run_records(records), start=1):
        checkpoint = record.artifacts.get("best_checkpoint_path") or ""
        lines.append(
            f"{index} | {record.task_name} | {_score(record):.4f} | "
            f"{record.run_dir} | {checkpoint}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare local dnn-pytorch runs.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Run directories, run.json files, or parent directories to scan.",
    )
    args = parser.parse_args(argv)
    print(format_leaderboard(collect_run_records(args.paths)))


if __name__ == "__main__":
    main()
