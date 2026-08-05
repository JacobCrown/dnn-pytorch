from dataclasses import asdict, dataclass, field, fields
from typing import Any, Self

from src.common.metrics import LossMetrics


@dataclass
class BaseMetrics:
    """Base metrics class for all tasks."""

    losses: LossMetrics = field(default_factory=LossMetrics)

    def to_dict(self) -> dict[str, Any]:
        """Convert the metrics instance to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls: type[Self], data: dict[str, Any]) -> Self:
        """Create an instance of the metrics class from a dictionary.

        Args:
            data: Dictionary containing metrics data

        Returns:
            An instance of the metrics class
        """
        instance: Self = cls()
        for key, value in data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance

    def get_latest_metrics(self) -> dict[str, float]:
        """Generic method for obtaining latest metrics"""
        latest_metrics: dict[str, float] = {}
        for attr in fields(self):
            latest_metrics.update(getattr(self, attr.name).get_latest_metrics)
        return latest_metrics
