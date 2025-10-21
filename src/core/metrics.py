from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from src.common.enums import TaskType


@dataclass
class BaseMetrics:
    """Base metrics class for all tasks."""

    task_type: TaskType = TaskType.UNDEFINED
    losses: Dict[str, List[float]] = field(default_factory=dict)

    def __post_init__(self):
        self.losses = {
            "train": [],
            "val": [],
            "test": [],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert the metrics instance to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseMetrics":
        """Create an instance of the metrics class from a dictionary.

        Args:
            data: Dictionary containing metrics data

        Returns:
            An instance of the metrics class
        """
        instance = cls()
        for key, value in data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        return instance
