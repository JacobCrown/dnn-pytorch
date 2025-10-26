from dataclasses import dataclass, field

from src.common.enums import TaskType
from src.core.metrics import BaseMetrics


@dataclass
class ClassificationMetrics(BaseMetrics):
    """Metrics specific to classification tasks."""

    task_type: TaskType = TaskType.CLASSIFICATION
    accuracies: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.accuracies = {
            "train": [],
            "val": [],
            "test": [],
        }
