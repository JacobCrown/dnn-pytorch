from dataclasses import dataclass, field
from typing import Dict, List

from src.common.enums import TaskType
from src.common.metrics import BaseMetrics


@dataclass
class ClassificationMetrics(BaseMetrics):
    """Metrics specific to classification tasks."""

    task_type: TaskType = TaskType.CLASSIFICATION
    accuracies: Dict[str, List[float]] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.accuracies = {
            "train": [],
            "val": [],
            "test": [],
        }
