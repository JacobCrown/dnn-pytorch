from dataclasses import dataclass, field
from typing import List, Dict, Any

from src.common.metrics import BaseMetrics
from src.common.enums import TaskType


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
    
    def add_accuracy(self, phase: str, accuracy: float) -> None:
        """Add accuracy for a specific phase."""
        if phase in self.accuracies:
            self.accuracies[phase].append(accuracy)
            # Also add to the generic metrics structure
            self.add_metric(phase, "accuracy", accuracy)
    
    def get_latest_accuracy(self, phase: str) -> float:
        """Get the latest accuracy for a specific phase."""
        if phase in self.accuracies and self.accuracies[phase]:
            return self.accuracies[phase][-1]
        return 0.0
