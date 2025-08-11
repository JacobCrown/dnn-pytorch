from dataclasses import dataclass, field
from typing import List, Dict, Any

from src.common.metrics import BaseMetrics


@dataclass
class ClassificationMetrics(BaseMetrics):
    accuracies: Dict[str, List[float]] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.accuracies = {
            "train": [],
            "val": [],
            "test": [],
        }
