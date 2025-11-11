from dataclasses import dataclass, field

from src.common.enums import TaskType
from src.core.metrics import BaseMetrics


@dataclass
class SegmentationMetrics(BaseMetrics):
    """Metrics specific to segmentation tasks."""

    task_type: TaskType = TaskType.SEGMENTATION
    ious: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.ious = {
            "train": [],
            "val": [],
            "test": [],
        }
