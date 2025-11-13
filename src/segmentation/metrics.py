from dataclasses import dataclass, field

from src.core.metrics import BaseMetrics


@dataclass
class SegmentationMetrics(BaseMetrics):
    """Metrics specific to segmentation tasks."""

    ious: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        self.ious = {
            "train": [],
            "val": [],
            "test": [],
        }
