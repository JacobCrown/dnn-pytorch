from dataclasses import dataclass, field

from src.common.metrics import IoUMetrics
from src.core.metrics import BaseMetrics


@dataclass
class SegmentationMetrics(BaseMetrics):
    """Metrics specific to segmentation tasks."""

    ious: IoUMetrics = field(default_factory=IoUMetrics)
