from dataclasses import dataclass, field

from src.common.metrics import AccuracyMetrics
from src.core.metrics import BaseMetrics


@dataclass
class ClassificationMetrics(BaseMetrics):
    """Metrics specific to classification tasks."""

    accuracies: AccuracyMetrics = field(default_factory=AccuracyMetrics)
