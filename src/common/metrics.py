from dataclasses import dataclass, field


@dataclass
class LossMetrics:
    """Loss metrics for training, validation, and test sets. Each item is the loss value for a given epoch."""

    train: list[float] = field(default_factory=list)
    val: list[float] = field(default_factory=list)
    test: list[float] = field(default_factory=list)


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for training, validation, and test sets. Each item is the accuracy value for a given epoch."""

    train: list[float] = field(default_factory=list)
    val: list[float] = field(default_factory=list)
    test: list[float] = field(default_factory=list)
