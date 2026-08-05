from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class Metrics:
    name: ClassVar[str] = ""
    train: list[float] = field(default_factory=list)
    val: list[float] = field(default_factory=list)
    test: list[float] = field(default_factory=list)
    def __post_init__(self) -> None:
        if not self.name:
            raise NotImplementedError(f"{type(self).__name__} must define name")
    @property
    def get_latest_metrics(self) -> dict[str, float]:
        return {
            f"train_{self.name}": self.train[-1],
            f"val_{self.name}": self.val[-1],
        }

    def to_dict(self):
        return {
            f"train_{self.name}": self.train,
            f"val_{self.name}": self.val,
            f"test_{self.name}": self.test,
        }


@dataclass
class LossMetrics(Metrics):
    name: ClassVar[str] = "loss"


@dataclass
class AccuracyMetrics(Metrics):
    name: ClassVar[str] = "accuracy"
