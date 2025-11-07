"""Configuration classes for classification tasks."""

from dataclasses import dataclass, field

from src.common.config import (
    AugmentationConfig,
    BaseConfig,
    BaseDataConfig,
    BaseTrainingConfig,
)


@dataclass
class ClassificationModelConfig(BaseConfig):
    """Configuration for model architecture."""

    name: str = "resnet18"
    num_classes: int = 10
    in_channels: int = 3
    pretrained: bool = True
    dropout: float = 0.2
    width: int = 224
    height: int = 224


@dataclass
class ClassificationConfig(BaseConfig):
    """Main configuration class for classification tasks."""

    model: ClassificationModelConfig = field(default_factory=ClassificationModelConfig)
    train: BaseTrainingConfig = field(default_factory=BaseTrainingConfig)
    augmentations: list[AugmentationConfig] = field(default_factory=list)
    data: BaseDataConfig = field(default_factory=BaseDataConfig)
