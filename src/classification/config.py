"""Configuration classes for classification tasks."""

from dataclasses import dataclass, field
from typing import List

from src.common.config import (
    AugmentationConfig,
    BaseConfig,
    BaseDataConfig,
    BaseTrainingConfig,
)


@dataclass
class ClassificationModelConfig(BaseConfig):
    """Configuration for model architecture."""

    name: str = "resnet50"
    num_classes: int = 10
    pretrained: bool = True
    dropout: float = 0.2
    freeze_backbone: bool = False
    freeze_layers: List[str] = field(default_factory=list)
    unfreeze_layers: List[str] = field(default_factory=list)
    partial_bn: bool = False


@dataclass
class ClassificationConfig(BaseConfig):
    """Main configuration class for classification tasks."""

    model: ClassificationModelConfig = field(default_factory=ClassificationModelConfig)
    train: BaseTrainingConfig = field(default_factory=BaseTrainingConfig)
    augmentations: List[AugmentationConfig] = field(default_factory=list)
    data: BaseDataConfig = field(default_factory=BaseDataConfig)
