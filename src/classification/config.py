"""Configuration classes for classification tasks."""

from dataclasses import dataclass, field

from src.common.config import (AugmentationConfig, BaseConfig, BaseDataConfig,
                               BaseTrainingConfig)


@dataclass
class ClassificationClassMappingConfig(BaseConfig):
    """Configuration describing how source folders merge into one class."""

    name: str = "class_name"
    map_classes: list[str] = field(default_factory=list)


@dataclass
class ClassificationDataConfig(BaseDataConfig):
    """Configuration for classification datasets."""

    class_mapping: list[ClassificationClassMappingConfig] | None = None


@dataclass
class ClassificationModelConfig(BaseConfig):
    """Configuration for model architecture."""

    name: str = "resnet18"
    num_classes: int | None = None
    in_channels: int = 3
    pretrained: bool = True
    dropout: float = 0.2
    freeze_backbone: bool = False
    freeze_backbone_bn_eval: bool = False
    width: int = 224
    height: int = 224
    preprocessing: list[AugmentationConfig] = field(default_factory=list)


@dataclass
class ClassificationConfig(BaseConfig):
    """Main configuration class for classification tasks."""

    model: ClassificationModelConfig = field(default_factory=ClassificationModelConfig)
    train: BaseTrainingConfig = field(default_factory=BaseTrainingConfig)
    augmentations: list[AugmentationConfig] = field(default_factory=list)
    data: ClassificationDataConfig = field(default_factory=ClassificationDataConfig)


@dataclass
class ClassificationTestConfig(BaseConfig):
    """Configuration class for classification test-time evaluation."""

    model: ClassificationModelConfig = field(default_factory=ClassificationModelConfig)
    train: BaseTrainingConfig = field(default_factory=BaseTrainingConfig)
    data: ClassificationDataConfig = field(default_factory=ClassificationDataConfig)
    eval_checkpoint: str | None = None
