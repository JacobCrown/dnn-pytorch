from dataclasses import dataclass, field
from typing import Any

from src.common.config import AugmentationConfig, BaseConfig, BaseTrainingConfig


@dataclass
class ClassConfig(BaseConfig):
    """Configuration for the segmentation dataset."""

    name: str = "class_name"
    pixel_value: list[int] = field(default_factory=list)


@dataclass
class SegmentationDataConfig(BaseConfig):
    """Configuration for the segmentation dataset."""

    root: str = "data/segmentation"
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: str | None = None
    extensions: list[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    )
    mask_extensions: list[str] = field(default_factory=lambda: [".png"])
    max_samples: int | None = None
    class_config: list[ClassConfig] = field(default_factory=list)


@dataclass
class SegmentationModelConfig(BaseConfig):
    """Configuration for the segmentation model."""

    architecture: str = "Unet"
    encoder_name: str = "resnet18"
    encoder_weights: str = "imagenet"
    height: int = 224
    width: int = 224
    in_channels: int = 3
    classes: int = 2
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentationConfig(BaseConfig):
    """Main configuration class for segmentation tasks."""

    model: SegmentationModelConfig = field(default_factory=SegmentationModelConfig)
    train: BaseTrainingConfig = field(default_factory=BaseTrainingConfig)
    augmentations: list[AugmentationConfig] = field(default_factory=list)
    data: SegmentationDataConfig = field(default_factory=SegmentationDataConfig)
