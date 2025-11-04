from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.common.config import (
    AugmentationConfig,
    BaseConfig,
    BaseTrainingConfig,
)


@dataclass
class ClassConfig(BaseConfig):
    """Configuration for the segmentation dataset."""

    name: str = "class_name"
    pixel_value: List[int] = field(default_factory=list)


@dataclass
class SegmentationDataConfig(BaseConfig):
    """Configuration for the segmentation dataset."""

    root: str = "data/segmentation"
    train_dir: str = "train"
    val_dir: str = "val"
    test_dir: Optional[str] = None
    extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    )
    mask_extensions: List[str] = field(default_factory=lambda: [".png"])
    max_samples: Optional[int] = None
    class_config: List[ClassConfig] = field(default_factory=list)


@dataclass
class SegmentationModelConfig(BaseConfig):
    """Configuration for the segmentation model."""

    architecture: str = "Unet"
    height: int = 224
    width: int = 224
    in_channels: int = 3
    classes: int = 2
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentationConfig(BaseConfig):
    """Main configuration class for segmentation tasks."""

    model: SegmentationModelConfig = field(default_factory=SegmentationModelConfig)
    train: BaseTrainingConfig = field(default_factory=BaseTrainingConfig)
    augmentations: List[AugmentationConfig] = field(default_factory=list)
    data: SegmentationDataConfig = field(default_factory=SegmentationDataConfig)
