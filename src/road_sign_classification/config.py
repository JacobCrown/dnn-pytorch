"""Configuration classes for road sign classification and generation."""

from dataclasses import dataclass, field

from src.common.config import (
    AugmentationConfig,
    BaseConfig,
    BaseDataConfig,
    BaseTrainingConfig,
)

DEFAULT_ROAD_SIGN_BACKGROUNDS_DIR = "data/generation/backgrounds"


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
class RoadSignDataConfig(BaseDataConfig):
    """Data settings for balanced per-class sampling (train/val/test)."""

    samples_per_class: int = 5
    backgrounds_dir: str = DEFAULT_ROAD_SIGN_BACKGROUNDS_DIR
    background_sample_stride: int = 30
    preload_backgrounds: bool = True


@dataclass
class RoadSignGenerationConfig(BaseConfig):
    """Settings specific to synthetic dataset export."""

    target_per_class: int = 10
    output_dir: str = "generated_data"
    incremental: bool = True
    num_workers: int | None = None
    chunk_size: int = 16
    template_cache_size: int = 128
    background_cache_size: int = 8
    png_compress_level: int = 1
    start_method: str | None = None
    backgrounds_dir: str | None = None
    background_sample_stride: int | None = None
    #: If true, after YAML augmentations also resize to ``model.width`` x ``model.height`` (e.g. 224).
    #: If false (default), saved PNGs keep whatever size the augment pipeline produces.
    resize_to_model: bool = False


@dataclass
class ClassificationConfig(BaseConfig):
    """Main configuration class for classification tasks."""

    model: ClassificationModelConfig = field(default_factory=ClassificationModelConfig)
    train: BaseTrainingConfig = field(default_factory=BaseTrainingConfig)
    augmentations: list[AugmentationConfig] = field(default_factory=list)
    family_search_results_dir: str | None = None
    data: RoadSignDataConfig = field(default_factory=RoadSignDataConfig)
    generation: RoadSignGenerationConfig = field(
        default_factory=RoadSignGenerationConfig
    )
