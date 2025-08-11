"""Configuration classes for classification tasks."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from src.common.config import (
    AugmentationConfig,
    BaseConfig,
    LossFunctionConfig,
    OptimizerConfig,
    SchedulerConfig,
)


@dataclass
class ModelConfig(BaseConfig):
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
class DataConfig(BaseConfig):
    """Configuration for data loading and preprocessing."""

    root: str = "data"
    train_dir: str = "train"
    val_dir: Optional[str] = None
    test_dir: Optional[str] = None
    max_samples: Optional[int] = None

    # Supported image extensions
    extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    )


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for training."""

    epochs: int = 100
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True

    # Optimizer
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # Scheduler
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Loss function
    loss_function: LossFunctionConfig = field(default_factory=LossFunctionConfig)

    # Early stopping
    early_stopping_patience: Optional[int] = 10

    # Checkpointing
    save_freq: int = 1
    save_best_only: bool = True
    resume: Optional[str] = None

    # Mixed precision
    use_amp: bool = False

    # Gradient clipping
    gradient_clip: Optional[float] = None


@dataclass
class ClassificationConfig(BaseConfig):
    """Main configuration class for classification tasks."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    augmentations: List[AugmentationConfig] = field(default_factory=list)
    data: DataConfig = field(default_factory=DataConfig)
