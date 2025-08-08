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

    # Supported image extensions
    extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    )


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for training."""

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
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

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate learning rate
        if self.train.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "train": self.train.__dict__,
            "augmentations": self.augmentations.__dict__,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ClassificationConfig":
        """Create configuration from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            train=TrainingConfig(**config_dict.get("train", {})),
            augmentations=[
                AugmentationConfig(**aug)
                for aug in config_dict.get("augmentations", [])
            ],
        )

    def save(self, filepath: str) -> None:
        """Save configuration to file."""
        import json

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, filepath: str) -> "ClassificationConfig":
        """Load configuration from file."""
        import json

        with open(filepath, "r") as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)
