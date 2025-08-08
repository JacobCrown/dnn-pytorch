"""Configuration classes for classification tasks."""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    name: str = "resnet50"
    num_classes: int = 10
    pretrained: bool = True
    dropout: float = 0.2
    input_channels: int = 3

    # Model-specific parameters
    model_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    data_root: str = "data"
    train_dir: str = "train"
    val_dir: Optional[str] = None
    test_dir: Optional[str] = None

    # Image preprocessing
    image_size: Tuple[int, int] = (224, 224)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Data augmentation
    use_augmentation: bool = True
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0

    # Data loading
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True

    # Dataset splitting
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Supported image extensions
    extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    )


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Training parameters
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-4

    # Optimizer
    optimizer: str = "adam"  # "adam", "sgd", "adamw"
    momentum: float = 0.9  # for SGD

    # Scheduler
    scheduler: str = "cosine"  # "cosine", "step", "plateau", "none"
    scheduler_params: Dict[str, Any] = field(default_factory=dict)

    # Loss function
    loss_function: str = "cross_entropy"  # "cross_entropy", "focal", "label_smoothing"
    label_smoothing: float = 0.0

    # Early stopping
    early_stopping_patience: Optional[int] = 10

    # Checkpointing
    save_freq: int = 1
    save_best_only: bool = True

    # Mixed precision
    use_amp: bool = False

    # Gradient clipping
    gradient_clip: Optional[float] = None


@dataclass
class ExperimentConfig:
    """Configuration for experiment management."""

    # Experiment details
    experiment_name: str = "classification_experiment"
    base_dir: str = "experiments"

    # Logging
    log_dir: str = "logs"
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: Optional[str] = None

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Device
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"

    # Evaluation
    eval_freq: int = 1
    save_predictions: bool = False


@dataclass
class ClassificationConfig:
    """Main configuration class for classification tasks."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate data ratios
        total_ratio = self.data.train_ratio + self.data.val_ratio + self.data.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Data ratios must sum to 1.0, got {total_ratio}")

        # Validate image size
        if len(self.data.image_size) != 2:
            raise ValueError("Image size must be a tuple of 2 integers")

        # Validate learning rate
        if self.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        # Validate batch size
        if self.data.batch_size <= 0:
            raise ValueError("Batch size must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "experiment": self.experiment.__dict__,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ClassificationConfig":
        """Create configuration from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            experiment=ExperimentConfig(**config_dict.get("experiment", {})),
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
