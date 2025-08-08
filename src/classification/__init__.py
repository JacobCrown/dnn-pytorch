"""The classification module provides a complete framework for image classification tasks.

The module includes:
- Dataset: ClassificationDataset for loading image data with class labels
- Augmentations: Various data augmentation transforms and techniques
- Models: Pre-built classification models (ResNet, EfficientNet, ViT, etc.)
- Trainer: Complete training loop with validation and evaluation
- Config: Configuration management with dataclasses
- Utils: Utility functions for data loading, metrics, and visualization
"""

from .dataset import ClassificationDataset
from .trainer import ClassificationTrainer
from .config import (
    ModelConfig,
    DataConfig,
    TrainingConfig,
    ExperimentConfig,
    ClassificationConfig,
    get_default_config,
    get_resnet_config,
    get_efficientnet_config,
    get_vit_config,
)
from ..common import utils as common_utils

__all__ = [
    # Dataset
    "ClassificationDataset",
    # Trainer
    "ClassificationTrainer",
    # Config
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "ClassificationConfig",
    "get_default_config",
    "get_resnet_config",
    "get_efficientnet_config",
    "get_vit_config",
    # Common utils module (namespaced)
    "common_utils",
]
