"""Utility to construct and optionally freeze timm models from Hydra config."""

import logging
from pathlib import Path

import albumentations as A
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from src.common.augmentations import AugmentationConfig, get_augmentations
from src.common.utils import DEVICE
from src.core.model_manager import BaseModelManager
from src.segmentation.config import SegmentationModelConfig

logger = logging.getLogger(__name__)


class SegmentationModelManager(BaseModelManager[SegmentationModelConfig]):
    """Segmentation model class."""

    def __init__(self, cfg: SegmentationModelConfig):
        """Initialize the segmentation model."""
        super().__init__(cfg)

    def create_model(self) -> nn.Module:
        return smp.create_model(
            self.model_cfg.architecture,
            encoder_name=self.model_cfg.encoder_name,
            encoder_weights=self.model_cfg.encoder_weights,
            in_channels=self.model_cfg.in_channels,
            classes=self.model_cfg.classes,
            **self.model_cfg.params,
        )

    def get_preprocessing(self) -> A.Compose:
        return get_augmentations(self.model_cfg.preprocessing)

    def load_preprocessing(self, preprocessing: dict) -> None:
        self.preprocessing = get_augmentations(
            [
                AugmentationConfig.from_dict(augmentation)
                for augmentation in preprocessing
            ]
        )

    def get_model_state_dict(self) -> dict:
        return self.model.state_dict()

    def load_model_state_dict(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict)

    def check_model_config(self, config: dict) -> None:
        if config["architecture"] != self.model_cfg.architecture:
            raise ValueError(
                f"Trained model config architecture ({config['architecture']}) does not match the current model config architecture ({self.model_cfg.architecture})"
            )
        if config["encoder_name"] != self.model_cfg.encoder_name:
            raise ValueError(
                f"Trained model config encoder_name ({config['encoder_name']}) does not match the current model config encoder_name ({self.model_cfg.encoder_name})"
            )
        if config["classes"] != self.model_cfg.classes:
            raise ValueError(
                f"Trained model config classes ({config['classes']}) does not match the current model config classes ({self.model_cfg.classes})"
            )
        if config["in_channels"] != self.model_cfg.in_channels:
            raise ValueError(
                f"Trained model config in_channels ({config['in_channels']}) does not match the current model config in_channels ({self.model_cfg.in_channels})"
            )
        if config["width"] != self.model_cfg.width:
            raise ValueError(
                f"Trained model config width ({config['width']}) does not match the current model config width ({self.model_cfg.width})"
            )
        if config["height"] != self.model_cfg.height:
            raise ValueError(
                f"Trained model config height ({config['height']}) does not match the current model config height ({self.model_cfg.height})"
            )
        current_preprocessing = [
            augmentation.to_dict() for augmentation in self.model_cfg.preprocessing
        ]
        if config["preprocessing"] != current_preprocessing:
            raise ValueError(
                f"Trained model config preprocessing ({config['preprocessing']}) does not match the current model config preprocessing ({current_preprocessing})"
            )

    @classmethod
    def from_checkpoint_path(cls, checkpoint_path: Path) -> "SegmentationModelManager":
        checkpoint = torch.load(
            str(checkpoint_path), map_location=DEVICE, weights_only=False
        )
        config = SegmentationModelConfig.from_dict(checkpoint["model_config"])
        return cls(config)
