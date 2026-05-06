"""Utility to construct and optionally freeze timm models from Hydra config."""

import logging
from pathlib import Path

import albumentations as A
import timm
import torch
import torch.nn as nn

from src.common.augmentations import AugmentationConfig, get_augmentations
from src.common.utils import DEVICE, get_opencv_interpolation_flag
from src.core.model_manager import BaseModelManager
from src.road_sign_classification.config import ClassificationModelConfig

logger = logging.getLogger(__name__)


def build_classification_preprocessing(
    model_cfg: ClassificationModelConfig,
    *,
    include_normalize: bool = True,
    include_tensor: bool = True,
) -> A.Compose:
    """Build preprocessing directly from timm pretrained metadata."""
    pretrained_cfg = timm.get_pretrained_cfg(model_cfg.name)
    mean = pretrained_cfg.mean
    std = pretrained_cfg.std
    _, h, w = pretrained_cfg.input_size
    _ = get_opencv_interpolation_flag(
        pretrained_cfg.interpolation
    )  # TODO: Interpolation flag gives worse results, investigate later.

    height = model_cfg.height
    width = model_cfg.width
    if height != h or width != w:
        logger.warning(
            "WARNING: You requested input size height=%s x width=%s, "
            "but this model was trained using input size height=%s x width=%s.",
            height,
            width,
            h,
            w,
        )

    transforms: list[A.BasicTransform] = [A.Resize(height, width)]
    if include_normalize:
        transforms.append(A.Normalize(mean=mean, std=std))
    if include_tensor:
        transforms.append(A.ToTensorV2())
    return A.Compose(transforms)


class ClassificationModelManager(BaseModelManager[ClassificationModelConfig]):
    """Classification model class."""

    def __init__(self, cfg: ClassificationModelConfig):
        """Initialize the classification model."""
        super().__init__(cfg)

    def create_model(self) -> nn.Module:
        return timm.create_model(
            self.model_cfg.name,
            pretrained=bool(self.model_cfg.pretrained),
            num_classes=int(self.model_cfg.num_classes),
            in_chans=int(self.model_cfg.in_channels),
            drop_rate=float(self.model_cfg.dropout),
        )

    def get_preprocessing(self) -> A.Compose:
        return build_classification_preprocessing(
            self.model_cfg,
            include_normalize=True,
            include_tensor=True,
        )

    def load_preprocessing(self, preprocessing: dict) -> None:
        """Match ``BaseModelManager`` API; timm-based pipeline is default."""
        if isinstance(preprocessing, list) and len(preprocessing) > 0:
            self.preprocessing = get_augmentations(
                [AugmentationConfig.from_dict(a) for a in preprocessing]
            )
            return
        self.preprocessing = self.get_preprocessing()

    def get_model_state_dict(self) -> dict:
        return self.model.state_dict()

    def load_model_state_dict(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict)

    def check_model_config(self, config: dict) -> None:
        if config["name"] != self.model_cfg.name:
            raise ValueError(
                f"Trained model config name ({config['name']}) does not match the current model config name ({self.model_cfg.name})"
            )
        if config["num_classes"] != self.model_cfg.num_classes:
            raise ValueError(
                f"Trained model config num_classes ({config['num_classes']}) does not match the current model config num_classes ({self.model_cfg.num_classes})"
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

    @classmethod
    def from_checkpoint_path(
        cls, checkpoint_path: Path
    ) -> "ClassificationModelManager":
        checkpoint = torch.load(
            str(checkpoint_path), map_location=DEVICE, weights_only=False
        )
        config = ClassificationModelConfig.from_dict(checkpoint["model_config"])
        return cls(config)
