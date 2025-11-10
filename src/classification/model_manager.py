"""Utility to construct and optionally freeze timm models from Hydra config."""

import logging
from pathlib import Path

import albumentations as A
import timm
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from timm.data.config import resolve_data_config

from src.classification.config import ClassificationModelConfig
from src.common.utils import DEVICE, get_opencv_interpolation_flag
from src.core.model_manager import BaseModelManager

logger = logging.getLogger(__name__)


class ClassificationModelManager(BaseModelManager):
    """Classification model class."""

    def __init__(self, cfg: ClassificationModelConfig):
        """Initialize the classification model."""
        self._model_cfg = cfg
        super().__init__()

    @property
    def model_cfg(self) -> ClassificationModelConfig:
        return self._model_cfg

    def create_model(self) -> nn.Module:
        return timm.create_model(
            self.model_cfg.name,
            pretrained=bool(self.model_cfg.pretrained),
            num_classes=int(self.model_cfg.num_classes),
            in_chans=int(self.model_cfg.in_channels),
            drop_rate=float(self.model_cfg.dropout),
        )

    def create_preprocessing(self) -> A.Compose:
        data_config = resolve_data_config(
            args={"in_chans": int(self.model_cfg.in_channels)}, model=self.model
        )
        mean = data_config.get("mean")
        std = data_config.get("std")
        _, h, w = data_config.get("input_size")
        _ = get_opencv_interpolation_flag(
            data_config.get("interpolation")
        )  # TODO: Interpolation flag gives worse results, Investigate later
        height = self.model_cfg.height
        width = self.model_cfg.width
        if height != h or width != w:
            logger.warning(
                f"WARNING: You requested input size height={height} x width={width}, "
                f"but this model was trained using input size height={h} x width={w}. "
            )
        return A.Compose(
            [
                A.Resize(height, width),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

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
