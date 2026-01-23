"""Utility to construct and optionally freeze timm models from Hydra config."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import albumentations as A
import torch.nn as nn

from src.common.config import BaseConfig
from src.common.utils import DEVICE


class BaseModelManager[ModelConfigType: BaseConfig](ABC):
    """Base model manager class."""

    def __init__(self, model_cfg: ModelConfigType) -> None:
        """Initialize the base model manager."""
        self.model_cfg: ModelConfigType = model_cfg
        self.model: nn.Module = self.create_model().to(DEVICE)
        self.preprocessing: A.Compose = self.get_preprocessing()

    @abstractmethod
    def create_model(self) -> nn.Module:
        pass

    @abstractmethod
    def get_preprocessing(self) -> A.Compose:
        pass

    @abstractmethod
    def load_preprocessing(self, preprocessing: dict) -> None:
        pass

    @abstractmethod
    def get_model_state_dict(self) -> dict:
        pass

    @abstractmethod
    def load_model_state_dict(self, state_dict: dict) -> None:
        pass

    @abstractmethod
    def check_model_config(self, config: dict) -> None:
        """Check if the model config is compatible with the current model config."""
        pass

    @classmethod
    @abstractmethod
    def from_checkpoint_path(cls, checkpoint_path: Path) -> Self:
        pass
