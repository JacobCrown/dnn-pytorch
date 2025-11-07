"""Utility to construct and optionally freeze timm models from Hydra config."""

import logging
from abc import ABC, abstractmethod

import albumentations as A
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseModelManager(ABC):
    """Base model manager class."""

    def __init__(self):
        """Initialize the base model manager."""
        self.model = self.create_model()
        self.preprocessing = self.create_preprocessing()

    @abstractmethod
    def create_model(self) -> nn.Module:
        pass

    @abstractmethod
    def create_preprocessing(self) -> A.Compose:
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
