from abc import ABC, abstractmethod

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from PIL.Image import Image

from src.core.model_manager import BaseModelManager


class BasePredictor(ABC):
    """Base predictor class for different task types."""

    def __init__(self, model_manager: BaseModelManager) -> None:
        """Initialize the predictor."""
        self.model_manager: BaseModelManager = model_manager
        self.model: nn.Module = self.model_manager.model.eval()
        self.preprocessing: A.Compose = self.model_manager.preprocessing

    @abstractmethod
    def _preprocess(self, input: Image | np.ndarray) -> torch.Tensor:
        pass

    @abstractmethod
    def _predict(self, input: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _postprocess(self, input: torch.Tensor) -> torch.Tensor:
        pass

    def predict(self, input: Image | np.ndarray) -> torch.Tensor:
        preprocess = self._preprocess(input)
        result = self._predict(preprocess)
        postprocess = self._postprocess(result)
        return postprocess
