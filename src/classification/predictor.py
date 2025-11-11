from pathlib import Path

import numpy as np
import torch
from PIL.Image import Image

from src.classification.model_manager import ClassificationModelManager
from src.common.utils import DEVICE
from src.core.predictor import BasePredictor


class ClassificationPredictor(BasePredictor):
    """Predictor class for classification models."""

    def __init__(self, model_path: Path):
        """Initialize the predictor."""
        self.model_manager = ClassificationModelManager.from_checkpoint_path(model_path)
        super().__init__(self.model_manager)

    def _preprocess(self, input: Image | np.ndarray) -> torch.Tensor:
        return self.preprocessing(image=input)["image"].unsqueeze(0).to(DEVICE)

    def _predict(self, input: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.inference_mode():
            return self.model(input)

    def _postprocess(self, input: torch.Tensor) -> torch.Tensor:
        return input
