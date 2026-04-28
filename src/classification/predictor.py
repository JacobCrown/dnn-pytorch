"""
Standalone inference: loads timm classification weights and the training preprocessing
pipeline serialized in the checkpoint (`model_config.preprocessing`).
"""

from pathlib import Path

import albumentations as A
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL.Image import Image

from src.common.augmentations import get_augmentations
from src.common.config import AugmentationConfig


class ClassificationPredictor:
    """Predictor class for classification models."""

    def __init__(self, checkpoint_path: Path):
        """Initialize the predictor."""
        self.checkpoint_path: Path = checkpoint_path
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model, self.preprocessing, self.class_names = (
            self.load_model_and_preprocessing()
        )

    def load_model_and_preprocessing(self) -> tuple[nn.Module, A.Compose, list[str]]:
        checkpoint = torch.load(
            str(self.checkpoint_path), map_location=self.device, weights_only=False
        )
        model_config = checkpoint["model_config"]
        model = timm.create_model(
            model_name=model_config["name"],
            num_classes=int(model_config["num_classes"]),
            pretrained=model_config["pretrained"],
            in_chans=int(model_config["in_channels"]),
            drop_rate=float(model_config["dropout"]),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        raw = model_config.get("preprocessing")
        if not raw:
            raise ValueError(
                "Checkpoint model_config must contain a non-empty preprocessing list "
                "(same YAML list used for training augmentation pipeline)."
            )
        preprocessing = get_augmentations(
            [AugmentationConfig.from_dict(a) for a in raw],
        )

        return model, preprocessing, checkpoint["class_names"]

    def _preprocess(self, image: Image | np.ndarray) -> torch.Tensor:
        return self.preprocessing(image=image)["image"].unsqueeze(0).to(self.device)

    def _predict(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.model(batch).argmax(dim=1, keepdim=True)

    def _postprocess(self, out: torch.Tensor) -> torch.Tensor:
        return out

    def predict(self, image: Image | np.ndarray) -> torch.Tensor:
        pre = self._preprocess(image)
        result = self._predict(pre)
        return self._postprocess(result)

    def predict_class(self, image: Image | np.ndarray) -> str:
        prediction = self.predict(image)
        return self.class_names[int(prediction.item())]
