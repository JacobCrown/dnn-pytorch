"""
This file is supposed to be used to predict the class of an image. It should work on it's own, without using
any other files from the project.
"""

from pathlib import Path

import albumentations as A
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL.Image import Image
from timm.data.config import resolve_data_config


class ClassificationPredictor:
    """Predictor class for classification models."""

    def __init__(self, checkpoint_path: Path):
        """Initialize the predictor."""
        self.checkpoint_path: Path = checkpoint_path
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model, self.preprocessing = self.load_model_and_preprocessing()
        self.class_names: list[str] = self.load_class_names()

    def load_model_and_preprocessing(self) -> tuple[nn.Module, A.Compose]:
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

        # preprocessing
        data_config = resolve_data_config(
            args={"in_chans": int(model_config["in_channels"])}, model=model
        )
        mean = data_config.get("mean")
        std = data_config.get("std")
        height = model_config["height"]
        width = model_config["width"]
        preprocessing = A.Compose(
            [
                A.Resize(height, width),
                A.Normalize(mean=mean, std=std),
                A.ToTensorV2(),
            ]
        )
        return model, preprocessing

    def _preprocess(self, input: Image | np.ndarray) -> torch.Tensor:
        return self.preprocessing(image=input)["image"].unsqueeze(0).to(self.device)

    def _predict(self, input: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.model(input).argmax(dim=1, keepdim=True)

    def _postprocess(self, input: torch.Tensor) -> torch.Tensor:
        return input

    def predict(self, input: Image | np.ndarray) -> torch.Tensor:
        preprocess = self._preprocess(input)
        result = self._predict(preprocess)
        postprocess = self._postprocess(result)
        return postprocess
