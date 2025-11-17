"""
This file is supposed to be used to predict the segmentation mask of an image. It should work on it's own, without using
any other files from the project.
"""

from pathlib import Path

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL.Image import Image


class SegmentationPredictor:
    """Predictor class for segmentation models."""

    def __init__(self, checkpoint_path: Path):
        """Initialize the predictor."""
        self.checkpoint_path: Path = checkpoint_path
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        (
            self.model,
            self.preprocessing,
            self.class_idx_to_name,
            self.class_idx_to_rgb,
        ) = self.load_model_and_preprocessing()

    def load_model_and_preprocessing(
        self,
    ) -> tuple[nn.Module, A.Compose, dict[int, str], dict[int, tuple[int, int, int]]]:
        checkpoint = torch.load(
            str(self.checkpoint_path), map_location=self.device, weights_only=False
        )
        model_config = checkpoint["model_config"]
        model = smp.create_model(
            arch=model_config["architecture"],
            encoder_name=model_config["encoder_name"],
            encoder_weights=model_config["encoder_weights"],
            in_channels=int(model_config["in_channels"]),
            classes=int(model_config["classes"]),
            **model_config["params"],
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        # preprocessing
        data_config = smp.encoders.get_preprocessing_params(
            encoder_name=model_config["encoder_name"],
            pretrained=model_config["encoder_weights"],
        )
        mean = data_config.get("mean")
        std = data_config.get("std")
        if isinstance(mean, int):
            mean = (mean,) * model_config["in_channels"]
        if isinstance(std, int):
            std = (std,) * model_config["in_channels"]
        height = model_config["height"]
        width = model_config["width"]
        preprocessing = A.Compose(
            [
                A.Resize(height, width),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(transpose_mask=True),
            ]
        )

        class_idx_to_rgb = checkpoint["class_idx_to_rgb"]
        class_idx_to_name = checkpoint["class_idx_to_name"]

        return model, preprocessing, class_idx_to_name, class_idx_to_rgb

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

    def predict_class(self, input: Image | np.ndarray) -> str:
        pass
