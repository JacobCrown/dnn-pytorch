"""Standalone segmentation inference from saved checkpoints."""

from pathlib import Path

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from PIL.Image import Image

from src.common.augmentations import get_augmentations
from src.common.config import AugmentationConfig


class SegmentationPredictor:
    """Load a segmentation checkpoint and predict class masks for images."""

    def __init__(self, checkpoint_path: Path):
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

        raw = model_config.get("preprocessing")
        if not raw:
            raise ValueError(
                "Checkpoint model_config must contain a non-empty preprocessing list."
            )
        preprocessing = get_augmentations(
            [AugmentationConfig.from_dict(item) for item in raw],
        )

        class_idx_to_rgb = checkpoint["class_idx_to_rgb"]
        class_idx_to_name = checkpoint["class_idx_to_name"]

        return model, preprocessing, class_idx_to_name, class_idx_to_rgb

    def _preprocess(self, input: Image | np.ndarray) -> torch.Tensor:
        return self.preprocessing(image=input)["image"].unsqueeze(0).to(self.device)

    def _predict(self, input: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.model(input).argmax(dim=1, keepdim=True)

    def predict(self, input: Image | np.ndarray) -> torch.Tensor:
        batch = self._preprocess(input)
        return self._predict(batch)

    def predict_class_name(self, class_index: int) -> str:
        return self.class_idx_to_name[class_index]
