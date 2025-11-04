"""Utility to construct segmentation models from Hydra config using segmentation-models-pytorch."""

import torch.nn as nn
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2

from src.segmentation.config import SegmentationModelConfig


from typing import Tuple
import albumentations as A


def get_model(cfg: SegmentationModelConfig) -> Tuple[nn.Module, A.Compose]:
    model = smp.create_model(
        cfg.architecture,
        in_channels=cfg.in_channels,
        classes=cfg.classes,
        **cfg.params,
    )

    preprocess_settings = smp.encoders.get_preprocessing_params(
        cfg.params["encoder_name"], pretrained=cfg.params["encoder_weights"]
    )
    mean = preprocess_settings["mean"]
    std = preprocess_settings["std"]
    if isinstance(mean, int):
        mean = (mean,) * cfg.in_channels
    if isinstance(std, int):
        std = (std,) * cfg.in_channels

    preprocessing = A.Compose(
        [
            A.Normalize(mean=mean, std=std),
            A.Resize(cfg.height, cfg.width),
            ToTensorV2(transpose_mask=True),
        ]
    )

    return model, preprocessing
