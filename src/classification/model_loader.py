"""Utility to construct and optionally freeze timm models from Hydra config."""

from typing import Iterable, Tuple
import logging

import albumentations as A
import timm
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from timm.data.config import resolve_data_config

from src.classification.config import ClassificationModelConfig

logger = logging.getLogger(__name__)


def _matches_any_prefix(name: str, prefixes: Iterable[str]) -> bool:
    return any(name.startswith(pfx) for pfx in prefixes)


def apply_freezing(
    model: nn.Module,
    freeze_backbone: bool = False,
    freeze_layers: Iterable[str] | None = None,
    unfreeze_layers: Iterable[str] | None = None,
    partial_bn: bool = False,
) -> None:
    """Freeze layers according to configuration.

    - freeze_backbone: freeze all except classifier/head
    - freeze_layers: freeze params whose name starts with any of provided prefixes
    - unfreeze_layers: ensure these stay trainable
    - partial_bn: set BatchNorm layers to eval and disable affine grads (common trick)
    """
    freeze_layers = list(freeze_layers or [])
    unfreeze_layers = list(unfreeze_layers or [])

    if freeze_backbone:
        # Attempt to identify classifier/head modules used by timm
        head_modules = []
        for attr in ["classifier", "fc", "head", "heads", "classifier_head"]:
            if hasattr(model, attr):
                head_modules.append(attr)
        # Freeze everything
        for name, param in model.named_parameters():
            param.requires_grad = False
        # Unfreeze heads
        for attr in head_modules:
            module = getattr(model, attr, None)
            if module is None:
                continue
            for name, param in module.named_parameters():
                param.requires_grad = True

    # Freeze specific prefixes
    if freeze_layers:
        for name, param in model.named_parameters():
            if _matches_any_prefix(name, freeze_layers):
                param.requires_grad = False

    # Unfreeze allowlist
    if unfreeze_layers:
        for name, param in model.named_parameters():
            if _matches_any_prefix(name, unfreeze_layers):
                param.requires_grad = True

    # Partial BN handling
    if partial_bn:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                if module.affine:
                    if module.weight is not None:
                        module.weight.requires_grad = False
                    if module.bias is not None:
                        module.bias.requires_grad = False


def get_model(cfg: ClassificationModelConfig) -> Tuple[nn.Module, A.Compose]:
    """Create a timm model and preprocessing according to cfg.model, apply freezing."""
    model = timm.create_model(
        cfg.name,
        pretrained=bool(cfg.pretrained),
        num_classes=int(cfg.num_classes),
        in_chans=int(cfg.in_channels),
        drop_rate=float(cfg.dropout),
    )

    # Apply optional freezing controls if present in cfg
    freeze_backbone = bool(cfg.freeze_backbone)
    freeze_layers = cfg.freeze_layers or []
    unfreeze_layers = cfg.unfreeze_layers or []
    partial_bn = bool(cfg.partial_bn)

    apply_freezing(
        model,
        freeze_backbone=freeze_backbone,
        freeze_layers=freeze_layers,
        unfreeze_layers=unfreeze_layers,
        partial_bn=partial_bn,
    )

    data_config = resolve_data_config(args={"in_chans": int(cfg.in_channels)}, model=model)
    mean = data_config.get("mean")
    std = data_config.get("std")
    in_channels, h, w = data_config.get("input_size")

    assert isinstance(mean, tuple) and isinstance(std, tuple) and isinstance(in_channels, int) and isinstance(h, int) and isinstance(w, int), "Input size must be a tuple of 3 elements: (in_channels, height, width)"
    height = (cfg.height or h)
    width = (cfg.width or w)

    if height != h or width != w:
        logger.warning(f"Height ({height}) or width ({width}) do not match the model input size ({h}x{w})."
                       "This might cause issues with the model training.")

    preprocessing = A.Compose(
        [
            A.Resize(height, width),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    return model, preprocessing
