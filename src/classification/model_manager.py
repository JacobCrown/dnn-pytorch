"""Utility to construct and optionally freeze timm models from Hydra config."""

import logging
from pathlib import Path

import albumentations as A
import timm
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from src.classification.config import ClassificationModelConfig
from src.common.augmentations import AugmentationConfig, get_augmentations
from src.common.utils import DEVICE
from src.core.model_manager import BaseModelManager

logger = logging.getLogger(__name__)


class ClassificationModelManager(BaseModelManager[ClassificationModelConfig]):
    """Classification model class."""

    def __init__(self, cfg: ClassificationModelConfig):
        """Initialize the classification model."""
        self.classifier_modules: tuple[nn.Module, ...] = ()
        super().__init__(cfg)
        self._validate_freeze_config()
        self.classifier_modules = self._resolve_classifier_modules()
        self._apply_freeze_policy()

    def create_model(self) -> nn.Module:
        if self.model_cfg.num_classes is None:
            raise ValueError(
                "num_classes is not set — it must be derived from the training dataset before building the model."
            )
        return timm.create_model(
            self.model_cfg.name,
            pretrained=bool(self.model_cfg.pretrained),
            num_classes=int(self.model_cfg.num_classes),
            in_chans=int(self.model_cfg.in_channels),
            drop_rate=float(self.model_cfg.dropout),
        )

    def get_preprocessing(self) -> A.Compose:
        return get_augmentations(self.model_cfg.preprocessing)

    def load_preprocessing(self, preprocessing: dict) -> None:
        self.preprocessing = get_augmentations(
            [AugmentationConfig.from_dict(aug) for aug in preprocessing]
        )

    def get_model_state_dict(self) -> dict:
        return self.model.state_dict()

    def load_model_state_dict(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict)
        self._apply_freeze_policy()

    def check_model_config(self, config: dict) -> None:
        if config["name"] != self.model_cfg.name:
            raise ValueError(
                f"Trained model config name ({config['name']}) does not match the current model config name ({self.model_cfg.name})"
            )
        if config["num_classes"] != self.model_cfg.num_classes:
            raise ValueError(
                f"Trained model config num_classes ({config['num_classes']}) does not match the current model config num_classes ({self.model_cfg.num_classes})"
            )
        if config["in_channels"] != self.model_cfg.in_channels:
            raise ValueError(
                f"Trained model config in_channels ({config['in_channels']}) does not match the current model config in_channels ({self.model_cfg.in_channels})"
            )
        if config["width"] != self.model_cfg.width:
            raise ValueError(
                f"Trained model config width ({config['width']}) does not match the current model config width ({self.model_cfg.width})"
            )
        if config["height"] != self.model_cfg.height:
            raise ValueError(
                f"Trained model config height ({config['height']}) does not match the current model config height ({self.model_cfg.height})"
            )
        if config.get("freeze_backbone", False) != self.model_cfg.freeze_backbone:
            raise ValueError(
                "Trained model config freeze_backbone "
                f"({config.get('freeze_backbone', False)}) does not match the current "
                f"model config freeze_backbone ({self.model_cfg.freeze_backbone})"
            )
        if (
            config.get("freeze_backbone_bn_eval", False)
            != self.model_cfg.freeze_backbone_bn_eval
        ):
            raise ValueError(
                "Trained model config freeze_backbone_bn_eval "
                f"({config.get('freeze_backbone_bn_eval', False)}) does not match the current "
                "model config freeze_backbone_bn_eval "
                f"({self.model_cfg.freeze_backbone_bn_eval})"
            )
        current_preprocessing = [
            augmentation.to_dict() for augmentation in self.model_cfg.preprocessing
        ]
        if config["preprocessing"] != current_preprocessing:
            raise ValueError(
                f"Trained model config preprocessing ({config['preprocessing']}) does not match the current model config preprocessing ({current_preprocessing})"
            )

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return parameters that should be optimized."""
        return [param for param in self.model.parameters() if param.requires_grad]

    def prepare_model_for_training(self) -> None:
        """Restore train mode while keeping frozen backbone batch norms in eval."""
        self.model.train()
        if self.model_cfg.freeze_backbone_bn_eval:
            self._set_backbone_batchnorm_eval()

    def _validate_freeze_config(self) -> None:
        if (
            self.model_cfg.freeze_backbone_bn_eval
            and not self.model_cfg.freeze_backbone
        ):
            raise ValueError(
                "freeze_backbone_bn_eval requires freeze_backbone to be enabled"
            )

    def _resolve_classifier_modules(self) -> tuple[nn.Module, ...]:
        classifier = self.model.get_classifier()
        if isinstance(classifier, nn.Module):
            classifier_modules = (classifier,)
        elif isinstance(classifier, list | tuple):
            classifier_modules = tuple(
                module for module in classifier if isinstance(module, nn.Module)
            )
        else:
            classifier_modules = ()

        classifier_modules = tuple(
            module
            for module in classifier_modules
            if not isinstance(module, nn.Identity)
        )
        if not classifier_modules:
            raise ValueError(
                f"Could not resolve a classifier head for model '{self.model_cfg.name}'"
            )
        return classifier_modules

    def _apply_freeze_policy(self) -> None:
        if not self.model_cfg.freeze_backbone:
            return

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        for classifier_module in self.classifier_modules:
            for parameter in classifier_module.parameters():
                parameter.requires_grad = True

        if not self.get_trainable_parameters():
            raise ValueError(
                f"No trainable classifier parameters found for model '{self.model_cfg.name}'"
            )

        if self.model_cfg.freeze_backbone_bn_eval:
            self._set_backbone_batchnorm_eval()

        logger.info(
            "Applied freeze policy to model '%s' with %d trainable parameters",
            self.model_cfg.name,
            sum(param.numel() for param in self.get_trainable_parameters()),
        )

    def _set_backbone_batchnorm_eval(self) -> None:
        classifier_module_ids = {
            id(module)
            for classifier_module in self.classifier_modules
            for module in classifier_module.modules()
        }
        for module in self.model.modules():
            if (
                isinstance(module, _BatchNorm)
                and id(module) not in classifier_module_ids
            ):
                module.eval()

    @classmethod
    def from_checkpoint_path(
        cls, checkpoint_path: Path
    ) -> "ClassificationModelManager":
        """Build a model manager and load trained weights from a checkpoint."""
        checkpoint = torch.load(
            str(checkpoint_path), map_location=DEVICE, weights_only=False
        )
        config = ClassificationModelConfig.from_dict(checkpoint["model_config"])
        manager = cls(config)
        manager.load_model_state_dict(checkpoint["model_state_dict"])
        return manager
