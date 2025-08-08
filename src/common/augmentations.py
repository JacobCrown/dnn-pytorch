"""
Generic augmentation pipeline builder for Albumentations.

This module provides two entry points:
- build_augmentation_pipeline for dataclass-driven configs
- get_augmentations for Hydra DictConfig-driven YAML
"""

from typing import Any, Dict, List, Tuple, cast

import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig

from src.common.config import StageAugmentConfig


def _create_transform_from_spec(spec: Dict[str, Any]) -> Any:
    """
    Recursively creates an Albumentations transform from a spec dictionary.

    This helper can instantiate nested transforms like `OneOf`.

    Args:
        spec: A dictionary with 'name' and 'params' keys.

    Returns:
        An instantiated Albumentations transform object.
    """
    name = spec["name"]
    params = spec.get("params", {})

    # Handle `OneOf` compositor, which contains a list of other transforms
    if name == "OneOf":
        nested_transforms: List[Any] = [
            _create_transform_from_spec(nested_spec)
            for nested_spec in params.get("transforms", [])
        ]
        # Pass other params like `p` to the OneOf constructor
        oneof_params = {k: v for k, v in params.items() if k != "transforms"}
        return A.OneOf(transforms=nested_transforms, **oneof_params)

    try:
        transform_class = getattr(A, name)
    except AttributeError as exc:
        raise AttributeError(
            f"Transform '{name}' not found in Albumentations."
        ) from exc

    return transform_class(**params)


def build_augmentation_pipeline(
    stage_config: StageAugmentConfig,
    img_size: Tuple[int, int],
) -> A.Compose:
    """
    Constructs a complete Albumentations pipeline from a stage configuration.

    The pipeline is built in a specific order:
    1. Augmentation transforms (e.g., flips, rotations).
    2. Resizing to the target `img_size`.
    3. Pre-processing transforms (e.g., normalization).
    4. Conversion to a PyTorch tensor.

    Args:
        stage_config: The configuration object for a specific stage (e.g., train).
        img_size: The target image size (height, width).

    Returns:
        A composed Albumentations pipeline.
    """
    transforms: List[Any] = []

    # 1. Add augmentation transforms
    for aug_spec in stage_config.augment:
        transform = _create_transform_from_spec(dataclass_to_dict(aug_spec))
        transforms.append(transform)

    # 2. Add resizing
    transforms.append(A.Resize(height=img_size[0], width=img_size[1]))

    # 3. Add pre-processing transforms
    for preprocess_spec in stage_config.preprocess:
        transform = _create_transform_from_spec(dataclass_to_dict(preprocess_spec))
        transforms.append(transform)

    # 4. Add conversion to tensor
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert dataclass instance to dict or return obj if already dict."""
    from dataclasses import asdict as dc_asdict, is_dataclass as dc_is_dataclass

    # Only convert instances, not classes
    if dc_is_dataclass(obj) and not isinstance(obj, type):
        return cast(Dict[str, Any], dc_asdict(obj))
    if isinstance(obj, dict):
        return obj
    raise TypeError("Expected dataclass instance or dict for augmentation spec")


# =============================================================================
# 2. Simplified DictConfig → Albumentations pipeline (for Hydra-driven training)
# =============================================================================
def _create_transform_from_cfg(name: str, params: dict) -> Any:
    """Create Albumentations transform from plain YAML dict entry."""
    if name == "OneOf":
        params_copy = dict(params) if params is not None else {}
        prob = params_copy.pop("p", 0.5)
        nested_transforms: List[Any] = [
            _create_transform_from_cfg(nested_name, nested_params or {})
            for nested_name, nested_params in params_copy.items()
        ]
        return A.OneOf(nested_transforms, p=prob)

    transform_cls = getattr(A, name, None)
    if transform_cls is None:
        raise ValueError(f"Albumentations transform '{name}' not found")
    return transform_cls(**(params or {}))


def get_augmentations(cfg: DictConfig, append_to_tensor: bool = True) -> A.Compose:
    """Parse `cfg.augmentations` and return an Albumentations `Compose`.

    Args:
        cfg: Hydra config with an `augmentations` node
        append_to_tensor: if True, append `ToTensorV2()` at the end
    """
    transforms: List[Any] = []
    for name, params in cfg.augmentations.items():
        transforms.append(_create_transform_from_cfg(name, params or {}))

    if append_to_tensor:
        transforms.append(ToTensorV2())
    return A.Compose(transforms)
