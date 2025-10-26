"""
Generic augmentation pipeline builder for Albumentations.

This module provides two entry points:
- build_augmentation_pipeline for dataclass-driven configs
- get_augmentations for Hydra DictConfig-driven YAML
"""

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.common.config import AugmentationConfig


def _create_transform_from_spec(spec: dict[str, Any]) -> Any:
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
        nested_transforms: list[Any] = [
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


def get_augmentations(
    cfg: list[AugmentationConfig], append_to_tensor: bool = True
) -> A.Compose:
    """Parse `cfg` and return an Albumentations `Compose`.

    Args:
        cfg: List of AugmentationConfig objects
        append_to_tensor: if True, append `ToTensorV2()` at the end
    """
    transforms: list[Any] = []
    for aug_spec in cfg:
        transforms.append(_create_transform_from_spec(aug_spec.to_dict()))

    if append_to_tensor:
        transforms.append(ToTensorV2())
    return A.Compose(transforms)
