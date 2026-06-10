"""Small JSON helpers for local model serving CLIs."""

from __future__ import annotations

import json
from typing import Any


def classification_response(*, label: str, image_path: str) -> dict[str, Any]:
    return {
        "task": "classification",
        "image_path": image_path,
        "label": label,
    }


def segmentation_response(
    *,
    mask_shape: tuple[int, ...],
    image_path: str,
) -> dict[str, Any]:
    return {
        "task": "segmentation",
        "image_path": image_path,
        "mask_shape": list(mask_shape),
    }


def to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
