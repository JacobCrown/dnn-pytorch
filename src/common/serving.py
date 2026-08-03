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
    output_mask_path: str | None = None,
    class_pixel_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task": "segmentation",
        "image_path": image_path,
        "mask_shape": list(mask_shape),
    }
    if output_mask_path is not None:
        payload["output_mask_path"] = output_mask_path
    if class_pixel_counts is not None:
        payload["class_pixel_counts"] = class_pixel_counts
    return payload


def to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
