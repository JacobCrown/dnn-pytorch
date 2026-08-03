"""Serve segmentation checkpoints locally as JSON CLI predictions."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from src.common.serving import segmentation_response, to_json
from src.segmentation.predictor import SegmentationPredictor


def _mask_to_rgb(
    class_indices: np.ndarray,
    class_idx_to_rgb: dict[int, list[int] | tuple[int, int, int]],
) -> np.ndarray:
    height, width = class_indices.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for class_index, color in class_idx_to_rgb.items():
        rgb[class_indices == int(class_index)] = color
    return rgb


def _count_classes(
    class_indices: np.ndarray,
    class_idx_to_name: dict[int, str],
) -> dict[str, int]:
    flat = class_indices.reshape(-1)
    counts = Counter(int(class_idx_to_name[index]) for index in flat.tolist())
    return dict(sorted(counts.items()))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run local segmentation inference.")
    parser.add_argument("checkpoint", help="Path to a segmentation checkpoint.")
    parser.add_argument("image", help="Path to an image file.")
    parser.add_argument(
        "--output-mask",
        help="Optional path to write the predicted RGB mask PNG.",
    )
    args = parser.parse_args(argv)

    predictor = SegmentationPredictor(Path(args.checkpoint))
    with Image.open(args.image) as image:
        mask = predictor.predict(image.convert("RGB"))

    class_indices = mask.squeeze().cpu().numpy()
    output_mask_path: str | None = None
    if args.output_mask:
        resolved = Path(args.output_mask).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        rgb_mask = _mask_to_rgb(class_indices, predictor.class_idx_to_rgb)
        Image.fromarray(rgb_mask).save(resolved)
        output_mask_path = str(resolved)

    print(
        to_json(
            segmentation_response(
                mask_shape=tuple(int(dim) for dim in mask.shape),
                image_path=args.image,
                output_mask_path=output_mask_path,
                class_pixel_counts=_count_classes(
                    class_indices,
                    predictor.class_idx_to_name,
                ),
            )
        ),
        end="",
    )


if __name__ == "__main__":
    main()
