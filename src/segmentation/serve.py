"""Serve segmentation checkpoints locally as JSON CLI predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from src.common.serving import segmentation_response, to_json
from src.segmentation.predictor import SegmentationPredictor


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run local segmentation inference.")
    parser.add_argument("checkpoint", help="Path to a segmentation checkpoint.")
    parser.add_argument("image", help="Path to an image file.")
    args = parser.parse_args(argv)

    predictor = SegmentationPredictor(Path(args.checkpoint))
    with Image.open(args.image) as image:
        mask = predictor.predict(image.convert("RGB"))

    print(
        to_json(
            segmentation_response(
                mask_shape=tuple(int(dim) for dim in mask.shape),
                image_path=args.image,
            )
        ),
        end="",
    )


if __name__ == "__main__":
    main()
