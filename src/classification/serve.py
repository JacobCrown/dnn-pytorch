"""Serve classification checkpoints locally as JSON CLI predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from src.classification.predictor import ClassificationPredictor
from src.common.serving import classification_response, to_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run local classification inference.")
    parser.add_argument(
        "checkpoint", help="Path to a classification predictor checkpoint."
    )
    parser.add_argument("image", help="Path to an image file.")
    args = parser.parse_args(argv)

    predictor = ClassificationPredictor(Path(args.checkpoint))
    with Image.open(args.image) as image:
        label = predictor.predict_class(image.convert("RGB"))

    print(to_json(classification_response(label=label, image_path=args.image)), end="")


if __name__ == "__main__":
    main()
