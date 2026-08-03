"""Export classification checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.classification.model_manager import ClassificationModelManager
from src.common.model_export import export_model_to_onnx


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a classification checkpoint.")
    parser.add_argument("checkpoint", help="Path to a classification .pth checkpoint.")
    parser.add_argument("output", help="Path to write the ONNX model.")
    parser.add_argument("--no-verify", action="store_true", help="Skip ONNX Runtime smoke test.")
    args = parser.parse_args(argv)

    manager = ClassificationModelManager.from_checkpoint_path(Path(args.checkpoint))
    cfg = manager.model_cfg
    result = export_model_to_onnx(
        model=manager.model,
        output_path=args.output,
        input_shape=(1, cfg.in_channels, cfg.height, cfg.width),
        verify=not args.no_verify,
    )
    print(
        f"Exported ONNX model to {result.output_path} "
        f"(input={result.input_shape}, output={result.output_shape})"
    )


if __name__ == "__main__":
    main()
