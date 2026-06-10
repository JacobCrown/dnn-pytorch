"""Model export helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


@dataclass(frozen=True)
class OnnxExportResult:
    output_path: Path
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


def verify_onnx_model(
    onnx_path: str | Path,
    *,
    input_shape: tuple[int, ...],
    input_name: str = "input",
) -> tuple[int, ...]:
    """Run a lightweight ONNX Runtime inference and return the output shape."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sample = np.random.randn(*input_shape).astype(np.float32)
    outputs = session.run(None, {input_name: sample})
    if not outputs:
        raise ValueError(f"ONNX model produced no outputs: {onnx_path}")
    return tuple(int(dim) for dim in outputs[0].shape)


def export_model_to_onnx(
    *,
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, ...],
    input_name: str = "input",
    output_name: str = "output",
    opset_version: int = 17,
    verify: bool = True,
) -> OnnxExportResult:
    """Export a PyTorch module to ONNX and optionally smoke-test it."""
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy_input = torch.randn(*input_shape)

    with torch.inference_mode(), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You are using the legacy TorchScript-based ONNX export.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="The feature will be removed.*",
            category=DeprecationWarning,
        )
        torch.onnx.export(
            model.cpu(),
            dummy_input,
            str(resolved_output_path),
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes={
                input_name: {0: "batch"},
                output_name: {0: "batch"},
            },
            opset_version=opset_version,
        )

    if verify:
        output_shape = verify_onnx_model(
            resolved_output_path,
            input_shape=input_shape,
            input_name=input_name,
        )
    else:
        with torch.inference_mode():
            output = model(dummy_input)
        output_shape = tuple(int(dim) for dim in output.shape)

    return OnnxExportResult(
        output_path=resolved_output_path,
        input_shape=input_shape,
        output_shape=output_shape,
    )
