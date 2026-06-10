from pathlib import Path

import torch
import torch.nn as nn

from src.common.model_export import export_model_to_onnx, verify_onnx_model


class TinyExportModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def test_export_model_to_onnx_writes_file_and_verifies_shape(tmp_path: Path) -> None:
    model = TinyExportModel()
    export_path = tmp_path / "tiny.onnx"

    result = export_model_to_onnx(
        model=model,
        output_path=export_path,
        input_shape=(1, 3, 4, 4),
    )

    assert result.output_path == export_path
    assert export_path.exists()
    assert result.input_shape == (1, 3, 4, 4)
    assert result.output_shape == (1, 2, 4, 4)


def test_verify_onnx_model_returns_runtime_output_shape(tmp_path: Path) -> None:
    model = TinyExportModel()
    export_path = tmp_path / "tiny.onnx"
    export_model_to_onnx(
        model=model,
        output_path=export_path,
        input_shape=(1, 3, 4, 4),
        verify=False,
    )

    output_shape = verify_onnx_model(export_path, input_shape=(1, 3, 4, 4))

    assert output_shape == (1, 2, 4, 4)
