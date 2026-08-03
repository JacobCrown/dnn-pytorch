from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.classification.config import ClassificationModelConfig
from src.classification.export import main as export_classification_main
from src.classification.model_manager import ClassificationModelManager
from src.common.config import AugmentationConfig
from src.common.model_export import export_model_to_onnx


def test_exported_onnx_matches_pytorch_output(tmp_path: Path) -> None:
    config = ClassificationModelConfig(
        name="mobilenetv3_small_075",
        num_classes=2,
        pretrained=False,
        height=32,
        width=32,
        preprocessing=[
            AugmentationConfig(name="Resize", params={"height": 32, "width": 32}),
            AugmentationConfig(name="ToTensorV2", params={}),
        ],
    )
    manager = ClassificationModelManager(config)
    checkpoint_path = tmp_path / "model.pth"
    torch.save(
        {
            "model_config": config.to_dict(),
            "model_state_dict": manager.get_model_state_dict(),
            "class_names": ["a", "b"],
        },
        checkpoint_path,
    )

    onnx_path = tmp_path / "model.onnx"
    export_classification_main([str(checkpoint_path), str(onnx_path), "--no-verify"])

    loaded = ClassificationModelManager.from_checkpoint_path(checkpoint_path)
    loaded.model.cpu().eval()
    sample = np.random.randn(1, 3, 32, 32).astype(np.float32)
    with torch.inference_mode():
        pytorch_out = loaded.model(torch.from_numpy(sample)).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": sample})[0]

    assert onnx_out.shape == pytorch_out.shape
    np.testing.assert_allclose(onnx_out, pytorch_out, rtol=1e-4, atol=1e-4)


def test_export_model_to_onnx_cpu_round_trip(tmp_path: Path) -> None:
    config = ClassificationModelConfig(
        name="mobilenetv3_small_075",
        num_classes=2,
        pretrained=False,
        height=32,
        width=32,
    )
    manager = ClassificationModelManager(config)
    manager.model.cpu().eval()
    onnx_path = tmp_path / "direct.onnx"
    input_shape = (1, 3, 32, 32)

    export_model_to_onnx(
        model=manager.model,
        output_path=onnx_path,
        input_shape=input_shape,
    )

    sample = np.random.randn(*input_shape).astype(np.float32)
    with torch.inference_mode():
        expected = manager.model(torch.from_numpy(sample)).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": sample})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
