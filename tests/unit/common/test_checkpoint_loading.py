from pathlib import Path

import torch

from src.classification.config import ClassificationModelConfig
from src.classification.model_manager import ClassificationModelManager
from src.common.config import AugmentationConfig
from src.segmentation.config import SegmentationModelConfig
from src.segmentation.model_manager import SegmentationModelManager


def test_classification_from_checkpoint_path_loads_trained_weights(
    tmp_path: Path,
) -> None:
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
    with torch.no_grad():
        manager.model.classifier.bias.fill_(1.0)

    checkpoint_path = tmp_path / "cls.pth"
    torch.save(
        {
            "model_config": config.to_dict(),
            "model_state_dict": manager.get_model_state_dict(),
            "class_names": ["a", "b"],
        },
        checkpoint_path,
    )

    random_manager = ClassificationModelManager(config)
    loaded_manager = ClassificationModelManager.from_checkpoint_path(checkpoint_path)

    loaded_bias = loaded_manager.get_model_state_dict()["classifier.bias"]
    random_bias = random_manager.get_model_state_dict()["classifier.bias"]
    assert torch.all(loaded_bias == 1.0)
    assert not torch.allclose(random_bias, loaded_bias)


def test_segmentation_from_checkpoint_path_loads_trained_weights(
    tmp_path: Path,
) -> None:
    config = SegmentationModelConfig(
        architecture="Unet",
        encoder_name="resnet18",
        encoder_weights=None,
        height=64,
        width=64,
        classes=2,
        preprocessing=[
            AugmentationConfig(name="Resize", params={"height": 64, "width": 64}),
            AugmentationConfig(name="ToTensorV2", params={"transpose_mask": True}),
        ],
    )
    manager = SegmentationModelManager(config)
    with torch.no_grad():
        manager.model.segmentation_head[0].weight.fill_(0.5)

    checkpoint_path = tmp_path / "seg.pth"
    torch.save(
        {
            "model_config": config.to_dict(),
            "model_state_dict": manager.get_model_state_dict(),
            "class_idx_to_name": {0: "background", 1: "road"},
            "class_idx_to_rgb": {0: [0, 0, 0], 1: [128, 64, 128]},
        },
        checkpoint_path,
    )

    random_manager = SegmentationModelManager(config)
    loaded_manager = SegmentationModelManager.from_checkpoint_path(checkpoint_path)

    weight_key = "segmentation_head.0.weight"
    loaded_weight = loaded_manager.get_model_state_dict()[weight_key]
    random_weight = random_manager.get_model_state_dict()[weight_key]
    assert torch.all(loaded_weight == 0.5)
    assert not torch.allclose(random_weight, loaded_weight)
