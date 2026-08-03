import segmentation_models_pytorch as smp

from src.common.config import LossFunctionConfig
from src.common.losses import CombinedLoss, get_loss_function


def test_get_loss_function_builds_single_loss() -> None:
    loss = get_loss_function(
        LossFunctionConfig(
            name="DiceLoss", params={"mode": "multiclass", "smooth": 1e-6}
        )
    )

    assert isinstance(loss, smp.losses.DiceLoss)


def test_get_loss_function_builds_combined_loss() -> None:
    loss = get_loss_function(
        [
            LossFunctionConfig(
                name="FocalLoss",
                weight=0.6,
                params={"mode": "multiclass", "gamma": 2.0},
            ),
            LossFunctionConfig(
                name="DiceLoss",
                weight=0.4,
                params={"mode": "multiclass", "smooth": 1e-6},
            ),
        ]
    )

    assert isinstance(loss, CombinedLoss)
    assert len(loss.losses) == 2
