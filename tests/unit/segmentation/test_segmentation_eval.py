import logging

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.segmentation.trainer import evaluate_segmentation_model


class TwoChannelModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, -x], dim=1)


def test_evaluate_segmentation_model_logs_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    data_loader = DataLoader(
        TensorDataset(
            torch.ones(2, 1, 2, 2),
            torch.zeros(2, 2, 2, dtype=torch.long),
        ),
        batch_size=1,
        shuffle=False,
    )
    model = TwoChannelModel()
    criterion = nn.CrossEntropyLoss()

    caplog.set_level(logging.INFO)
    summary = evaluate_segmentation_model(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        use_amp=False,
        num_classes=2,
        split_name="Test",
    )

    assert summary.loss >= 0.0
    assert 0.0 <= summary.iou <= 1.0
    assert "Test | loss=" in caplog.text
