import logging

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.classification.trainer import (build_predictor_checkpoint_payload,
                                        evaluate_classification_model)


class SignBasedModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_feature = x[:, 0]
        return torch.stack((1.0 - first_feature, first_feature), dim=1)


def test_build_predictor_checkpoint_payload_keeps_only_predictor_fields() -> None:
    checkpoint = {
        "model_config": {"name": "resnet18"},
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "class_names": ["a", "b"],
        "optimizer_state_dict": {"lr": 0.1},
    }

    payload = build_predictor_checkpoint_payload(checkpoint)

    assert payload == {
        "model_config": {"name": "resnet18"},
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "class_names": ["a", "b"],
    }


def test_build_predictor_checkpoint_payload_requires_predictor_fields() -> None:
    with pytest.raises(ValueError, match="missing predictor fields"):
        build_predictor_checkpoint_payload({"model_config": {}, "class_names": []})


def test_evaluate_classification_model_logs_report(
    caplog: pytest.LogCaptureFixture,
) -> None:
    data_loader = DataLoader(
        TensorDataset(
            torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32),
            torch.tensor([1, 0, 1, 1], dtype=torch.long),
        ),
        batch_size=2,
        shuffle=False,
    )
    model = SignBasedModel()
    criterion = nn.CrossEntropyLoss()

    caplog.set_level(logging.DEBUG)
    summary = evaluate_classification_model(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        use_amp=False,
        class_names=["negative", "positive"],
        split_name="Validation",
    )

    assert summary.total == 4
    assert summary.correct == 3
    assert summary.accuracy == pytest.approx(75.0)
    assert summary.confusion_matrix.tolist() == [[1, 0], [1, 2]]
    assert "macro avg" in summary.classification_report
    assert "Validation | loss=" in caplog.text and "diagonal=3/4" in caplog.text
