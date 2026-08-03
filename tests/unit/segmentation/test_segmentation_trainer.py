from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.segmentation.config import SegmentationConfig
from src.segmentation.trainer import SegmentationTrainer


class TinySegmentationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TinySegmentationModelManager:
    def __init__(self) -> None:
        self.model = TinySegmentationModel()


def _make_trainer(tmp_path: Path) -> SegmentationTrainer:
    manager = TinySegmentationModelManager()
    dataset = TensorDataset(
        torch.randn(2, 3, 4, 4),
        torch.zeros(2, 4, 4, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=2)
    optimizer = torch.optim.SGD(manager.model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    config = SegmentationConfig()
    config.model.classes = 2
    config.train.root_dir = str(tmp_path)
    config.train.use_amp = False
    config.train.ema.enabled = True
    config.train.ema.decay = 0.5

    return SegmentationTrainer(
        model_manager=manager,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        val_loader=loader,
        cfg=config,
    )


def test_train_epoch_updates_ema_after_optimizer_step(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path)

    trainer._train_epoch()

    assert trainer.ema is not None
    assert trainer.ema.num_updates == 1
