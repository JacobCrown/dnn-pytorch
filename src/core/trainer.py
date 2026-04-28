import logging
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler  # type: ignore[reportMissingImports]
from torch.utils.data import DataLoader

from src.common.config import BaseTrainingConfig
from src.common.utils import DEVICE
from src.core.ema import ModelEMA
from src.core.metrics import BaseMetrics
from src.core.model_manager import BaseModelManager
from src.core.visualizer import BaseVisualizer

logger = logging.getLogger(__name__)


class BaseTrainer[
    VisualizerType: BaseVisualizer,
    MetricsType: BaseMetrics,
    ModelManagerType: BaseModelManager,
](ABC):
    """Base trainer class for all tasks."""

    metrics_cls: type[MetricsType]
    visualizer_cls: type[VisualizerType]

    def __init__(
        self,
        model_manager: ModelManagerType,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        cfg: BaseTrainingConfig,
        test_loader: DataLoader | None = None,
    ):
        """Initialize the trainer.

        Args:
            model_manager: Base model manager to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            train_cfg: Training configuration
            test_loader: Test data loader (optional)
        """

        self.model_manager: ModelManagerType = model_manager
        self.model: nn.Module = self.model_manager.model.to(DEVICE)
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader | None = val_loader
        self.test_loader: DataLoader | None = test_loader
        self.train_cfg: BaseTrainingConfig = cfg
        self.root_dir: Path = self._get_root_dir_path()
        self.checkpoints_dir: Path = self.root_dir / self.train_cfg.checkpoints_dir
        self.log_dir: Path = self.root_dir / self.train_cfg.logs_dir
        self.visualize_dir: Path = self.root_dir / self.train_cfg.visualize_dir

        # Create directories
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.visualize_dir.mkdir(parents=True, exist_ok=True)

        # Setup loss, optimizer, and scheduler
        self.criterion: nn.Module = criterion
        self.optimizer: optim.Optimizer = optimizer
        self.scheduler: optim.lr_scheduler._LRScheduler = scheduler

        # Training configuration
        self.resume_from_checkpoint_path: Path | None = (
            Path(self.train_cfg.resume_from_checkpoint)
            if self.train_cfg.resume_from_checkpoint
            else None
        )
        self.early_stopping_patience: int | None = (
            self.train_cfg.early_stopping_patience
        )

        # Mixed precision
        self.use_amp: bool = self.train_cfg.use_amp
        self.grad_clip: float | None = self.train_cfg.gradient_clip
        self.scaler: GradScaler = GradScaler(device=DEVICE, enabled=self.use_amp)

        # Training state
        self.current_epoch: int = 0
        self.num_epochs: int = self.train_cfg.epochs
        self.ema: ModelEMA | None = (
            ModelEMA(
                self.model,
                decay=self.train_cfg.ema.decay,
                update_after_step=self.train_cfg.ema.update_after_step,
                update_every=self.train_cfg.ema.update_every,
            )
            if self.train_cfg.ema.enabled
            else None
        )

        # Setup visualizer
        self.visualizer: VisualizerType = self.visualizer_cls(
            save_dir=self.visualize_dir
        )

        # Visualization flags
        self.visualize_pre_train: bool = self.train_cfg.visualize_pre_train
        self.visualize_post_train: bool = self.train_cfg.visualize_post_train

        # Setup metrics
        self.metrics: MetricsType = self.metrics_cls()

        # Try to resume from checkpoint if specified
        self._try_resume_from_checkpoint()

    def _try_resume_from_checkpoint(self) -> None:
        """Resume training from the latest checkpoint."""
        if self.resume_from_checkpoint_path is None:
            return
        try:
            self._load_checkpoint(self.resume_from_checkpoint_path)
        except Exception as e:
            raise ValueError(f"Error resuming from checkpoint: {e}") from e
        logger.info(f"Resumed from epoch {self.current_epoch + 1}")

    def _load_checkpoint(self, filepath: Path) -> None:
        """Load model checkpoint.

        Args:
            filepath: Path to the checkpoint file
        """
        try:
            checkpoint = torch.load(
                str(filepath), map_location=DEVICE, weights_only=False
            )
        except Exception as e:
            raise ValueError(f"Error loading checkpoint {filepath}: {e}") from e

        self._load_checkpoint_common(checkpoint)
        self._load_checkpoint_task_specific(checkpoint)

    def _load_checkpoint_common(self, checkpoint: dict) -> None:
        if "model_config" in checkpoint:
            self.model_manager.check_model_config(checkpoint["model_config"])
            self.model_manager.load_preprocessing(
                checkpoint["model_config"]["preprocessing"]
            )
        else:
            raise ValueError("Model config not found in checkpoint")

        model_state_key = (
            "training_model_state_dict"
            if "training_model_state_dict" in checkpoint
            else "model_state_dict"
        )
        if model_state_key in checkpoint:
            self.model_manager.load_model_state_dict(checkpoint[model_state_key])
        else:
            raise ValueError("Model state dict not found in checkpoint")

        if self.ema is not None:
            if "ema_state_dict" in checkpoint:
                self.ema.load_state_dict(checkpoint["ema_state_dict"])
            else:
                logger.warning(
                    "EMA is enabled, but checkpoint has no ema_state_dict. "
                    "Initializing EMA from loaded model weights."
                )

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            raise ValueError("Optimizer state dict not found in checkpoint")

        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            raise ValueError("Scheduler state dict not found in checkpoint")

        if "epoch" in checkpoint:
            self.current_epoch = checkpoint["epoch"]
        else:
            raise ValueError("Epoch information not found in checkpoint")

        if "metrics_data" in checkpoint:
            self.metrics = self.metrics_cls.from_dict(checkpoint["metrics_data"])
        else:
            raise ValueError("Metrics data not found in checkpoint")

    def train(self):
        """Train the model."""
        if self.visualize_pre_train:
            self._visualize_pre_train()
        self._train()
        if self.visualize_post_train:
            self._visualize_post_train()

    def _after_optimizer_step(self) -> None:
        """Update optional model EMA after a successful optimizer step."""
        if self.ema is not None:
            self.ema.update(self.model)

    def _save_checkpoint(
        self,
        filename: str,
        *,
        model_state_dict: dict | None = None,
        include_training_model_state_dict: bool = False,
        extra_fields: dict | None = None,
    ) -> None:
        """Save model checkpoint.

        Args:
            filename: Name of the checkpoint file
        """
        checkpoint = self._create_checkpoint(
            model_state_dict=model_state_dict,
            include_training_model_state_dict=include_training_model_state_dict,
            extra_fields=extra_fields,
        )
        filepath = self.checkpoints_dir / filename
        torch.save(checkpoint, filepath)

    def _create_checkpoint(
        self,
        *,
        model_state_dict: dict | None = None,
        include_training_model_state_dict: bool = False,
        extra_fields: dict | None = None,
    ) -> dict:
        checkpoint_dict = self._create_checkpoint_common(
            model_state_dict=model_state_dict,
            include_training_model_state_dict=include_training_model_state_dict,
        )
        checkpoint_dict.update(self._create_checkpoint_task_specific())
        if extra_fields:
            checkpoint_dict.update(extra_fields)
        return checkpoint_dict

    def _create_checkpoint_common(
        self,
        *,
        model_state_dict: dict | None = None,
        include_training_model_state_dict: bool = False,
    ) -> dict:
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": model_state_dict
            if model_state_dict is not None
            else self.model_manager.get_model_state_dict(),
            "model_config": self.model_manager.model_cfg.to_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics_data": self.metrics.to_dict(),
        }
        if include_training_model_state_dict:
            checkpoint["training_model_state_dict"] = (
                self.model_manager.get_model_state_dict()
            )
        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()
        return checkpoint

    # ================================ Abstract Methods ================================

    @abstractmethod
    def _get_root_dir_path(self) -> Path:
        pass

    @abstractmethod
    def _visualize_pre_train(self):
        pass

    @abstractmethod
    def _visualize_post_train(self):
        pass

    @abstractmethod
    def _train_epoch(self):
        pass

    @abstractmethod
    def _train(self):
        pass

    @abstractmethod
    def _validate(self):
        pass

    @abstractmethod
    def _evaluate(self):
        pass

    @abstractmethod
    def _create_checkpoint_task_specific(self) -> dict:
        pass

    @abstractmethod
    def _load_checkpoint_task_specific(self, checkpoint: dict) -> None:
        pass
