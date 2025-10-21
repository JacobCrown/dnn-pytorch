from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler  # type: ignore[reportMissingImports]
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.common.config import BaseTrainingConfig
from src.common.enums import TaskType
from src.core.metrics import BaseMetrics


class BaseTrainer(ABC):
    """Trainer class for classification models."""

    _TASK_TYPE: TaskType = TaskType.UNDEFINED
    metrics: BaseMetrics = BaseMetrics()

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_cfg: BaseTrainingConfig,
        test_loader: Optional[DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the trainer.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            train_cfg: Training configuration
            test_loader: Test data loader (optional)
            device: Device to train on
        """

        self.model: nn.Module = model.to(device)
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.test_loader: Optional[DataLoader] = test_loader
        self.device: str = device
        self.root_dir: Path = Path(train_cfg.root_dir) / self._TASK_TYPE.value / self.model.__class__.__name__
        self.checkpoints_dir: Path = self.root_dir / train_cfg.checkpoints_dir
        self.log_dir: Path = self.root_dir / train_cfg.logs_dir
        self.visualize_dir: Path = self.root_dir / train_cfg.visualize_dir

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
        self.resume: Optional[str] = train_cfg.resume
        self.early_stopping_patience: Optional[int] = train_cfg.early_stopping_patience

        # Mixed precision
        self.use_amp: bool = train_cfg.use_amp
        self.grad_clip: Optional[float] = train_cfg.gradient_clip
        self.scaler: GradScaler = GradScaler(device=self.device, enabled=self.use_amp)

        # Setup logging
        self.writer: SummaryWriter = SummaryWriter(self.log_dir)

        # Training state
        self.current_epoch: int = 0
        self.num_epochs: int = train_cfg.epochs

        # Metrics tracking
        self.best_val_loss: float = float("inf")
        self.best_val_metric: float = 0.0  # Accuracy in case of classification

        # Resume from checkpoint if specified
        if self.resume:
            self._resume_from_checkpoint()

        self._check_all_vars_initialized()

    def _check_all_vars_initialized(self) -> None:
        """Check if all required variables are initialized."""
        if self._TASK_TYPE == TaskType.UNDEFINED:
            raise ValueError("Task type is not defined")
        if self.model is None:
            raise ValueError("Model is not defined")
        if self.train_loader is None:
            raise ValueError("Training data loader is not defined")
        if self.val_loader is None:
            raise ValueError("Validation data loader is not defined")
        if self.criterion is None:
            raise ValueError("Loss function is not defined")
        if self.optimizer is None:
            raise ValueError("Optimizer is not defined")
        if self.scheduler is None:
            raise ValueError("Scheduler is not defined")
        if self.resume is not None and isinstance(self.resume, str):
            if not Path(self.resume).exists():
                raise FileNotFoundError(f"Checkpoint file not found: {self.resume}")
        if self.grad_clip is not None:
            if not isinstance(self.grad_clip, (int, float)):
                raise ValueError("Gradient clipping must be a number")
        if self.scaler is None:
            raise ValueError("Scaler is not defined")
        if self.writer is None:
            raise ValueError("Writer is not defined")
        if self.current_epoch is None:
            raise ValueError("Current epoch is not defined")
        if self.num_epochs is None:
            raise ValueError("Number of epochs is not defined")
        if self.metrics is None:
            raise ValueError("Metrics are not defined")
        if self.best_val_loss is None:
            raise ValueError("Best validation loss is not defined")
        if self.best_val_metric is None:
            raise ValueError("Best validation accuracy is not defined")

    def _validate_checkpoint(self, filepath: Path) -> bool:
        """Validate if a checkpoint file is valid and can be loaded.

        Args:
            filepath: Path to the checkpoint file

        Returns:
            True if checkpoint is valid, False otherwise
        """
        try:
            # Try to load the checkpoint to see if it's valid
            checkpoint = torch.load(
                str(filepath), map_location=self.device, weights_only=False
            )

            # Check if essential keys are present
            required_keys = ["model_state_dict", "optimizer_state_dict", "epoch"]
            for key in required_keys:
                if key not in checkpoint:
                    print(f"Warning: Checkpoint missing required key: {key}")
                    return False

            return True
        except Exception as e:
            print(f"Checkpoint validation failed for {filepath}: {e}")
            return False

    def _resume_from_checkpoint(self) -> None:
        """Resume training from the latest checkpoint."""
        # Look for the latest checkpoint
        checkpoint_files = list(self.checkpoints_dir.glob("*.pth"))
        if not checkpoint_files:
            print("No checkpoint files found. Starting from scratch.")
            return

        # Find the latest checkpoint (prioritize 'last.pth', then most recent)
        latest_checkpoint = None
        if (self.checkpoints_dir / "last.pth").exists():
            latest_checkpoint = self.checkpoints_dir / "last.pth"
        else:
            # Find the most recent checkpoint file
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)

        if latest_checkpoint:
            # Validate the checkpoint before attempting to load
            if self._validate_checkpoint(latest_checkpoint):
                print(f"Resuming from checkpoint: {latest_checkpoint}")
                self.load_checkpoint(str(latest_checkpoint))
                print(f"Resumed from epoch {self.current_epoch + 1}")
            else:
                print(
                    f"Checkpoint {latest_checkpoint} is corrupted. Starting from scratch."
                )
                # Optionally remove corrupted checkpoint
                try:
                    latest_checkpoint.unlink()
                    print(f"Removed corrupted checkpoint: {latest_checkpoint}")
                except Exception as e:
                    print(f"Could not remove corrupted checkpoint: {e}")

    @abstractmethod
    def train_epoch(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    def save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            filename: Name of the checkpoint file
            is_best: Whether this is the best model so far
        """
        # Save only essential data, not the entire metrics object
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "best_val_loss": self.best_val_loss,
            "best_val_metric": self.best_val_metric,
            # Save metrics data as a dict instead of the object
            "metrics_data": (
                self.metrics.to_dict() if hasattr(self.metrics, "to_dict") else {}
            ),
        }

        filepath = self.checkpoints_dir / filename
        torch.save(checkpoint, filepath)

        if is_best:
            best_filepath = self.checkpoints_dir / "best.pth"
            torch.save(checkpoint, best_filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint.

        Args:
            filepath: Path to the checkpoint file
        """
        try:
            # Use weights_only=False explicitly since we trust our own checkpoints
            checkpoint = torch.load(
                filepath, map_location=self.device, weights_only=False
            )
        except Exception as e:
            print(f"Error loading checkpoint {filepath}: {e}")
            print("Starting training from scratch...")
            return

        # Load model state
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print("Warning: No model state dict found in checkpoint")

        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            print("Warning: No optimizer state dict found in checkpoint")

        # Load scheduler state
        if (
            self.scheduler
            and "scheduler_state_dict" in checkpoint
            and checkpoint["scheduler_state_dict"]
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load epoch
        if "epoch" in checkpoint:
            self.current_epoch = checkpoint["epoch"]
        else:
            print("Warning: No epoch information found in checkpoint")
            self.current_epoch = 0

        # Restore best values
        if "best_val_loss" in checkpoint:
            self.best_val_loss = checkpoint["best_val_loss"]
        if "best_val_metric" in checkpoint:
            self.best_val_metric = checkpoint["best_val_metric"]

        # Restore metrics data if available
        if "metrics_data" in checkpoint and hasattr(self.metrics, "from_dict"):
            try:
                self.metrics.from_dict(checkpoint["metrics_data"])
            except Exception as e:
                print(f"Warning: Could not restore metrics from checkpoint: {e}")
        elif "metrics" in checkpoint:
            # Fallback for old checkpoint format
            try:
                self.metrics = checkpoint["metrics"]
            except Exception as e:
                print(
                    f"Warning: Could not restore metrics from old checkpoint format: {e}"
                )
                print("Initializing fresh metrics...")

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def predict_labels(self):
        pass

    @abstractmethod
    def visualize(self):
        pass