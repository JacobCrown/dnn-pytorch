import logging
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler  # type: ignore[reportMissingImports]
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.common.config import BaseTrainingConfig
from src.common.enums import TaskType
from src.common.utils import DEVICE
from src.core.metrics import BaseMetrics
from src.core.model_manager import BaseModelManager

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Trainer class for classification models."""

    _TASK_TYPE: TaskType = TaskType.UNDEFINED
    metrics: BaseMetrics = BaseMetrics()

    def __init__(
        self,
        model_manager: BaseModelManager,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
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

        self.model_manager: BaseModelManager = model_manager
        self.model: nn.Module = self.model_manager.model.to(DEVICE)
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.test_loader: DataLoader | None = test_loader
        self.train_cfg: BaseTrainingConfig = cfg
        self.root_dir: Path = self.get_root_dir_path()
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

        # Setup logging
        self.writer: SummaryWriter = SummaryWriter(self.log_dir)

        # Training state
        self.current_epoch: int = 0
        self.num_epochs: int = self.train_cfg.epochs

        # Resume from checkpoint if specified
        if self.resume_from_checkpoint_path:
            self._resume_from_checkpoint()

        self._check_all_vars_initialized()

    def _check_all_vars_initialized(self) -> None:
        """Check if all required variables are initialized."""
        if self._TASK_TYPE == TaskType.UNDEFINED:
            raise ValueError("Task type is not defined")
        if self.model_manager is None:
            raise ValueError("Model manager is not defined")
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
        if (
            self.resume_from_checkpoint_path is not None
            and isinstance(self.resume_from_checkpoint_path, Path)
            and not Path(self.resume_from_checkpoint_path).exists()
        ):
            raise FileNotFoundError(
                f"Checkpoint file not found: {self.resume_from_checkpoint_path}"
            )
        if self.grad_clip is not None and not isinstance(self.grad_clip, int | float):
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

    def _log_gpu_memory(self) -> None:
        """Log GPU memory usage if available."""
        if torch.cuda.is_available():
            max_mem_allocated = torch.cuda.max_memory_allocated() / (1024**2)
            self.writer.add_scalar(
                "Memory/Max_GPU_Memory_Allocated_MB",
                max_mem_allocated,
                self.current_epoch,
            )
            logger.info(f"  Max GPU Memory Allocated: {max_mem_allocated:.2f} MB")
            torch.cuda.reset_peak_memory_stats()

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
                str(filepath), map_location=DEVICE, weights_only=False
            )

            # Check if essential keys are present
            required_keys = ["model_state_dict", "optimizer_state_dict", "epoch"]
            for key in required_keys:
                if key not in checkpoint:
                    logger.warning(f"Checkpoint missing required key: {key}")
                    return False

            return True
        except Exception as e:
            logger.warning(f"Checkpoint validation failed for {filepath}: {e}")
            return False

    def _resume_from_checkpoint(self) -> None:
        """Resume training from the latest checkpoint."""
        if self.resume_from_checkpoint_path:
            # Validate the checkpoint before attempting to load
            if self._validate_checkpoint(self.resume_from_checkpoint_path):
                logger.info(
                    f"Resuming from checkpoint: {self.resume_from_checkpoint_path}"
                )
                self.load_checkpoint(self.resume_from_checkpoint_path)
                logger.info(f"Resumed from epoch {self.current_epoch + 1}")
            else:
                raise ValueError(
                    f"Checkpoint {self.resume_from_checkpoint_path} is corrupted. Starting from scratch."
                )

    @abstractmethod
    def get_root_dir_path(self) -> Path:
        """Get the path of the root directory. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def train_epoch(self):
        pass

    @abstractmethod
    def train_(self):
        pass

    def train(self):
        """Train the model."""
        self.visualize_pre_train()
        self.train_()
        self.visualize_post_train()

    @abstractmethod
    def validate(self):
        pass

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.

        Args:
            filename: Name of the checkpoint file
        """
        # Save only essential data, not the entire metrics object
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model_manager.get_model_state_dict(),
            "model_config": self.model_manager.model_cfg.to_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            # Save metrics data as a dict instead of the object
            "metrics_data": (
                self.metrics.to_dict() if hasattr(self.metrics, "to_dict") else {}
            ),
        }

        filepath = self.checkpoints_dir / filename
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: Path) -> None:
        """Load model checkpoint.

        Args:
            filepath: Path to the checkpoint file
        """
        try:
            # Use weights_only=False explicitly since we trust our own checkpoints
            checkpoint = torch.load(
                str(filepath), map_location=DEVICE, weights_only=False
            )
        except Exception as e:
            raise ValueError(f"Error loading checkpoint {filepath}: {e}") from e

        if "model_config" in checkpoint:
            self.model_manager.check_model_config(checkpoint["model_config"])
        else:
            raise ValueError("Model config not found in checkpoint")

        # Load model state
        if "model_state_dict" in checkpoint:
            self.model_manager.load_model_state_dict(checkpoint["model_state_dict"])
        else:
            raise ValueError("Model state dict not found in checkpoint")

        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            raise ValueError("Optimizer state dict not found in checkpoint")

        # Load scheduler state
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            raise ValueError("Scheduler state dict not found in checkpoint")

        # Load epoch
        if "epoch" in checkpoint:
            self.current_epoch = checkpoint["epoch"]
        else:
            raise ValueError("Epoch information not found in checkpoint")

        # Restore metrics data if available
        if "metrics_data" in checkpoint and hasattr(self.metrics, "from_dict"):
            self.metrics = self.metrics.from_dict(checkpoint["metrics_data"])
        else:
            raise ValueError("Metrics data not found in checkpoint")

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def predict_labels(self):
        pass

    @abstractmethod
    def visualize_pre_train(self):
        pass

    @abstractmethod
    def visualize_post_train(self):
        pass
