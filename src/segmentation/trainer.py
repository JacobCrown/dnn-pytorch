"""Trainer for segmentation models."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast  # type: ignore[reportMissingImports]
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.enums import TaskType
from src.common.utils import DEVICE
from src.core.trainer import BaseTrainer
from src.segmentation.config import SegmentationConfig
from src.segmentation.metrics import SegmentationMetrics
from src.segmentation.model_manager import SegmentationModelManager
from src.segmentation.visualizer import SegmentationVisualizer

logger = logging.getLogger(__name__)


class SegmentationTrainer(BaseTrainer[SegmentationVisualizer, SegmentationMetrics]):
    """Trainer class for segmentation models."""

    _TASK_TYPE: TaskType = TaskType.SEGMENTATION
    metrics_cls = SegmentationMetrics
    visualizer_cls = SegmentationVisualizer

    def __init__(
        self,
        model_manager: SegmentationModelManager,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: SegmentationConfig,
        test_loader: DataLoader | None = None,
    ):
        self.cfg: SegmentationConfig = cfg
        super().__init__(
            model_manager=model_manager,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg.train,
            test_loader=test_loader,
        )

    def train_epoch(self) -> tuple[float, float]:
        """Train for one epoch.

        Returns:
            Tuple of (average_loss, average_iou)
        """
        self.model.train()
        total_loss = 0.0
        total_iou = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for _, (data, target) in enumerate(pbar):
            data, target = data.to(DEVICE), target.to(DEVICE)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=DEVICE, enabled=self.use_amp):
                output = self.model(data)
                loss = self.criterion(output, target)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.optimizer.step()

            total_loss += loss.item()

            # Calculate mIoU
            pred = output.argmax(dim=1)
            miou = self._calculate_iou(pred, target)
            total_iou += miou

            # Update progress bar
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "mIoU": f"{miou:.4f}",
                }
            )

        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches

        return avg_loss, avg_iou

    def validate(self) -> tuple[float, float]:
        """Validate the model.

        Returns:
            A tuple of (average_loss, mIoU).
        """
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        num_batches = len(self.val_loader)

        with torch.inference_mode():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(DEVICE), target.to(DEVICE)

                with autocast(device_type=DEVICE, enabled=self.use_amp):
                    output = self.model(data)
                    loss = self.criterion(output, target)

                total_loss += loss.item()

                # Calculate mIoU
                pred = output.argmax(dim=1, keepdim=True)
                iou = self._calculate_iou(pred, target)
                total_iou += iou

        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches
        return avg_loss, avg_iou

    def _train(
        self,
    ) -> None:
        """Train the model for a specified number of epochs."""
        patience_counter = 0
        best_val_loss = float("inf")

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch

            # Training
            train_loss, train_miou = self.train_epoch()
            self.metrics.losses["train"].append(train_loss)
            self.metrics.ious["train"].append(train_miou)

            # Validation
            val_loss, val_miou = self.validate()
            self.metrics.losses["val"].append(val_loss)
            self.metrics.ious["val"].append(val_miou)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Logging
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("mIoU/Train", train_miou, epoch)
            self.writer.add_scalar("mIoU/Val", val_miou, epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}:")
            logger.info(f"  Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}")

            self._log_gpu_memory()

            # Always save last checkpoint
            self.save_checkpoint("last.pth")

            # Save best model when validation metric improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint("best.pth")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if (
                self.early_stopping_patience
                and patience_counter >= self.early_stopping_patience
            ):
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        self.writer.close()

        # Evaluate on test set if provided
        if self.test_loader is not None:
            logger.info("\n" + "=" * 50)
            logger.info("FINAL TEST EVALUATION")
            logger.info("=" * 50)
            self.evaluate(self.test_loader)
            logger.info("=" * 50)

    def evaluate(self, test_loader: DataLoader) -> None:
        """Evaluate the model on test data."""
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        num_batches = len(test_loader)

        with torch.inference_mode():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(DEVICE), target.to(DEVICE)

                with autocast(device_type=DEVICE, enabled=self.use_amp):
                    output = self.model(data)
                    loss = self.criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                iou = self._calculate_iou(pred, target)
                total_iou += iou

        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches

        logger.info("Test Results:")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  mIoU: {avg_iou:.4f}")

    def _calculate_iou(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate the Intersection over Union for a batch using a vectorized approach.

        Args:
            pred: Predicted masks of shape (B, H, W).
            target: Ground truth masks of shape (B, H, W).

        Returns:
            The mean IoU score for the batch.
        """
        pred = pred.view(-1)
        target = target.view(-1)

        # Calculate the confusion matrix
        num_classes = self.cfg.model.classes
        confusion_vector = target * num_classes + pred
        counts = torch.bincount(confusion_vector, minlength=num_classes**2)
        confusion_matrix = counts.reshape(num_classes, num_classes)

        # Calculate intersection and union
        intersection = torch.diag(confusion_matrix)
        union = (
            torch.sum(confusion_matrix, dim=1)
            + torch.sum(confusion_matrix, dim=0)
            - intersection
        )

        # Calculate IoU per class
        # division by zero will result in NaN, which is desired
        iou_per_class = intersection.float() / union.float()

        # mIoU is the mean of IoUs over classes that have a non-zero union.
        # This is equivalent to nanmean.
        valid_ious = iou_per_class[~torch.isnan(iou_per_class)]
        miou = valid_ious.mean().item() if valid_ious.numel() > 0 else 0.0

        return miou

    def visualize_pre_train(self) -> None:
        """Visualize all aspects of training.

        This method creates visualizations for:
        - Example images (with and without augmentations)
        - Data distribution (pixel counts per class)
        """
        logger.info("\n" + "=" * 50)
        logger.info("GENERATING PRE-TRAIN VISUALIZATIONS")
        logger.info("=" * 50)

        # Visualize example images
        logger.info("\n1. Visualizing example images...")
        self.visualizer.visualize_example_images(
            data_loader=self.train_loader,
            num_samples=4,
        )

        # Visualize data distribution
        logger.info("\n2. Visualizing data distribution...")
        self.visualizer.visualize_data_distribution(
            data_loader=self.train_loader,
        )

    def visualize_post_train(self) -> None:
        """Visualize all aspects of training.

        This method creates visualizations for:
        - Model predictions
        - Training metrics (loss and mIoU curves)
        """
        logger.info("=" * 50)
        logger.info("GENERATING POST-TRAIN VISUALIZATIONS")
        logger.info("=" * 50)
        # Visualize predictions (use test set if available, otherwise validation set)
        logger.info("\n1. Visualizing predictions...")
        prediction_loader = (
            self.test_loader if self.test_loader is not None else self.val_loader
        )
        self.visualizer.visualize_predictions(
            data_loader=prediction_loader,
            model=self.model,
            use_amp=self.use_amp,
            num_samples=4,
        )

        # Visualize metrics
        logger.info("\n2. Visualizing training metrics...")
        self.visualizer.visualize_metrics(
            metrics=self.metrics,
        )

        # Visualize per class metrics
        logger.info("\n3. Visualizing per class metrics...")
        self.visualizer.visualize_per_class_metrics(
            data_loader=prediction_loader,
            model=self.model,
            use_amp=self.use_amp,
        )

        logger.info("\n" + "=" * 50)
        logger.info(f"All visualizations saved to: {self.visualize_dir}")
        logger.info("=" * 50)

    def get_root_dir_path(self) -> Path:
        """Get the path of the root directory."""
        arch = self.cfg.model.architecture
        model = self.cfg.model.encoder_name
        height = self.cfg.model.height
        width = self.cfg.model.width
        epochs = self.cfg.train.epochs
        bs = self.cfg.train.batch_size
        dataset_path = Path(self.cfg.data.root)
        dataset_name = dataset_path.name
        return (
            Path(self.cfg.train.root_dir)
            / self._TASK_TYPE.value
            / dataset_name
            / f"{arch}_{model}_{height}x{width}_{epochs}epochs_{bs}bs"
        )
