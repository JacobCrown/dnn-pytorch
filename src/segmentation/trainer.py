"""Trainer for segmentation models."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.amp import autocast  # type: ignore[reportMissingImports]
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.segmentation.metrics import SegmentationMetrics
from src.common.enums import TaskType
from src.core.trainer import BaseTrainer
from src.segmentation.visualizer import SegmentationVisualizer
from src.segmentation.config import SegmentationConfig

logger = logging.getLogger(__name__)


class SegmentationTrainer(BaseTrainer):
    """Trainer class for segmentation models."""

    _TASK_TYPE: TaskType = TaskType.SEGMENTATION
    metrics: SegmentationMetrics = SegmentationMetrics()
    visualizer = SegmentationVisualizer

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: SegmentationConfig,
        num_classes: int,
        test_loader: DataLoader | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.num_classes = num_classes
        self.cfg: SegmentationConfig = cfg
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg.train,
            test_loader=test_loader,
            device=device,
        )

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch.

        Returns:
            A tuple of (average_loss, mIoU).
        """
        self.model.train()
        total_loss = 0.0
        total_iou = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for _, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device, enabled=self.use_amp):
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

    def validate(self) -> Tuple[float, float]:
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
                data, target = data.to(self.device), target.to(self.device)

                with autocast(device_type=self.device, enabled=self.use_amp):
                    output = self.model(data)
                    loss = self.criterion(output, target)

                total_loss += loss.item()

                # Calculate mIoU
                pred = output.argmax(dim=1)
                iou = self._calculate_iou(pred, target)
                total_iou += iou

        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches
        return avg_loss, avg_iou

    def train(
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

            print(f"Epoch {epoch + 1}/{self.num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}")

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
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        self.writer.close()

        # Evaluate on test set if provided
        if self.test_loader is not None:
            print("\n" + "=" * 50)
            print("FINAL TEST EVALUATION")
            print("=" * 50)
            test_metrics = self.evaluate(self.test_loader)
            print(f"Final Test mIoU: {test_metrics['test_miou']:.4f}")
            print("=" * 50)

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on test data.

        Args:
            test_loader: DataLoader for the test set.

        Returns:
            A dictionary of evaluation metrics (loss, mIoU).
        """
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        num_batches = len(test_loader)
        all_predictions: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        with torch.inference_mode():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)

                with autocast(device_type=self.device, enabled=self.use_amp):
                    output = self.model(data)
                    loss = self.criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                iou = self._calculate_iou(pred, target)
                total_iou += iou

                all_predictions.append(pred.cpu())
                all_targets.append(target.cpu())

        avg_loss = total_loss / num_batches
        avg_iou = total_iou / num_batches

        all_predictions_tensor = torch.cat(all_predictions).flatten()
        all_targets_tensor = torch.cat(all_targets).flatten()

        cm = confusion_matrix(
            all_targets_tensor.numpy(), all_predictions_tensor.numpy()
        )
        report = classification_report(
            all_targets_tensor.numpy(), all_predictions_tensor.numpy(), output_dict=True
        )

        metrics = {
            "test_loss": avg_loss,
            "test_miou": avg_iou,
            "confusion_matrix": cm,
            "classification_report": report,
        }

        print("Test Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  mIoU: {avg_iou:.4f}")

        return metrics

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
        confusion_vector = target * self.num_classes + pred
        counts = torch.bincount(confusion_vector, minlength=self.num_classes**2)
        confusion_matrix = counts.reshape(self.num_classes, self.num_classes)

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

    def predict_labels(self, data_loader: DataLoader) -> Tuple[List[int], List[int]]:
        """Return true and predicted labels for a loader (CPU lists)."""
        self.model.eval()
        all_predictions: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        with torch.inference_mode():
            for data, target in tqdm(data_loader, desc="Predict"):
                data, target = data.to(self.device), target.to(self.device)
                with autocast(device_type=self.device, enabled=self.use_amp):
                    output = self.model(data)
                    pred = output.argmax(dim=1)
                all_predictions.append(pred.cpu())
                all_targets.append(target.cpu())

        y_true = torch.cat(all_targets).flatten().tolist()
        y_pred = torch.cat(all_predictions).flatten().tolist()

        return y_true, y_pred

    def visualize(self) -> None:
        """Visualize all aspects of training.

        This method creates visualizations for:
        - Example images (with and without augmentations)
        - Data distribution (pixel counts per class)
        - Model predictions
        - Training metrics (loss and mIoU curves)
        """
        vis_save_dir = self.root_dir / "visualizations"

        logger.info("\n" + "=" * 50)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 50)

        # Visualize example images
        logger.info("\n1. Visualizing example images...")
        self.visualizer.visualize_example_images(
            data_loader=self.train_loader,
            save_dir=vis_save_dir,
            num_samples=4,
        )

        # Visualize data distribution
        logger.info("\n2. Visualizing data distribution...")
        self.visualizer.visualize_data_distribution(
            data_loader=self.train_loader,
            save_dir=vis_save_dir,
        )

        # Visualize predictions (use test set if available, otherwise validation set)
        logger.info("\n3. Visualizing predictions...")
        prediction_loader = (
            self.test_loader if self.test_loader is not None else self.val_loader
        )
        self.visualizer.visualize_predictions(
            data_loader=prediction_loader,
            model=self.model,
            save_dir=vis_save_dir,
            device=self.device,
            use_amp=self.use_amp,
            num_samples=4,
        )

        # Visualize metrics
        logger.info("\n4. Visualizing training metrics...")
        self.visualizer.visualize_metrics(
            metrics=self.metrics,
            save_dir=vis_save_dir,
        )

        logger.info("\n" + "=" * 50)
        logger.info(f"All visualizations saved to: {vis_save_dir}")
        logger.info("=" * 50)

    def get_root_dir_path(self) -> Path:
        """Get the path of the root directory."""
        arch = self.cfg.model.architecture
        model = self.cfg.model.params["encoder_name"]
        height = self.cfg.model.height
        width = self.cfg.model.width
        epochs = self.cfg.train.epochs
        return Path(self.cfg.train.root_dir) / self._TASK_TYPE.value / f"{arch}_{model}_{height}x{width}_{epochs}epochs"