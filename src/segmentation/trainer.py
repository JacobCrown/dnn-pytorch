"""Trainer for segmentation models."""

import logging
import typing as t
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast  # type: ignore[reportMissingImports]
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multiclass_jaccard_index
from tqdm import tqdm

from src.common.experiment_paths import segmentation_experiment_root
from src.common.logging import log_gpu_memory
from src.common.utils import DEVICE
from src.core.trainer import BaseTrainer
from src.segmentation.config import SegmentationConfig
from src.segmentation.dataset import SegmentationDataset
from src.segmentation.metrics import SegmentationMetrics
from src.segmentation.model_manager import SegmentationModelManager
from src.segmentation.visualizer import SegmentationVisualizer

logger = logging.getLogger(__name__)


class SegmentationTrainer(
    BaseTrainer[SegmentationVisualizer, SegmentationMetrics, SegmentationModelManager]
):
    """Trainer class for segmentation models."""

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

    # ================================ Overridden Abstract Methods ================================

    def _get_root_dir_path(self) -> Path:
        """Get the path of the root directory."""
        return segmentation_experiment_root(
            train_root_dir=self.cfg.train.root_dir,
            data_root=self.cfg.data.root,
            architecture=self.cfg.model.architecture,
            encoder_name=self.cfg.model.encoder_name,
            height=self.cfg.model.height,
            width=self.cfg.model.width,
            epochs=self.cfg.train.epochs,
            batch_size=self.cfg.train.batch_size,
        )

    def _visualize_pre_train(self) -> None:
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
            num_samples=5,
        )

        # Visualize data distribution
        logger.info("\n2. Visualizing data distribution...")
        self.visualizer.visualize_data_distribution(
            data_loader=self.train_loader,
        )

    def _visualize_post_train(self) -> None:
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
            num_samples=5,
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

    def _train_epoch(self) -> tuple[float, float]:
        """Train for one epoch.

        Returns:
            Tuple of (average_loss, average_iou)
        """
        self.model.train()
        total_loss = torch.tensor(0.0, device=DEVICE)
        total_iou = torch.tensor(0.0, device=DEVICE)
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for _, (data, target) in enumerate(pbar):
            data, target = data.to(DEVICE), target.to(DEVICE)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=DEVICE, enabled=self.use_amp):
                output = self.model(data)
                loss = self.criterion(output, target)

            # Calculate mIoU
            pred = output.argmax(dim=1)
            miou = self._calculate_iou(pred, target)
            total_iou += miou

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

            total_loss += loss

        avg_loss = total_loss.item() / num_batches
        avg_iou = total_iou.item() / num_batches

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
            train_loss, train_miou = self._train_epoch()
            self.metrics.losses["train"].append(train_loss)
            self.metrics.ious["train"].append(train_miou)

            # Validation
            val_loss, val_miou = self._validate()
            self.metrics.losses["val"].append(val_loss)
            self.metrics.ious["val"].append(val_miou)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Logging
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}:")
            logger.info(f"  Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}")

            log_gpu_memory(logger)

            # Always save last checkpoint
            self._save_checkpoint("last.pth")

            # Save best model when validation metric improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint("best.pth")
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

        # Evaluate on test set if provided
        if self.test_loader is not None:
            logger.info("\n" + "=" * 50)
            logger.info("FINAL TEST EVALUATION")
            logger.info("=" * 50)
            self._evaluate(self.test_loader)
            logger.info("=" * 50)

    def _validate(self) -> tuple[float, float]:
        """Validate the model.

        Returns:
            A tuple of (average_loss, mIoU).
        """
        self.model.eval()
        total_loss = torch.tensor(0.0, device=DEVICE)
        total_iou = torch.tensor(0.0, device=DEVICE)
        num_batches = len(self.val_loader)

        with torch.inference_mode():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(DEVICE), target.to(DEVICE)

                with autocast(device_type=DEVICE, enabled=self.use_amp):
                    output = self.model(data)
                    loss = self.criterion(output, target)

                total_loss += loss

                pred = output.argmax(dim=1)
                total_iou += self._calculate_iou(pred, target)

        avg_loss = total_loss.item() / num_batches
        avg_iou = total_iou.item() / num_batches
        return avg_loss, avg_iou

    def _evaluate(self, test_loader: DataLoader) -> None:
        """Evaluate the model on test data."""
        self.model.eval()
        total_loss = torch.tensor(0.0, device=DEVICE)
        total_iou = torch.tensor(0.0, device=DEVICE)
        num_batches = len(test_loader)

        with torch.inference_mode():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(DEVICE), target.to(DEVICE)

                with autocast(device_type=DEVICE, enabled=self.use_amp):
                    output = self.model(data)
                    loss = self.criterion(output, target)

                pred = output.argmax(dim=1)
                total_iou += self._calculate_iou(pred, target)

                total_loss += loss

        avg_loss = total_loss.item() / num_batches
        avg_iou = total_iou.item() / num_batches

        logger.info("Test Results:")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  mIoU: {avg_iou:.4f}")

    def _create_checkpoint_task_specific(self) -> dict:
        dataset = t.cast(SegmentationDataset, self.train_loader.dataset)
        return {
            "class_names": dataset.get_class_names(),
            "class_idx_to_rgb": dataset.class_idx_to_rgb,
            "class_idx_to_name": dataset.class_idx_to_name,
        }

    def _load_checkpoint_task_specific(self, checkpoint: dict) -> None:
        dataset = t.cast(SegmentationDataset, self.train_loader.dataset)
        if "class_names" in checkpoint:
            checkpoint_class_names = checkpoint["class_names"]
            if checkpoint_class_names != dataset.get_class_names():
                raise ValueError(
                    f"Class names in checkpoint ({checkpoint_class_names}) do not match the class names in the trainer ({dataset.get_class_names()})"
                )
        else:
            raise ValueError("Class names not found in checkpoint")
        if "class_idx_to_rgb" in checkpoint:
            checkpoint_class_idx_to_rgb = checkpoint["class_idx_to_rgb"]
            if checkpoint_class_idx_to_rgb != dataset.class_idx_to_rgb:
                raise ValueError(
                    f"Class idx to rgb in checkpoint ({checkpoint_class_idx_to_rgb}) do not match the class idx to rgb in the trainer ({dataset.class_idx_to_rgb})"
                )
        if "class_idx_to_name" in checkpoint:
            checkpoint_class_idx_to_name = checkpoint["class_idx_to_name"]
            if checkpoint_class_idx_to_name != dataset.class_idx_to_name:
                raise ValueError(
                    f"Class idx to name in checkpoint ({checkpoint_class_idx_to_name}) do not match the class idx to name in the trainer ({dataset.class_idx_to_name})"
                )

    # ================================ Helper Methods ================================

    def _calculate_iou(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate mean IoU (Jaccard) for a batch using TorchMetrics.

        Args:
            pred: Predicted masks of shape (B, H, W).
            target: Ground truth masks of shape (B, H, W).

        Returns:
            The mean IoU score for the batch.
        """
        score = multiclass_jaccard_index(
            preds=pred,
            target=target,
            num_classes=self.cfg.model.classes,
            average="macro",
            ignore_index=255,
        )
        return score
