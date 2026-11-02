"""Trainer for segmentation models."""

import logging
import typing as t
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast  # type: ignore[reportMissingImports]
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multiclass_jaccard_index

from src.common.experiment_paths import segmentation_experiment_root
from src.common.logging import artifact_rel_to_experiment, gpu_epoch_log_suffix
from src.common.progress_tools import iterate_with_progress
from src.common.tracker import MLFlowTracker
from src.common.utils import DEVICE
from src.core.trainer import BaseTrainer
from src.segmentation.config import SegmentationConfig
from src.segmentation.dataset import SegmentationDataset
from src.segmentation.metrics import SegmentationMetrics
from src.segmentation.model_manager import SegmentationModelManager
from src.segmentation.visualizer import SegmentationVisualizer

logger = logging.getLogger(__name__)

def build_predictor_checkpoint_payload(checkpoint: dict) -> dict:
    """Keep only the fields required by ClassificationPredictor."""
    required_keys = ("model_config", "model_state_dict", "class_idx_to_rgb", "class_idx_to_name")
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise ValueError(
            "Checkpoint is missing predictor fields: " + ", ".join(missing_keys)
        )
    return {key: checkpoint[key] for key in required_keys}


@dataclass
class SegmentationEvaluationSummary:
    loss: float
    iou: float
    model_input: torch.Tensor


def evaluate_segmentation_model(
    *,
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    use_amp: bool,
    num_classes: int,
    split_name: str = "Test",
) -> SegmentationEvaluationSummary:
    """Evaluate a segmentation model and log aggregate statistics."""
    model = model.to(DEVICE)
    model.eval()

    total_loss = torch.tensor(0.0, device=DEVICE)
    total_iou = torch.tensor(0.0, device=DEVICE)
    num_batches = len(data_loader)

    with torch.inference_mode():
        for data, target in iterate_with_progress(
            data_loader,
            description=f"Eval · {split_name}",
            transient=True,
        ):
            data, target = data.to(DEVICE), target.to(DEVICE)

            with autocast(device_type=DEVICE, enabled=use_amp):
                output = model(data)
                loss = criterion(output, target)

            pred = output.argmax(dim=1)
            total_loss += loss
            total_iou += multiclass_jaccard_index(
                preds=pred,
                target=target,
                num_classes=num_classes,
                average="macro",
                ignore_index=255,
            )

    avg_loss = total_loss.item() / num_batches
    avg_iou = total_iou.item() / num_batches
    logger.info(
        "%s | loss=%.4f iou=%.4f",
        split_name,
        avg_loss,
        avg_iou,
    )
    return SegmentationEvaluationSummary(loss=avg_loss, iou=avg_iou, model_input=data)


class SegmentationTrainer(
    BaseTrainer[SegmentationVisualizer, SegmentationMetrics, SegmentationModelManager, MLFlowTracker]
):
    """Trainer class for segmentation models."""

    metrics_cls = SegmentationMetrics
    visualizer_cls = SegmentationVisualizer
    tracker_cls = MLFlowTracker

    def __init__(
        self,
        model_manager: SegmentationModelManager,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        cfg: SegmentationConfig,
        run_directory: Path,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None = None,
    ):
        self.cfg: SegmentationConfig = cfg
        self.class_names = train_loader.dataset.get_class_names()
        super().__init__(
            model_manager=model_manager,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg.train,
            run_directory=run_directory,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )

    # ================================ Overridden Abstract Methods ================================

    def _get_root_dir_path(self) -> Path:
        """Get the path of the root directory."""
        return segmentation_experiment_root(config=self.cfg).resolve()

    def _visualize_pre_train(self) -> None:
        """Visualize all aspects of training.

        This method creates visualizations for:
        - Example images (with and without augmentations)
        - Data distribution (pixel counts per class)
        """
        logger.info("[viz] pre-train · example grids (with / without aug)")

        # Visualize example images
        self.visualizer.visualize_example_images(
            data_loader=self.train_loader,
            num_samples=5,
        )

        # Visualize data distribution
        logger.info("[viz] pre-train · class histogram")
        self.visualizer.visualize_data_distribution(
            data_loader=self.train_loader,
        )

    def _visualize_post_train(self) -> None:
        """Visualize all aspects of training.

        This method creates visualizations for:
        - Model predictions
        - Training metrics (loss and mIoU curves)
        """
        logger.info("[viz] post-train · prediction panel")
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
        logger.info("[viz] post-train · curves & confusion matrix")
        self.visualizer.visualize_metrics(
            metrics=self.metrics,
        )

        # Visualize per class metrics
        logger.info("[viz] post-train · per-class accuracy")
        self.visualizer.visualize_per_class_metrics(
            data_loader=prediction_loader,
            model=self.model,
            use_amp=self.use_amp,
        )

        logger.info(
            "[viz] outputs under %s",
            artifact_rel_to_experiment(self.visualize_dir, self.root_dir),
        )


    def _train_epoch(self) -> tuple[float, float]:
        """Train for one epoch.

        Returns:
            Tuple of (average_loss, average_iou)
        """
        self.model.train()
        total_loss = torch.tensor(0.0, device=DEVICE)
        total_iou = torch.tensor(0.0, device=DEVICE)
        num_batches = len(self.train_loader)
        trainable_parameters = self.model_manager.get_trainable_parameters()


        for data, target in iterate_with_progress(
            self.train_loader,
            description=f"Epoch {self.current_epoch + 1}/{self.num_epochs} · train",
            transient=True,
        ):
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
                        trainable_parameters, self.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters, self.grad_clip
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
            self.metrics.losses.train.append(train_loss)
            self.metrics.ious.train.append(train_miou)

            # Validation
            val_loss, val_miou = self._validate()
            self.metrics.losses.val.append(val_loss)
            self.metrics.ious.val.append(val_miou)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()


            gpu = gpu_epoch_log_suffix()
            logger.info(
                "Epoch %d/%d | train loss=%.4f iou=%.2f | val loss=%.4f iou=%.2f%s",
                epoch + 1,
                self.num_epochs,
                train_loss,
                train_miou,
                val_loss,
                val_miou,
                gpu,
            )

            self.tracker.log_metrics(self.metrics.get_latest_metrics(), step=epoch)

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

        self._save_predictor_checkpoint()

        # Evaluate on test set if provided
        if self.test_loader is not None:
            logger.info("Hold-out test set")
            self._evaluate(self.test_loader)

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
            for data, target in iterate_with_progress(
                self.val_loader,
                description="Validation",
                transient=True,
            ):
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
        best_model = self.model_manager.from_checkpoint_path(self.checkpoints_dir / "predictor.pth").model
        best_model.eval()
        summary = evaluate_segmentation_model(
            model=best_model,
            data_loader=test_loader,
            criterion=self.criterion,
            use_amp=self.use_amp,
            num_classes=len(self.class_names),
            split_name="Test",
        )
        self.metrics.losses.test.append(summary.loss)
        self.metrics.ious.test.append(summary.iou)
        self.tracker.log_model(pytorch_model=best_model, name="best model", input_example=summary.model_input)

    def _save_predictor_checkpoint(self, filename: str = "predictor.pth") -> None:
        best_checkpoint = self.checkpoints_dir / "best.pth"
        if not best_checkpoint.exists():
            logger.warning(
                "Skipping predictor checkpoint export because no best checkpoint exists"
            )
            return

        checkpoint = self._load_checkpoint_dict(best_checkpoint)
        predictor_checkpoint = build_predictor_checkpoint_payload(checkpoint)
        predictor_checkpoint_path = self.checkpoints_dir / filename
        torch.save(predictor_checkpoint, predictor_checkpoint_path)
        logger.info(
            "Saved predictor %s (%.2f MB)",
            artifact_rel_to_experiment(predictor_checkpoint_path, self.root_dir),
            predictor_checkpoint_path.stat().st_size / (1024 * 1024),
        )

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
