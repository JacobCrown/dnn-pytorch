"""Trainer for classification models."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast  # type: ignore[reportMissingImports]
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classification.config import ClassificationConfig
from src.classification.metrics import ClassificationMetrics
from src.classification.model_manager import ClassificationModelManager
from src.classification.visualizer import ClassificationVisualizer
from src.common.logging import log_gpu_memory
from src.common.utils import DEVICE
from src.core.trainer import BaseTrainer

logger = logging.getLogger(__name__)


class ClassificationTrainer(
    BaseTrainer[
        ClassificationVisualizer, ClassificationMetrics, ClassificationModelManager
    ]
):
    """Trainer class for classification models."""

    metrics_cls = ClassificationMetrics
    visualizer_cls = ClassificationVisualizer

    def __init__(
        self,
        model_manager: ClassificationModelManager,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: ClassificationConfig,
        test_loader: DataLoader | None = None,
    ):
        self.cfg: ClassificationConfig = cfg
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
        # Task specific attributes
        self.class_names: list[str] = train_loader.dataset.get_class_names()

    # ================================ Overridden Abstract Methods ================================

    def _get_root_dir_path(self) -> Path:
        """Get the path of the root directory."""
        model = self.cfg.model.name
        height = self.cfg.model.height
        width = self.cfg.model.width
        epochs = self.cfg.train.epochs
        bs = self.cfg.train.batch_size
        dataset_path = Path(self.cfg.data.root)
        dataset_name = dataset_path.name
        return (
            Path(self.cfg.train.root_dir)
            / "classification"
            / dataset_name
            / f"{model}_{height}x{width}_{epochs}epochs_{bs}bs"
        )

    def _visualize_pre_train(self) -> None:
        """Visualize all aspects of training.

        This method creates visualizations for:
        - Example images (with and without augmentations)
        - Data distribution
        """
        logger.info("=" * 50)
        logger.info("GENERATING PRE-TRAIN VISUALIZATIONS")
        logger.info("=" * 50)

        # Visualize example images
        logger.info("\n1. Visualizing example images...")
        self.visualizer.visualize_example_images(
            data_loader=self.train_loader,
            num_samples_per_class=5,
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
        - Training metrics (loss, accuracy, confusion matrix)
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
            num_samples_per_class=5,
        )

        # Visualize metrics
        logger.info("\n2. Visualizing training metrics...")
        self.visualizer.visualize_metrics(
            metrics=self.metrics,
            data_loader=prediction_loader,
            model=self.model,
            use_amp=self.use_amp,
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
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for _, (data, target) in enumerate(pbar):
            data, target = data.to(DEVICE), target.to(DEVICE)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=DEVICE, enabled=self.use_amp):
                output = self.model(data)
                loss = self.criterion(output, target)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

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

            # Update progress bar
            pbar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
            )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def _train(
        self,
    ) -> None:
        """Train the model."""
        patience_counter = 0
        best_val_loss = float("inf")

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch

            # Training
            train_loss, train_acc = self._train_epoch()
            self.metrics.losses["train"].append(train_loss)
            self.metrics.accuracies["train"].append(train_acc)

            # Validation
            val_loss, val_acc = self._validate()
            self.metrics.losses["val"].append(val_loss)
            self.metrics.accuracies["val"].append(val_acc)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Logging
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}:")
            logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

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
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.inference_mode():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(DEVICE), target.to(DEVICE)

                with autocast(device_type=DEVICE, enabled=self.use_amp):
                    output = self.model(data)
                    loss = self.criterion(output, target)

                total_loss += loss.item()

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def _evaluate(self, test_loader: DataLoader) -> None:
        """Evaluate the model on test data."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.inference_mode():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(DEVICE), target.to(DEVICE)

                with autocast(device_type=DEVICE, enabled=self.use_amp):
                    output = self.model(data)
                    loss = self.criterion(output, target)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

                total_loss += loss.item()

        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(test_loader)

        logger.info("Test Results:")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  Accuracy: {accuracy:.2f}%")
        logger.info(f"  Correct: {correct}/{total}")

    def _create_checkpoint_task_specific(self) -> dict:
        return {
            "class_names": self.class_names,
        }

    def _load_checkpoint_task_specific(self, checkpoint: dict) -> None:
        if "class_names" in checkpoint:
            checkpoint_class_names = checkpoint["class_names"]
            if checkpoint_class_names != self.class_names:
                raise ValueError(
                    f"Class names in checkpoint ({checkpoint_class_names}) do not match the class names in the trainer ({self.class_names})"
                )
        else:
            raise ValueError("Class names not found in checkpoint")
