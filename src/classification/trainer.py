"""Trainer for classification models."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch import autocast
from torch.utils.data import DataLoader

from src.classification.config import ClassificationConfig
from src.classification.evaluate import evaluate_classification_model
from src.classification.metrics import ClassificationMetrics
from src.classification.model_manager import ClassificationModelManager
from src.classification.visualizer import ClassificationVisualizer
from src.common.experiment_paths import classification_experiment_root
from src.common.logging import artifact_rel_to_experiment, gpu_epoch_log_suffix
from src.common.progress_tools import iterate_with_progress
from src.common.tracker import MLFlowTracker
from src.common.utils import DEVICE
from src.core.trainer import BaseTrainer

logger = logging.getLogger(__name__)


def build_predictor_checkpoint_payload(checkpoint: dict) -> dict:
    """Keep only the fields required by ClassificationPredictor."""
    required_keys = ("model_config", "model_state_dict", "class_names")
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise ValueError(
            "Checkpoint is missing predictor fields: " + ", ".join(missing_keys)
        )
    return {key: checkpoint[key] for key in required_keys}


class ClassificationTrainer(
    BaseTrainer[
        ClassificationVisualizer, ClassificationMetrics, ClassificationModelManager, MLFlowTracker
    ]
):
    """Trainer class for classification models."""

    metrics_cls = ClassificationMetrics
    visualizer_cls = ClassificationVisualizer
    tracker_cls = MLFlowTracker

    def __init__(
        self,
        model_manager: ClassificationModelManager,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        cfg: ClassificationConfig,
        run_directory: Path,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None = None,
    ):
        self.cfg: ClassificationConfig = cfg
        # Task specific attributes
        self.class_names: list[str] = train_loader.dataset.get_class_names()
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
        self._verify_train_batch_spatial_matches_model_config()

    def _verify_train_batch_spatial_matches_model_config(self) -> None:
        """Ensure one train batch tensor H×W matches ``model.height`` / ``model.width``.

        Misaligned ``Resize`` in preprocessing vs YAML run naming fields fail fast before epochs.
        """
        exp_h = self.cfg.model.height
        exp_w = self.cfg.model.width
        try:
            batch, _ = next(iter(self.train_loader))
        except StopIteration as exc:
            raise ValueError(
                "Cannot validate spatial dimensions: train DataLoader yielded no batches "
                "(empty dataset or batch size)."
            ) from exc
        if batch.dim() != 4:
            raise ValueError(
                f"Expected batched tensors (N,C,H,W), got shape {tuple(batch.shape)}"
            )
        _, _, h, w = batch.shape
        if (h, w) != (exp_h, exp_w):
            raise ValueError(
                f"After preprocessing, tensors are {h}x{w}, "
                f"but model.height={exp_h} and model.width={exp_w}. "
                "Align ``Resize`` in ``model.preprocessing`` with ``model.height`` / ``model.width``."
            )
        logger.info(
            "Train batch spatial size %dx%d matches model.height / model.width.",
            exp_h,
            exp_w,
        )

    # ================================ Overridden Abstract Methods ================================

    def _get_root_dir_path(self) -> Path:
        """Get the path of the root directory."""
        return classification_experiment_root(
            train_root_dir=self.cfg.train.root_dir,
            data_root=self.cfg.data.root,
            model_name=self.cfg.model.name,
            height=self.cfg.model.height,
            width=self.cfg.model.width,
            epochs=self.cfg.train.epochs,
            batch_size=self.cfg.train.batch_size,
        )

    def _visualize_pre_train(self) -> None:
        """Visualize all aspects of training.

        This method creates visualizations for:
        - Example images (with and without augmentations)
        - Data distribution
        """
        logger.info("[viz] pre-train · example grids (with / without aug)")

        # Visualize example images
        self.visualizer.visualize_example_images(
            data_loader=self.train_loader,
            num_samples_per_class=5,
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
        - Training metrics (loss, accuracy, confusion matrix)
        """
        logger.info("[viz] post-train · prediction panel")
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
        logger.info("[viz] post-train · curves & confusion matrix")
        self.visualizer.visualize_metrics(
            metrics=self.metrics,
            data_loader=prediction_loader,
            model=self.model,
            use_amp=self.use_amp,
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
            Tuple of (average_loss, accuracy)
        """
        # self.model_manager.prepare_model_for_training()
        self.model.train()
        total_loss = torch.tensor(0.0, device=DEVICE)
        correct = torch.tensor(0.0, device=DEVICE, dtype=torch.int64)
        total = torch.tensor(0.0, device=DEVICE, dtype=torch.int64)
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

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum()
            total += target.size(0)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, self.grad_clip)
                self.optimizer.step()

            total_loss += loss

        avg_loss = total_loss.item() / len(self.train_loader)
        accuracy = 100.0 * (correct / total).item()

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
            self.metrics.losses.train.append(train_loss)
            self.metrics.accuracies.train.append(train_acc)

            # Validation
            val_loss, val_acc = self._validate()
            self.metrics.losses.val.append(val_loss)
            self.metrics.accuracies.val.append(val_acc)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Logging
            gpu = gpu_epoch_log_suffix()
            logger.info(
                "Epoch %d/%d | train loss=%.4f acc=%.2f%% | val loss=%.4f acc=%.2f%%%s",
                epoch + 1,
                self.num_epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                gpu,
            )

            self.tracker.log_metrics(self.metrics.get_latest_metrics(), step=epoch)

            # Always save last checkpoint
            self._save_classification_checkpoint("last.pth")

            # Save best model when validation metric improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_classification_checkpoint("best.pth")
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
            Tuple of (average_loss, accuracy)
        """
        if self.val_loader is None:
            raise RuntimeError("Validation requested but val_loader is None")

        self.model.eval()
        total_loss = torch.tensor(0.0, device=DEVICE)
        correct = torch.tensor(0.0, device=DEVICE, dtype=torch.int64)
        total = torch.tensor(0.0, device=DEVICE, dtype=torch.int64)

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

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum()
                total += target.size(0)

        avg_loss = total_loss.item() / len(self.val_loader)
        accuracy = 100.0 * (correct / total).item()

        return avg_loss, accuracy

    def _evaluate(self, test_loader: DataLoader) -> None:
        """Evaluate the model on test data."""
        best_model = self.model_manager.from_checkpoint_path(self.checkpoints_dir / "predictor.pth").model
        best_model.eval()
        avg_loss, accuracy, _ = evaluate_classification_model(
            model=best_model,
            data_loader=test_loader,
            criterion=self.criterion,
            use_amp=self.use_amp,
            class_names=self.class_names,
            split_name="Test",
        )
        self.metrics.losses.test.append(avg_loss)
        self.metrics.accuracies.test.append(accuracy)
        # Skip loggin for now. Now focus on different things
        # self.tracker.log_model(pytorch_model=best_model, name="best model", input_example=model_input)

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

    def _save_classification_checkpoint(
        self,
        filename: str,
    ) -> None:
        self._save_checkpoint(filename)

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
