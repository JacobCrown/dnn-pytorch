"""Trainer for classification models."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch import autocast
from torch.utils.data import DataLoader

from src.classification.config import ClassificationConfig
from src.classification.metrics import ClassificationMetrics
from src.classification.model_manager import ClassificationModelManager
from src.classification.visualizer import ClassificationVisualizer
from src.common.experiment_paths import classification_experiment_root
from src.common.logging import artifact_rel_to_experiment, gpu_epoch_log_suffix
from src.common.progress_tools import iterate_with_progress
from src.common.tracking import Tracker
from src.common.utils import DEVICE
from src.core.trainer import BaseTrainer
from src.classification.evaluate import evaluate_classification_model

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
        val_loader: DataLoader | None,
        cfg: ClassificationConfig,
        test_loader: DataLoader | None = None,
        tracker: Tracker | None = None,
    ):
        self.cfg: ClassificationConfig = cfg
        self.tracker = tracker
        # Task specific attributes
        self.class_names: list[str] = train_loader.dataset.get_class_names()
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
            self.test_loader
            if self.test_loader is not None
            else self.val_loader
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
        self.model_manager.prepare_model_for_training()
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
                scale_before_step = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scaler.get_scale() >= scale_before_step:
                    self._after_optimizer_step()
            else:
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, self.grad_clip)
                self.optimizer.step()
                self._after_optimizer_step()

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
        best_ema_val_loss = float("inf")

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

            ema_val_loss: float | None = None
            ema_val_acc: float | None = None
            if self.ema is not None:
                ema_val_loss, ema_val_acc = self._validate_with_ema()

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
            if ema_val_loss is not None and ema_val_acc is not None:
                logger.info(
                    "Epoch %d/%d | ema val loss=%.4f acc=%.2f%%",
                    epoch + 1,
                    self.num_epochs,
                    ema_val_loss,
                    ema_val_acc,
                )

            if self.tracker is not None:
                epoch_metrics = {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                }
                if ema_val_loss is not None and ema_val_acc is not None:
                    epoch_metrics["ema_val_loss"] = ema_val_loss
                    epoch_metrics["ema_val_accuracy"] = ema_val_acc
                self.tracker.log_metrics(epoch_metrics, step=epoch + 1)

            # Always save last checkpoint
            self._save_classification_checkpoint(
                "last.pth",
                variant="online",
                val_loss=val_loss,
            )
            if self.ema is not None and ema_val_loss is not None:
                self._save_ema_classification_checkpoint(
                    "last_ema.pth",
                    val_loss=ema_val_loss,
                )

            # Save best model when validation metric improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_classification_checkpoint(
                    "best.pth",
                    variant="online",
                    val_loss=val_loss,
                )
                patience_counter = 0
            else:
                patience_counter += 1
            if (
                self.ema is not None
                and ema_val_loss is not None
                and ema_val_loss < best_ema_val_loss
            ):
                best_ema_val_loss = ema_val_loss
                self._save_ema_classification_checkpoint(
                    "best_ema.pth",
                    val_loss=ema_val_loss,
                )

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

    def _validate_with_ema(self) -> tuple[float, float]:
        if self.ema is None:
            raise RuntimeError("EMA validation requested but EMA is disabled")
        with self.ema.average_parameters(self.model):
            return self._validate()

    def _evaluate(self, test_loader: DataLoader) -> None:
        """Evaluate the model on test data."""
        summary = evaluate_classification_model(
            model=self.model,
            data_loader=test_loader,
            criterion=self.criterion,
            use_amp=self.use_amp,
            class_names=self.class_names,
            split_name="Test",
        )
        self.metrics.losses.test.append(summary.loss)
        self.metrics.accuracies.test.append(summary.accuracy)

    def _save_predictor_checkpoint(self, filename: str = "predictor.pth") -> None:
        candidates = [self.checkpoints_dir / "best.pth"]
        if self.ema is not None:
            candidates.append(self.checkpoints_dir / "best_ema.pth")
        existing_candidates = [path for path in candidates if path.exists()]
        if not existing_candidates:
            logger.warning(
                "Skipping predictor checkpoint export because no best checkpoint exists"
            )
            return

        source_path, checkpoint = self._select_predictor_source_checkpoint(
            existing_candidates
        )
        predictor_checkpoint = build_predictor_checkpoint_payload(checkpoint)
        predictor_checkpoint_path = self.checkpoints_dir / filename
        torch.save(predictor_checkpoint, predictor_checkpoint_path)
        logger.info(
            "Saved predictor %s from %s (%s, val_loss=%s; %.2f MB)",
            artifact_rel_to_experiment(predictor_checkpoint_path, self.root_dir),
            artifact_rel_to_experiment(source_path, self.root_dir),
            checkpoint.get("checkpoint_variant", "unknown"),
            self._format_selection_metric(checkpoint),
            predictor_checkpoint_path.stat().st_size / (1024 * 1024),
        )

    def _save_classification_checkpoint(
        self,
        filename: str,
        *,
        variant: str,
        val_loss: float,
    ) -> None:
        self._save_checkpoint(
            filename,
            extra_fields=self._checkpoint_selection_fields(
                variant=variant,
                val_loss=val_loss,
            ),
        )

    def _save_ema_classification_checkpoint(
        self,
        filename: str,
        *,
        val_loss: float,
    ) -> None:
        if self.ema is None:
            raise RuntimeError("EMA checkpoint requested but EMA is disabled")
        self._save_checkpoint(
            filename,
            model_state_dict=self.ema.model_state_dict(self.model),
            include_training_model_state_dict=True,
            extra_fields=self._checkpoint_selection_fields(
                variant="ema",
                val_loss=val_loss,
            ),
        )

    @staticmethod
    def _checkpoint_selection_fields(*, variant: str, val_loss: float) -> dict:
        return {
            "checkpoint_variant": variant,
            "selection_metric": {
                "name": "val_loss",
                "value": float(val_loss),
            },
        }

    def _select_predictor_source_checkpoint(
        self, candidates: list[Path]
    ) -> tuple[Path, dict]:
        loaded = [
            (
                path,
                torch.load(str(path), map_location=DEVICE, weights_only=False),
            )
            for path in candidates
        ]
        if len(loaded) == 1:
            return loaded[0]

        metric_values = [
            self._selection_metric_value(checkpoint) for _, checkpoint in loaded
        ]
        if any(value is None for value in metric_values):
            best_path = self.checkpoints_dir / "best.pth"
            for path, checkpoint in loaded:
                if path == best_path:
                    return path, checkpoint
            return loaded[0]

        return min(
            loaded,
            key=lambda item: self._selection_metric_value(item[1]) or float("inf"),
        )

    @staticmethod
    def _selection_metric_value(checkpoint: dict) -> float | None:
        metric = checkpoint.get("selection_metric")
        if not isinstance(metric, dict) or metric.get("name") != "val_loss":
            return None
        return float(metric["value"])

    @classmethod
    def _format_selection_metric(cls, checkpoint: dict) -> str:
        value = cls._selection_metric_value(checkpoint)
        if value is None:
            return "unknown"
        return f"{value:.4f}"

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
