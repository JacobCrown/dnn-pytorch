"""Trainer for classification models."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.amp import autocast  # type: ignore[reportMissingImports]
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classification.metrics import ClassificationMetrics
from src.classification.visualizer import ClassificationVisualizer
from src.common.config import BaseTrainingConfig
from src.common.enums import TaskType
from src.core.trainer import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    """Trainer class for classification models."""

    _TASK_TYPE: TaskType = TaskType.CLASSIFICATION
    metrics: ClassificationMetrics = ClassificationMetrics()
    visualizer = ClassificationVisualizer

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
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            train_cfg=train_cfg,
            test_loader=test_loader,
            device=device,
        )

    def train_epoch(self) -> Tuple[float, float]:
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
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device, enabled=self.use_amp):
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

    def validate(self) -> Tuple[float, float]:
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
                data, target = data.to(self.device), target.to(self.device)

                # Handle mixup/cutmix labels
                if isinstance(target, tuple):
                    labels, _, _ = target
                    labels = labels.to(self.device)
                    with autocast(device_type=self.device, enabled=self.use_amp):
                        output = self.model(data)
                        loss = self.criterion(output, labels)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(labels.view_as(pred)).sum().item()
                    total += labels.size(0)
                else:
                    with autocast(device_type=self.device, enabled=self.use_amp):
                        output = self.model(data)
                        loss = self.criterion(output, target)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)

                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(
        self,
    ) -> None:
        """Train the model."""
        patience_counter = 0

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch

            # Training
            train_loss, train_acc = self.train_epoch()
            self.metrics.losses["train"].append(train_loss)
            self.metrics.accuracies["train"].append(train_acc)

            # Validation
            val_loss, val_acc = self.validate()
            self.metrics.losses["val"].append(val_loss)
            self.metrics.accuracies["val"].append(val_acc)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Logging
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/Val", val_acc, epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

            print(f"Epoch {epoch + 1}/{self.num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Always save last checkpoint
            self.save_checkpoint("last.pth")

            # Save best model when validation metric improves
            if val_acc > self.best_val_metric:
                self.best_val_metric = val_acc
                self.save_checkpoint("best.pth", is_best=True)
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
            print(f"Final Test Accuracy: {test_metrics['test_accuracy']:.2f}%")
            print("=" * 50)

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on test data.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []

        with torch.inference_mode():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)

                with autocast(device_type=self.device, enabled=self.use_amp):
                    output = self.model(data)
                    loss = self.criterion(output, target)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

                total_loss += loss.item()

                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())

        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(test_loader)

        cm = confusion_matrix(all_targets, all_predictions)
        report = classification_report(
            all_targets, all_predictions, output_dict=True, zero_division=0  # type: ignore[arg-type]
        )

        metrics = {
            "test_loss": avg_loss,
            "test_accuracy": accuracy,
            "correct_predictions": correct,
            "total_samples": total,
            "confusion_matrix": cm,
            "classification_report": report,
        }

        print("Test Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Correct: {correct}/{total}")

        return metrics

    def predict_labels(self, data_loader: DataLoader) -> Tuple[List[int], List[int]]:
        """Return true and predicted labels for a loader (CPU lists)."""
        self.model.eval()
        all_predictions: List[int] = []
        all_targets: List[int] = []
        with torch.inference_mode():
            for data, target in tqdm(data_loader, desc="Predict"):
                data = data.to(self.device)
                with autocast(device_type=self.device, enabled=self.use_amp):
                    output = self.model(data)
                    pred = output.argmax(dim=1)
                all_predictions.extend(pred.cpu().tolist())
                if isinstance(target, tuple):
                    target = target[0]
                all_targets.extend(target.cpu().tolist())
        return all_targets, all_predictions

    def visualize(self) -> None:
        """Visualize all aspects of training.
        
        This method creates visualizations for:
        - Example images (with and without augmentations)
        - Data distribution
        - Model predictions
        - Training metrics (loss, accuracy, confusion matrix)
        """
        vis_save_dir = self.root_dir / "visualizations"
        
        print("\n" + "=" * 50)
        print("GENERATING VISUALIZATIONS")
        print("=" * 50)
        
        # Visualize example images
        print("\n1. Visualizing example images...")
        self.visualizer.visualize_example_images(
            data_loader=self.train_loader,
            save_dir=vis_save_dir,
            num_samples_per_class=5,
        )
        
        # Visualize data distribution
        print("\n2. Visualizing data distribution...")
        self.visualizer.visualize_data_distribution(
            data_loader=self.train_loader,
            save_dir=vis_save_dir,
        )
        
        # Visualize predictions (use test set if available, otherwise validation set)
        print("\n3. Visualizing predictions...")
        prediction_loader = self.test_loader if self.test_loader is not None else self.val_loader
        self.visualizer.visualize_predictions(
            data_loader=prediction_loader,
            model=self.model,
            save_dir=vis_save_dir,
            device=self.device,
            use_amp=self.use_amp,
            num_samples_per_class=5,
        )
        
        # Visualize metrics
        print("\n4. Visualizing training metrics...")
        self.visualizer.visualize_metrics(
            metrics=self.metrics,
            save_dir=vis_save_dir,
            data_loader=self.val_loader,
            model=self.model,
            device=self.device,
            use_amp=self.use_amp,
        )
        
        print("\n" + "=" * 50)
        print(f"All visualizations saved to: {vis_save_dir}")
        print("=" * 50)