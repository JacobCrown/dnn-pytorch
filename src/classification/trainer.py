"""Trainer for classification models."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler  # type: ignore[reportMissingImports]
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from src.common.utils import plot_confusion_matrix, plot_training_history
from src.classification.config import TrainingConfig


class ClassificationTrainer:
    """Trainer class for classification models."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_cfg: TrainingConfig,
        root_dir: str,
        test_loader: Optional[DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir_name: str = "checkpoints",
        log_dir_name: str = "logs",
    ):
        """Initialize the trainer.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            save_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.model: nn.Module = model.to(device)
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.test_loader: Optional[DataLoader] = test_loader
        self.device: str = device
        self.root_dir: Path = Path(root_dir)
        self.save_dir: Path = Path(root_dir) / save_dir_name
        self.log_dir: Path = Path(root_dir) / log_dir_name

        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup loss, optimizer, and scheduler
        self.criterion: nn.Module = criterion
        self.optimizer: optim.Optimizer = optimizer
        self.scheduler: optim.lr_scheduler._LRScheduler = scheduler

        # Training configuration
        self.resume: Optional[str] = train_cfg.resume
        self.save_freq: int = train_cfg.save_freq
        self.save_best_only: bool = train_cfg.save_best_only
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
        self.best_val_loss: float = float("inf")
        self.best_val_acc: float = 0.0

        # Metrics tracking
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []

        # History containing all metrics
        self.history: Dict[str, List[float]] = {}

        # Resume from checkpoint if specified
        if self.resume:
            self._resume_from_checkpoint()

    def _resume_from_checkpoint(self) -> None:
        """Resume training from the latest checkpoint."""
        # Look for the latest checkpoint
        checkpoint_files = list(self.save_dir.glob("*.pth"))
        if not checkpoint_files:
            print("No checkpoint files found. Starting from scratch.")
            return

        # Find the latest checkpoint (prioritize 'last.pth', then most recent)
        latest_checkpoint = None
        if (self.save_dir / "last.pth").exists():
            latest_checkpoint = self.save_dir / "last.pth"
        else:
            # Find the most recent checkpoint file
            latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)

        if latest_checkpoint:
            print(f"Resuming from checkpoint: {latest_checkpoint}")
            self.load_checkpoint(str(latest_checkpoint))
            print(f"Resumed from epoch {self.current_epoch + 1}")

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

            # Handle mixup/cutmix labels
            if isinstance(target, tuple):
                # For mixup/cutmix, target is (labels, labels_mixed, lam)
                labels, labels_mixed, lam = target
                labels, labels_mixed = (
                    labels.to(self.device),
                    labels_mixed.to(self.device),
                )

                self.optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=self.device, enabled=self.use_amp):
                    output = self.model(data)
                    # Calculate mixup loss
                    loss = lam * self.criterion(output, labels) + (
                        1 - lam
                    ) * self.criterion(output, labels_mixed)

                # For accuracy, use original labels
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                total += labels.size(0)
            else:
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
    ) -> Dict[str, List[float]]:
        """Train the model for specified number of epochs.

        Args:
            num_epochs: Number of epochs to train

        Returns:
            Dictionary containing training history
        """
        patience_counter = 0

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch

            # Training
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validation
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

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
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint("best.pth", is_best=True)
                patience_counter = 0
            else:
                patience_counter += 1

            # Periodic checkpointing
            if self.save_freq and ((epoch + 1) % self.save_freq == 0):
                self.save_checkpoint(f"checkpoint_{epoch + 1:03d}.pth")

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

        self.history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
        }

        return self.history

    def save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            filename: Name of the checkpoint file
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
        }

        filepath = self.save_dir / filename
        torch.save(checkpoint, filepath)

        if is_best:
            best_filepath = self.save_dir / "best.pth"
            torch.save(checkpoint, best_filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint.

        Args:
            filepath: Path to the checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_val_acc = checkpoint["best_val_acc"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.train_accs = checkpoint["train_accs"]
        self.val_accs = checkpoint["val_accs"]

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
        report = classification_report(all_targets, all_predictions, output_dict=True)

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

    def visualize_metrics(self) -> None:
        """Visualize training and validation metrics."""
        # Training curves (Loss and Accuracy)
        plot_training_history(
            self.history, save_path=str(self.log_dir / "training_curves.png")
        )

        # Confusion matrix on validation set
        y_true, y_pred = self.predict_labels(self.val_loader)
        cm = confusion_matrix(y_true, y_pred)
        class_names = self.val_loader.dataset.get_class_names()
        plot_confusion_matrix(
            cm, class_names, save_path=str(self.log_dir / "confusion_matrix.png")
        )
