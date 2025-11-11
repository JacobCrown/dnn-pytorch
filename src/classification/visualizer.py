"""Improved Visualizer for classification tasks with many classes."""

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.amp import autocast  # type: ignore[reportMissingImports]
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.classification.dataset import ClassificationDataset
from src.common.enums import TaskType
from src.common.utils import DEVICE
from src.core.metrics import BaseMetrics
from src.core.visualizer import BaseVisualizer

logger = logging.getLogger(__name__)


class ClassificationVisualizer(BaseVisualizer):
    """Visualizer class for classification tasks with improved handling of many classes."""

    _TASK_TYPE: TaskType = TaskType.CLASSIFICATION

    @staticmethod
    def _calculate_optimal_figsize(
        num_images: int,
        num_cols: int,
        image_size: tuple[int, int] = (64, 64),
        min_cell_size: float = 1.5,
    ) -> tuple[float, float]:
        """Calculate optimal figure size based on number of images.

        Args:
            num_images: Total number of images to display
            num_cols: Number of columns
            image_size: Original image size (height, width)
            min_cell_size: Minimum size per cell in inches

        Returns:
            Optimal (width, height) for figure
        """
        num_rows = (num_images + num_cols - 1) // num_cols

        # Calculate based on image aspect ratio
        aspect_ratio = image_size[1] / image_size[0]  # width / height
        cell_height = min_cell_size
        cell_width = min_cell_size * aspect_ratio

        width = num_cols * cell_width
        height = num_rows * cell_height

        # Apply reasonable limits
        width = max(10, min(50, width))
        height = max(8, min(40, height))

        return (width, height)

    @staticmethod
    def _get_samples_per_class(
        dataset: ClassificationDataset,
        num_samples_per_class: int,
        with_augmentations: bool = False,
    ) -> tuple[list[np.ndarray], list[int], list[str]]:
        """Get sample images for each class."""
        class_names = dataset.get_class_names()
        num_classes = len(class_names)

        samples_by_class = {i: [] for i in range(num_classes)}
        for file_path, label in dataset.samples:
            samples_by_class[label].append(file_path)

        images = []
        labels = []

        for class_idx in range(num_classes):
            class_samples = samples_by_class[class_idx][:num_samples_per_class]

            for file_path in class_samples:
                image = cv2.imread(file_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    continue

                if with_augmentations and dataset.transforms is not None:
                    augmented = dataset.transforms(image=image)
                    image = augmented["image"]
                    if isinstance(image, torch.Tensor):
                        image = image.cpu().numpy()

                images.append(image)
                labels.append(class_idx)

        return images, labels, class_names

    @staticmethod
    def _plot_image_grid(
        images: list[np.ndarray],
        labels: list[int],
        class_names: list[str],
        num_cols: int,
        title: str,
        save_path: Path,
        figsize: tuple[float, float] | None = None,
        show_title_labels: bool = True,
    ) -> None:
        """Plot a grid of images with dynamic sizing."""
        num_images = len(images)
        num_rows = (num_images + num_cols - 1) // num_cols

        # Calculate optimal size if not provided
        if figsize is None:
            if num_images > 0:
                sample_img = ClassificationVisualizer._denormalize_image(images[0])
                figsize = ClassificationVisualizer._calculate_optimal_figsize(
                    num_images, num_cols, sample_img.shape[:2]
                )
            else:
                figsize = (15, 10)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)

        if num_rows == 1:
            axes = axes.reshape(1, -1)
        if num_cols == 1:
            axes = axes.reshape(-1, 1)

        # Calculate font size based on number of classes
        num_classes = len(set(labels))
        if num_classes > 50:
            title_fontsize = 6
        elif num_classes > 30:
            title_fontsize = 8
        else:
            title_fontsize = 10

        for idx in range(num_rows * num_cols):
            row = idx // num_cols
            col = idx % num_cols
            ax = axes[row, col]

            if idx < num_images:
                image = ClassificationVisualizer._denormalize_image(images[idx])
                ax.imshow(image)
                if show_title_labels:
                    ax.set_title(
                        f"{class_names[labels[idx]]}", fontsize=title_fontsize, pad=2
                    )
            else:
                ax.axis("off")

            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def visualize_example_images(
        self,
        data_loader: DataLoader,
        num_samples_per_class: int | None = None,
        num_cols: int | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Visualize example training images with automatic adjustment for many classes.

        Args:
            data_loader: DataLoader containing the dataset
            num_samples_per_class: Number of samples per class (auto-adjusted for many classes)
            num_cols: Number of columns (auto-adjusted for many classes)
            figsize: Figure size (auto-calculated if None)
        """
        dataset = data_loader.dataset
        if not isinstance(dataset, ClassificationDataset):
            raise ValueError("Dataset must be a ClassificationDataset")

        class_names = dataset.get_class_names()
        num_classes = len(class_names)

        # Auto-adjust parameters based on number of classes
        if num_samples_per_class is None:
            if num_classes > 50:
                num_samples_per_class = 2  # Mniej próbek dla wielu klas
            elif num_classes > 30:
                num_samples_per_class = 3
            else:
                num_samples_per_class = 5

        if num_cols is None:
            num_cols = num_samples_per_class

        logger.info(
            f"Visualizing {num_classes} classes with {num_samples_per_class} samples each"
        )

        # Get images without augmentations
        images_no_aug, labels_no_aug, _ = (
            ClassificationVisualizer._get_samples_per_class(
                dataset, num_samples_per_class, with_augmentations=False
            )
        )

        # Get images with augmentations
        images_aug, labels_aug, _ = ClassificationVisualizer._get_samples_per_class(
            dataset, num_samples_per_class, with_augmentations=True
        )

        # Plot without augmentations
        ClassificationVisualizer._plot_image_grid(
            images_no_aug,
            labels_no_aug,
            class_names,
            num_cols,
            "Example Images (No Augmentation)",
            self.save_dir / "example_images_no_aug.png",
            figsize,
        )

        # Plot with augmentations
        ClassificationVisualizer._plot_image_grid(
            images_aug,
            labels_aug,
            class_names,
            num_cols,
            "Example Images (With Augmentation)",
            self.save_dir / "example_images_with_aug.png",
            figsize,
        )

        logger.info(f"Example images saved to {self.save_dir}")

    def visualize_data_distribution(
        self,
        data_loader: DataLoader,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Visualize the distribution of classes with dynamic sizing."""
        dataset = data_loader.dataset
        if not isinstance(dataset, ClassificationDataset):
            raise ValueError("Dataset must be a ClassificationDataset")

        data_distribution = dataset.get_data_distribution()
        class_names = [dist.class_name for dist in data_distribution]
        counts = [dist.count for dist in data_distribution]

        num_classes = len(class_names)

        # Auto-calculate figure size based on number of classes
        if figsize is None:
            bar_width = max(12, num_classes * 0.3)
            bar_width = min(bar_width, 30)  # Max width
            figsize = (bar_width, 6)

        # Create figure with one subplot
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Bar plot with counts
        colors = sns.color_palette("husl", len(class_names))
        bars1 = ax.bar(range(len(class_names)), counts, color=colors)
        ax.set_xlabel("Class", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Class Distribution (Counts)", fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(class_names)))

        # Adjust label size and rotation based on number of classes
        if num_classes > 50:
            ax.set_xticklabels(class_names, rotation=90, ha="right", fontsize=6)
        elif num_classes > 30:
            ax.set_xticklabels(class_names, rotation=90, ha="right", fontsize=8)
        else:
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)

        ax.grid(axis="y", alpha=0.3)

        # Add value labels only if not too many classes
        if num_classes <= 30:
            for bar in bars1:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.tight_layout()
        save_path = self.save_dir / "data_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Data distribution plot saved to {save_path}")

    def visualize_predictions(
        self,
        data_loader: DataLoader,
        model: nn.Module,
        use_amp: bool = False,
        num_samples_per_class: int | None = None,
        num_cols: int | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Visualize model predictions with automatic adjustment for many classes."""
        dataset = data_loader.dataset
        if not isinstance(dataset, ClassificationDataset):
            raise ValueError("Dataset must be a ClassificationDataset")

        model = model.to(DEVICE)
        model.eval()

        class_names = dataset.get_class_names()
        num_classes = len(class_names)

        # Auto-adjust parameters
        if num_samples_per_class is None:
            if num_classes > 50:
                num_samples_per_class = 2
            elif num_classes > 30:
                num_samples_per_class = 3
            else:
                num_samples_per_class = 5

        if num_cols is None:
            num_cols = num_samples_per_class

        # Collect samples
        samples_by_class = {
            i: {"images": [], "true_labels": [], "pred_labels": []}
            for i in range(num_classes)
        }

        with torch.inference_mode():
            for data, target in tqdm(data_loader, desc="Collecting predictions"):
                data = data.to(DEVICE)

                if isinstance(target, tuple):
                    target = target[0]

                with autocast(device_type=DEVICE, enabled=use_amp):
                    output = model(data)
                    pred = output.argmax(dim=1)

                batch_images = data.cpu().numpy()
                batch_true = target.cpu().numpy()
                batch_pred = pred.cpu().numpy()

                for img, true_lbl, pred_lbl in zip(
                    batch_images, batch_true, batch_pred, strict=False
                ):
                    if (
                        len(samples_by_class[true_lbl]["images"])
                        < num_samples_per_class
                    ):
                        samples_by_class[true_lbl]["images"].append(img)
                        samples_by_class[true_lbl]["true_labels"].append(true_lbl)
                        samples_by_class[true_lbl]["pred_labels"].append(pred_lbl)

                if all(
                    len(samples_by_class[i]["images"]) >= num_samples_per_class
                    for i in range(num_classes)
                ):
                    break

        # Flatten samples
        images = []
        true_labels = []
        pred_labels = []

        for class_idx in range(num_classes):
            images.extend(samples_by_class[class_idx]["images"])
            true_labels.extend(samples_by_class[class_idx]["true_labels"])
            pred_labels.extend(samples_by_class[class_idx]["pred_labels"])

        # Calculate grid dimensions
        total_samples = len(images)
        num_rows = (total_samples + num_cols - 1) // num_cols

        # Calculate optimal size
        if figsize is None and total_samples > 0:
            sample_img = ClassificationVisualizer._denormalize_image(images[0])
            figsize = ClassificationVisualizer._calculate_optimal_figsize(
                total_samples, num_cols, sample_img.shape[:2], min_cell_size=2.0
            )

        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        fig.suptitle(
            "Model Predictions (Green=Correct, Red=Wrong)",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        if num_rows == 1:
            axes = axes.reshape(1, -1)
        if num_cols == 1:
            axes = axes.reshape(-1, 1)

        # Adjust font size
        if num_classes > 50:
            title_fontsize = 5
        elif num_classes > 30:
            title_fontsize = 7
        else:
            title_fontsize = 9

        for idx in range(num_rows * num_cols):
            row = idx // num_cols
            col = idx % num_cols
            ax = axes[row, col]

            if idx < len(images):
                image = ClassificationVisualizer._denormalize_image(images[idx])
                true_label = true_labels[idx]
                pred_label = pred_labels[idx]

                is_correct = true_label == pred_label
                border_color = "green" if is_correct else "red"

                # Add border by creating a larger image with padding
                border_width = 6

                # Create padded image with border color
                border_rgb = [0, 255, 0] if border_color == "green" else [255, 0, 0]

                # Create image with border (padding)
                padded_image = np.pad(
                    image,
                    (
                        (border_width, border_width),
                        (border_width, border_width),
                        (0, 0),
                    ),
                    mode="constant",
                    constant_values=0,
                )

                # Set border pixels to border color
                padded_image[:border_width, :] = border_rgb  # top
                padded_image[-border_width:, :] = border_rgb  # bottom
                padded_image[:, :border_width] = border_rgb  # left
                padded_image[:, -border_width:] = border_rgb  # right

                ax.imshow(padded_image)

                title_text = (
                    f"T: {class_names[true_label]}\nP: {class_names[pred_label]}"
                )
                ax.set_title(
                    title_text, fontsize=title_fontsize, color=border_color, pad=2
                )
            else:
                ax.axis("off")

            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        save_path = self.save_dir / "predictions.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Predictions visualization saved to {save_path}")

        correct = sum(
            1 for t, p in zip(true_labels, pred_labels, strict=False) if t == p
        )
        accuracy = 100.0 * correct / len(true_labels)
        logger.info(
            f"Accuracy on visualized samples: {accuracy:.2f}% ({correct}/{len(true_labels)})"
        )

    def visualize_metrics(
        self,
        metrics: BaseMetrics,
        data_loader: DataLoader | None = None,
        model: nn.Module | None = None,
        use_amp: bool = False,
        figsize_curves: tuple[float, float] = (12, 5),
        figsize_cm: tuple[float, float] | None = None,
    ) -> None:
        """Visualize training metrics with improved confusion matrix for many classes."""
        # Plot loss curves
        self._plot_loss_curves(metrics, figsize_curves)

        # Plot accuracy curves
        if hasattr(metrics, "accuracies"):
            self._plot_accuracy_curves(metrics, figsize_curves)

        # Plot confusion matrix with dynamic sizing
        if data_loader is not None and model is not None:
            dataset = data_loader.dataset
            if isinstance(dataset, ClassificationDataset):
                num_classes = len(dataset.get_class_names())

                # Calculate optimal confusion matrix size
                if figsize_cm is None:
                    cm_size = max(12, min(40, num_classes * 0.4))
                    figsize_cm = (cm_size, cm_size * 0.9)

                self._plot_confusion_matrix(data_loader, model, use_amp, figsize_cm)
        else:
            logger.info("Skipping confusion matrix (requires data_loader and model)")

        logger.info(f"All metrics visualizations saved to {self.save_dir}")

    def visualize_per_class_metrics(
        self,
        data_loader: DataLoader,
        model: nn.Module,
        use_amp: bool = False,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Visualize per-class accuracy as a bar plot.

        Args:
            data_loader: DataLoader for evaluation.
            model: The model to evaluate.
            use_amp: Whether to use automatic mixed precision.
            figsize: Figure size (auto-calculated if None).
        """
        dataset = data_loader.dataset
        if not isinstance(dataset, ClassificationDataset):
            raise ValueError("Dataset must be a ClassificationDataset")

        model = model.to(DEVICE)
        model.eval()

        all_predictions: list[int] = []
        all_targets: list[int] = []

        with torch.inference_mode():
            for data, target in tqdm(data_loader, desc="Evaluating"):
                data, target = data.to(DEVICE), target.to(DEVICE)

                with autocast(device_type=DEVICE, enabled=use_amp):
                    output = model(data)

                pred = output.argmax(dim=1, keepdim=True)

                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())

        class_names = dataset.get_class_names()
        cm = confusion_matrix(
            all_targets, all_predictions, labels=np.arange(len(class_names))
        )
        num_classes = len(class_names)

        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_acc = cm.diagonal() / cm.sum(axis=1)
            per_class_acc[np.isnan(per_class_acc)] = 0.0
        per_class_acc *= 100.0

        if figsize is None:
            bar_width = max(12, num_classes * 0.4)
            bar_width = min(bar_width, 40)
            figsize = (bar_width, 8)

        fig, ax = plt.subplots(figsize=figsize)
        colors = sns.color_palette("viridis", num_classes)
        bars = ax.bar(range(num_classes), per_class_acc, color=colors)

        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_ylim(0, 105)
        ax.set_title("Per-Class Accuracy", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(range(num_classes))

        if num_classes > 70:
            tick_fontsize = 4
        elif num_classes > 50:
            tick_fontsize = 5
        elif num_classes > 30:
            tick_fontsize = 6
        else:
            tick_fontsize = 8

        ax.set_xticklabels(class_names, rotation=90, ha="right", fontsize=tick_fontsize)

        if num_classes <= 30:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=tick_fontsize - 1 if tick_fontsize > 4 else 4,
                )

        plt.tight_layout()
        save_path = self.save_dir / "per_class_accuracy.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Per-class accuracy plot saved to {save_path}")

    def _plot_loss_curves(
        self, metrics: BaseMetrics, figsize: tuple[float, float]
    ) -> None:
        """Plot loss curves."""
        fig, ax = plt.subplots(figsize=figsize)

        for key, values in metrics.losses.items():
            if values:
                ax.plot(values, label=f"{key.capitalize()} Loss", linewidth=2)

        ax.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / "loss_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Loss curves saved to {save_path}")

    def _plot_accuracy_curves(
        self, metrics: BaseMetrics, figsize: tuple[float, float]
    ) -> None:
        """Plot accuracy curves."""
        fig, ax = plt.subplots(figsize=figsize)

        accuracies = getattr(metrics, "accuracies", {})
        for key, values in accuracies.items():
            if values:
                ax.plot(values, label=f"{key.capitalize()} Accuracy", linewidth=2)

        ax.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / "accuracy_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Accuracy curves saved to {save_path}")

    def _plot_confusion_matrix(
        self,
        data_loader: DataLoader,
        model: nn.Module,
        use_amp: bool,
        figsize: tuple[float, float],
    ) -> None:
        """Plot confusion matrix with improved handling for many classes."""
        dataset = data_loader.dataset
        if not isinstance(dataset, ClassificationDataset):
            raise ValueError("Dataset must be a ClassificationDataset")

        model = model.to(DEVICE)
        model.eval()

        all_predictions = []
        all_targets = []

        with torch.inference_mode():
            for data, target in tqdm(data_loader, desc="Computing confusion matrix"):
                data = data.to(DEVICE)

                if isinstance(target, tuple):
                    target = target[0]

                with autocast(device_type=DEVICE, enabled=use_amp):
                    output = model(data)
                    pred = output.argmax(dim=1)

                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        cm = confusion_matrix(all_targets, all_predictions)
        class_names = dataset.get_class_names()
        num_classes = len(class_names)

        # Create improved confusion matrix plot
        fig, ax = plt.subplots(figsize=figsize)

        # For many classes, don't show annotations
        if num_classes > 30:
            sns.heatmap(
                cm,
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues",
                ax=ax,
                cbar=True,
                annot=False,  # Wyłącz liczby
                square=True,
            )
        else:
            sns.heatmap(
                cm,
                xticklabels=class_names,
                yticklabels=class_names,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                cbar=True,
                square=True,
            )

        # Adjust tick label sizes
        if num_classes > 70:
            tick_fontsize = 4
        elif num_classes > 50:
            tick_fontsize = 5
        elif num_classes > 30:
            tick_fontsize = 6
        else:
            tick_fontsize = 8

        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=90, ha="right", fontsize=tick_fontsize
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tick_fontsize)

        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

        plt.tight_layout()

        # Save both PNG and SVG for scalability
        save_path_png = self.save_dir / "confusion_matrix.png"
        save_path_svg = self.save_dir / "confusion_matrix.svg"

        plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
        plt.savefig(save_path_svg, format="svg", bbox_inches="tight")
        plt.close()

        logger.info(f"Confusion matrix saved to {save_path_png} and {save_path_svg}")
