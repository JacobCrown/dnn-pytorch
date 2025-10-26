"""Visualizer for classification tasks."""

from pathlib import Path

import cv2
import matplotlib.patches as patches
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
from src.common.utils import plot_confusion_matrix
from src.core.metrics import BaseMetrics
from src.core.visualizer import BaseVisualizer


class ClassificationVisualizer(BaseVisualizer):
    """Visualizer class for classification tasks."""

    _TASK_TYPE: TaskType = TaskType.CLASSIFICATION

    @staticmethod
    def _denormalize_image(image: np.ndarray) -> np.ndarray:
        """Denormalize image for visualization.

        Args:
            image: Normalized image array (C, H, W) in range [0, 1] or normalized

        Returns:
            Denormalized image array (H, W, C) in range [0, 255]
        """
        # Transpose from (C, H, W) to (H, W, C)
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))

        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            # Common ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            # Check if image looks normalized (values around 0)
            if image.min() < 0 or image.max() < 1.5:
                # Denormalize
                image = image * std + mean

            # Clip and convert to [0, 255]
            image = np.clip(image * 255, 0, 255).astype(np.uint8)

        # Handle grayscale
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            squeezed = image.squeeze()
            converted = cv2.cvtColor(squeezed, cv2.COLOR_GRAY2RGB)  # type: ignore
            if converted is not None:
                image = converted

        return image

    @staticmethod
    def _get_samples_per_class(
        dataset: ClassificationDataset,
        num_samples_per_class: int,
        with_augmentations: bool = False,
    ) -> tuple[list[np.ndarray], list[int], list[str]]:
        """Get sample images for each class.

        Args:
            dataset: Classification dataset
            num_samples_per_class: Number of samples to get per class
            with_augmentations: Whether to apply augmentations

        Returns:
            Tuple of (images, labels, class_names)
        """
        class_names = dataset.get_class_names()
        num_classes = len(class_names)

        # Organize samples by class
        samples_by_class = {i: [] for i in range(num_classes)}
        for file_path, label in dataset.samples:
            samples_by_class[label].append(file_path)

        # Collect samples
        images = []
        labels = []

        for class_idx in range(num_classes):
            class_samples = samples_by_class[class_idx][:num_samples_per_class]

            for file_path in class_samples:
                # Load image without transforms
                image = cv2.imread(file_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: ignore
                else:
                    continue

                # Apply augmentations if requested
                if with_augmentations and dataset.transforms is not None:
                    augmented = dataset.transforms(image=image)
                    image = augmented["image"]
                    # Convert tensor to numpy if needed
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
        figsize: tuple[int, int],
    ) -> None:
        """Plot a grid of images.

        Args:
            images: List of images to plot
            labels: List of labels corresponding to images
            class_names: List of class names
            num_cols: Number of columns in the grid
            title: Title for the plot
            save_path: Path to save the plot
            figsize: Figure size
        """
        num_images = len(images)
        num_rows = (num_images + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Flatten axes for easier indexing
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        if num_cols == 1:
            axes = axes.reshape(-1, 1)

        for idx in range(num_rows * num_cols):
            row = idx // num_cols
            col = idx % num_cols
            ax = axes[row, col]

            if idx < num_images:
                image = ClassificationVisualizer._denormalize_image(images[idx])
                ax.imshow(image)
                ax.set_title(f"{class_names[labels[idx]]}", fontsize=10)
            else:
                ax.axis("off")

            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def visualize_example_images(
        data_loader: DataLoader,
        save_dir: Path | None = None,
        num_samples_per_class: int = 5,
        figsize: tuple[int, int] = (15, 10),
    ) -> None:
        """Visualize example training images in a grid.

        Creates two plots:
        1. Grid of images without augmentations
        2. Grid of images with augmentations

        Args:
            data_loader: DataLoader containing the dataset
            save_dir: Directory to save visualizations
            num_samples_per_class: Number of samples to show per class
            figsize: Figure size for each plot
        """
        if save_dir is None:
            save_dir = Path("visualizations")
        save_dir.mkdir(parents=True, exist_ok=True)

        dataset = data_loader.dataset
        if not isinstance(dataset, ClassificationDataset):
            raise ValueError("Dataset must be a ClassificationDataset")

        class_names = dataset.get_class_names()

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
            num_samples_per_class,
            "Example Images (No Augmentation)",
            save_dir / "example_images_no_aug.png",
            figsize,
        )

        # Plot with augmentations
        ClassificationVisualizer._plot_image_grid(
            images_aug,
            labels_aug,
            class_names,
            num_samples_per_class,
            "Example Images (With Augmentation)",
            save_dir / "example_images_with_aug.png",
            figsize,
        )

        print(f"Example images saved to {save_dir}")

    @staticmethod
    def visualize_data_distribution(
        data_loader: DataLoader,
        save_dir: Path | None = None,
        figsize: tuple[int, int] = (12, 6),
    ) -> None:
        """Visualize the distribution of classes in the dataset.

        Args:
            data_loader: DataLoader containing the dataset
            save_dir: Directory to save visualizations
            figsize: Figure size for the plot
        """
        if save_dir is None:
            save_dir = Path("visualizations")
        save_dir.mkdir(parents=True, exist_ok=True)

        dataset = data_loader.dataset
        if not isinstance(dataset, ClassificationDataset):
            raise ValueError("Dataset must be a ClassificationDataset")

        # Get data distribution
        data_distribution = dataset.get_data_distribution()

        class_names = [dist.class_name for dist in data_distribution]
        counts = [dist.count for dist in data_distribution]
        percentages = [dist.percentage for dist in data_distribution]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Bar plot with counts
        colors = sns.color_palette("husl", len(class_names))
        bars1 = ax1.bar(range(len(class_names)), counts, color=colors)
        ax1.set_xlabel("Class", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.set_title("Class Distribution (Counts)", fontsize=14, fontweight="bold")
        ax1.set_xticks(range(len(class_names)))
        ax1.set_xticklabels(class_names, rotation=45, ha="right")
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Pie chart with percentages
        ax2.pie(
            percentages,
            labels=class_names,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        ax2.set_title(
            "Class Distribution (Percentages)", fontsize=14, fontweight="bold"
        )

        plt.tight_layout()
        save_path = save_dir / "data_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Data distribution plot saved to {save_path}")

    @staticmethod
    def visualize_predictions(
        data_loader: DataLoader,
        model: nn.Module | None = None,
        save_dir: Path | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = False,
        num_samples_per_class: int = 5,
        num_cols: int = 5,
        figsize: tuple[int, int] = (15, 12),
    ) -> None:
        """Visualize model predictions on sample images.

        Green border = correct prediction
        Red border = wrong prediction

        Args:
            data_loader: DataLoader to get samples from
            model: Trained model for predictions
            save_dir: Directory to save visualizations
            device: Device to run model on
            use_amp: Whether to use automatic mixed precision
            num_samples_per_class: Number of samples to visualize per class
            num_cols: Number of columns in the grid
            figsize: Figure size for the plot
        """
        if model is None:
            raise ValueError("Model is required for visualizing predictions")

        if save_dir is None:
            save_dir = Path("visualizations")
        save_dir.mkdir(parents=True, exist_ok=True)

        dataset = data_loader.dataset
        if not isinstance(dataset, ClassificationDataset):
            raise ValueError("Dataset must be a ClassificationDataset")

        model = model.to(device)
        model.eval()

        class_names = dataset.get_class_names()
        num_classes = len(class_names)

        # Collect samples organized by class
        samples_by_class = {
            i: {"images": [], "true_labels": [], "pred_labels": []}
            for i in range(num_classes)
        }

        with torch.inference_mode():
            for data, target in tqdm(data_loader, desc="Collecting predictions"):
                data = data.to(device)

                # Handle tuple targets (from augmentation)
                if isinstance(target, tuple):
                    target = target[0]

                with autocast(device_type=device, enabled=use_amp):
                    output = model(data)
                    pred = output.argmax(dim=1)

                # Store samples organized by true label
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

                # Check if we have enough samples for all classes
                if all(
                    len(samples_by_class[i]["images"]) >= num_samples_per_class
                    for i in range(num_classes)
                ):
                    break

        # Flatten the samples to create the final lists
        images = []
        true_labels = []
        pred_labels = []

        for class_idx in range(num_classes):
            images.extend(samples_by_class[class_idx]["images"])
            true_labels.extend(samples_by_class[class_idx]["true_labels"])
            pred_labels.extend(samples_by_class[class_idx]["pred_labels"])

        # Calculate total samples and grid dimensions
        total_samples = len(images)
        num_rows = (total_samples + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        fig.suptitle(
            "Model Predictions (Green=Correct, Red=Wrong)",
            fontsize=16,
            fontweight="bold",
        )

        # Flatten axes for easier indexing
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        if num_cols == 1:
            axes = axes.reshape(-1, 1)

        for idx in range(num_rows * num_cols):
            row = idx // num_cols
            col = idx % num_cols
            ax = axes[row, col]

            if idx < len(images):
                image = ClassificationVisualizer._denormalize_image(images[idx])
                true_label = true_labels[idx]
                pred_label = pred_labels[idx]

                ax.imshow(image)

                # Determine border color
                is_correct = true_label == pred_label
                border_color = "green" if is_correct else "red"

                # Add colored border
                rect = patches.Rectangle(
                    (0, 0),
                    image.shape[1] - 1,
                    image.shape[0] - 1,
                    linewidth=3,
                    edgecolor=border_color,
                    facecolor="none",
                )
                ax.add_patch(rect)

                # Add title with true and predicted labels
                title_text = (
                    f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}"
                )
                ax.set_title(title_text, fontsize=9, color=border_color)
            else:
                ax.axis("off")

            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        save_path = save_dir / "predictions.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Predictions visualization saved to {save_path}")

        # Print accuracy statistics
        correct = sum(
            1 for t, p in zip(true_labels, pred_labels, strict=False) if t == p
        )
        accuracy = 100.0 * correct / len(true_labels)
        print(
            f"Accuracy on visualized samples: {accuracy:.2f}% ({correct}/{len(true_labels)})"
        )

    @staticmethod
    def visualize_metrics(
        metrics: BaseMetrics,
        save_dir: Path | None = None,
        data_loader: DataLoader | None = None,
        model: nn.Module | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = False,
        figsize_curves: tuple[int, int] = (12, 5),
        figsize_cm: tuple[int, int] = (10, 8),
    ) -> None:
        """Visualize training metrics.

        Creates three separate plots:
        1. Loss curves (train and validation)
        2. Accuracy curves (train and validation)
        3. Confusion matrix (if data_loader and model provided)

        Args:
            metrics: Metrics object with training history (BaseMetrics or subclass)
            save_dir: Directory to save visualizations
            data_loader: DataLoader for computing confusion matrix (optional)
            model: Trained model for confusion matrix (optional)
            device: Device to run model on
            use_amp: Whether to use automatic mixed precision
            figsize_curves: Figure size for loss/accuracy curves
            figsize_cm: Figure size for confusion matrix
        """
        if save_dir is None:
            save_dir = Path("visualizations")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: Loss curves
        ClassificationVisualizer._plot_loss_curves(metrics, save_dir, figsize_curves)

        # Plot 2: Accuracy curves (if available)
        if hasattr(metrics, "accuracies"):
            ClassificationVisualizer._plot_accuracy_curves(
                metrics, save_dir, figsize_curves
            )

        # Plot 3: Confusion matrix (requires data_loader and model)
        if data_loader is not None and model is not None:
            ClassificationVisualizer._plot_confusion_matrix(
                data_loader, model, save_dir, device, use_amp, figsize_cm
            )
        else:
            print("Skipping confusion matrix (requires data_loader and model)")

        print(f"All metrics visualizations saved to {save_dir}")

    @staticmethod
    def _plot_loss_curves(
        metrics: BaseMetrics, save_dir: Path, figsize: tuple[int, int]
    ) -> None:
        """Plot loss curves.

        Args:
            metrics: Metrics object
            save_dir: Directory to save the plot
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        for key, values in metrics.losses.items():
            if values:  # Only plot if there are values
                ax.plot(values, label=f"{key.capitalize()} Loss", linewidth=2)

        ax.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = save_dir / "loss_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Loss curves saved to {save_path}")

    @staticmethod
    def _plot_accuracy_curves(
        metrics: BaseMetrics, save_dir: Path, figsize: tuple[int, int]
    ) -> None:
        """Plot accuracy curves.

        Args:
            metrics: Metrics object (must have accuracies attribute)
            save_dir: Directory to save the plot
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Access accuracies - will work for any metrics subclass that has it
        accuracies = getattr(metrics, "accuracies", {})
        for key, values in accuracies.items():
            if values:  # Only plot if there are values
                ax.plot(values, label=f"{key.capitalize()} Accuracy", linewidth=2)

        ax.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = save_dir / "accuracy_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Accuracy curves saved to {save_path}")

    @staticmethod
    def _plot_confusion_matrix(
        data_loader: DataLoader,
        model: nn.Module,
        save_dir: Path,
        device: str,
        use_amp: bool,
        figsize: tuple[int, int],
    ) -> None:
        """Plot confusion matrix.

        Args:
            data_loader: DataLoader to evaluate on
            model: Trained model
            save_dir: Directory to save the plot
            device: Device to run model on
            use_amp: Whether to use automatic mixed precision
            figsize: Figure size
        """
        dataset = data_loader.dataset
        if not isinstance(dataset, ClassificationDataset):
            raise ValueError("Dataset must be a ClassificationDataset")

        model = model.to(device)
        model.eval()

        all_predictions = []
        all_targets = []

        with torch.inference_mode():
            for data, target in tqdm(data_loader, desc="Computing confusion matrix"):
                data = data.to(device)

                # Handle tuple targets
                if isinstance(target, tuple):
                    target = target[0]

                with autocast(device_type=device, enabled=use_amp):
                    output = model(data)
                    pred = output.argmax(dim=1)

                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        # Compute confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        class_names = dataset.get_class_names()

        # Plot using utility function
        save_path = save_dir / "confusion_matrix.png"
        plot_confusion_matrix(
            cm, class_names, save_path=str(save_path), figsize=figsize
        )

        print(f"Confusion matrix saved to {save_path}")
