"""Improved Visualizer for classification tasks with many classes."""

import logging
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.amp import autocast  # type: ignore[reportMissingImports]
from torch.utils.data import DataLoader

from src.classification.dataset import ClassificationDataset
from src.common.logging import artifact_rel_to_experiment
from src.common.progress_tools import iterate_with_progress
from src.common.utils import DEVICE
from src.core.metrics import BaseMetrics
from src.core.visualizer import BaseVisualizer

logger = logging.getLogger(__name__)

PlotData = dict[str, list[np.ndarray]]
PredictionData = dict[int, dict[str, list[np.ndarray | int]]]


class ClassificationVisualizer(BaseVisualizer):
    """Visualizer class for classification tasks with improved handling of many classes."""

    def __init__(self, save_dir: Path):
        """Initialize the classification visualizer.

        Args:
            save_dir: Directory to save visualizations
        """
        super().__init__(save_dir)
        self._setup_plot_style()

    def _artifact_rel(self, file_path: Path | str) -> str:
        """Experiment-relative path for logs (cwd is experiment folder under Hydra)."""
        return artifact_rel_to_experiment(file_path, self.save_dir.parent)

    def _setup_plot_style(self) -> None:
        """Setup consistent plotting style."""
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 12
        sns.set_style("whitegrid")

    @staticmethod
    def _calculate_optimal_figsize(
        num_rows: int,
        num_cols: int,
        base_image_size: tuple[int, int] = (64, 64),
        scale_factor: float = 1.0,
    ) -> tuple[float, float]:
        """Calculate optimal figure size for image grids.

        Args:
            num_rows: Number of rows in the grid
            num_cols: Number of columns in the grid
            base_image_size: Base size of individual images (height, width)
            scale_factor: Scaling factor for the figure

        Returns:
            Optimal (width, height) for the figure
        """
        aspect_ratio = base_image_size[1] / base_image_size[0]  # width / height
        width = num_cols * aspect_ratio * scale_factor
        height = num_rows * scale_factor
        return (width, height)

    @staticmethod
    def _load_and_process_image(
        file_path: Path, transforms: A.Compose | None = None
    ) -> np.ndarray | None:
        """Load and process a single image.

        Args:
            file_path: Path to the image file
            transforms: Optional transforms to apply

        Returns:
            Processed image or None if loading failed
        """
        try:
            image = cv2.imread(str(file_path))
            if image is None:
                return None

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if transforms is not None:
                augmented = transforms(image=image)
                image = augmented["image"]
                if isinstance(image, torch.Tensor):
                    image = image.cpu().numpy()

            return image
        except Exception as e:
            logger.warning(f"Failed to load image {file_path}: {e}")
            return None

    def _get_samples_per_class(
        self,
        dataset: ClassificationDataset,
        num_samples_per_class: int,
        with_augmentations: bool = False,
    ) -> PlotData:
        """Get sample images for each class.

        Args:
            dataset: Classification dataset
            num_samples_per_class: Number of samples to collect per class
            with_augmentations: Whether to apply augmentations

        Returns:
            Dictionary mapping class names to lists of images
        """
        class_names = dataset.get_class_names()
        idx_to_class = dataset.get_idx_to_class()
        transforms = dataset.transforms if with_augmentations else None

        # Group samples by class
        samples_by_class: PlotData = {class_name: [] for class_name in class_names}
        file_paths_by_class: dict[str, list[Path]] = {
            class_name: [] for class_name in class_names
        }

        # Collect file paths
        for file_path, label in dataset.samples:
            class_name = idx_to_class[label]
            if len(file_paths_by_class[class_name]) >= num_samples_per_class:
                continue
            file_paths_by_class[class_name].append(Path(file_path))

        # Load and process images
        for class_name in class_names:
            file_paths = file_paths_by_class[class_name]
            for file_path in file_paths:
                image = self._load_and_process_image(file_path, transforms)
                if image is not None:
                    samples_by_class[class_name].append(image)

                if len(samples_by_class[class_name]) >= num_samples_per_class:
                    break

        return samples_by_class

    def _create_image_grid(
        self,
        samples_by_class: PlotData,
        title: str,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Create an image grid figure.

        Args:
            samples_by_class: Dictionary of class names to images
            title: Figure title

        Returns:
            Figure and axes array
        """
        num_classes = len(samples_by_class)
        max_images = max(len(images) for images in samples_by_class.values())

        figsize = self._calculate_optimal_figsize(
            num_rows=max_images, num_cols=num_classes, scale_factor=1.2
        )

        fig, axes = plt.subplots(
            max_images, num_classes, figsize=figsize, squeeze=False
        )
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)

        return fig, axes

    def _plot_image_grid(
        self,
        samples_by_class: PlotData,
        title: str,
        save_path: Path,
        show_title_labels: bool = True,
    ) -> None:
        """Plot a grid of images with dynamic sizing.

        Args:
            samples_by_class: Dictionary mapping class names to lists of images
            title: Title for the plot
            save_path: Where to save the plot
            show_title_labels: Whether to show class names as titles
        """
        _, axes = self._create_image_grid(samples_by_class, title)
        class_names = list(samples_by_class.keys())
        max_images = axes.shape[0]

        for col, class_name in enumerate(class_names):
            images = samples_by_class[class_name]
            for row in range(max_images):
                ax = axes[row, col]

                if row < len(images):
                    image = self._denormalize_image(images[row])
                    ax.imshow(image)

                    if show_title_labels and row == 0:
                        ax.set_title(class_name, fontsize=12, pad=2)
                else:
                    ax.axis("off")

                ax.axis("off")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.debug(f"Image grid saved to {save_path}")

    def visualize_example_images(
        self,
        data_loader: DataLoader,
        num_samples_per_class: int = 5,
    ) -> None:
        """Visualize example training images with automatic adjustment for many classes.

        Args:
            data_loader: DataLoader containing the dataset
            num_samples_per_class: Number of samples per class to display
        """
        dataset = self._validate_dataset(data_loader)

        # Get samples without and with augmentations
        samples_no_aug = self._get_samples_per_class(
            dataset, num_samples_per_class, with_augmentations=False
        )
        samples_with_aug = self._get_samples_per_class(
            dataset, num_samples_per_class, with_augmentations=True
        )

        # Create visualizations
        self._plot_image_grid(
            samples_no_aug,
            "Example Images (No Augmentation)",
            self.save_dir / "example_images_no_aug.png",
        )
        self._plot_image_grid(
            samples_with_aug,
            "Example Images (With Augmentation)",
            self.save_dir / "example_images_with_aug.png",
        )

        logger.info(
            "[viz] wrote example_images_no_aug.png + example_images_with_aug.png (%s)",
            self._artifact_rel(self.save_dir),
        )

    def _validate_dataset(self, data_loader: DataLoader) -> ClassificationDataset:
        """Validate that the dataset is a ClassificationDataset.

        Args:
            data_loader: DataLoader to validate

        Returns:
            Validated ClassificationDataset

        Raises:
            ValueError: If dataset is not a ClassificationDataset
        """
        dataset = data_loader.dataset
        if not isinstance(dataset, ClassificationDataset):
            raise ValueError("Dataset must be a ClassificationDataset")
        return dataset

    def visualize_data_distribution(
        self,
        data_loader: DataLoader,
        max_classes_for_labels: int = 50,
    ) -> None:
        """Visualize the distribution of classes with dynamic sizing.

        Args:
            data_loader: DataLoader containing the dataset
            max_classes_for_labels: Maximum number of classes to show value labels for
        """
        dataset = self._validate_dataset(data_loader)
        data_distribution = dataset.get_data_distribution()

        class_names = [dist.class_name for dist in data_distribution]
        counts = [dist.count for dist in data_distribution]
        num_classes = len(class_names)

        # Dynamic figure sizing
        fig_width = max(12, min(40, num_classes * 0.4))
        figsize = (fig_width, 8)

        _, ax = plt.subplots(figsize=figsize)
        colors = sns.color_palette("husl", num_classes)

        bars = ax.bar(range(num_classes), counts, color=colors)
        ax.set_xlabel("Class", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Class Distribution", fontsize=14, fontweight="bold")

        # Dynamic label sizing
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=90, ha="right", fontsize=12)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels only for reasonable number of classes
        if num_classes <= max_classes_for_labels:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        plt.tight_layout()
        save_path = self.save_dir / "data_distribution.png"
        plt.savefig(save_path)
        plt.close()
        logger.info("[viz] wrote %s", self._artifact_rel(save_path))

    def _get_predictions(
        self,
        data_loader: DataLoader,
        model: nn.Module,
        num_classes: int,
        num_samples_per_class: int = 5,
        use_amp: bool = False,
    ) -> PredictionData:
        """Get predictions from the model.

        Args:
            data_loader: DataLoader for inference
            model: Model to use for predictions
            num_classes: Number of classes
            num_samples_per_class: Number of samples to collect per class
            use_amp: Whether to use automatic mixed precision

        Returns:
            Dictionary containing prediction data per class
        """
        samples_by_class: PredictionData = {
            i: {"images": [], "true_labels": [], "pred_labels": []}
            for i in range(num_classes)
        }

        model = model.to(DEVICE)
        model.eval()

        with torch.inference_mode():
            for data, target in iterate_with_progress(
                data_loader,
                description="Collect predictions",
                transient=True,
            ):
                data, target = data.to(DEVICE), target.to(DEVICE)

                with autocast(device_type=DEVICE, enabled=use_amp):
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)

                # Process batch
                for img, true_lbl, pred_lbl in zip(
                    data.cpu().numpy(),
                    target.cpu().numpy(),
                    pred.cpu().numpy(),
                    strict=False,
                ):
                    true_lbl, pred_lbl = int(true_lbl), int(pred_lbl[0])

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

        return samples_by_class

    def _create_prediction_grid(
        self,
        class_names: list[str],
        num_samples_per_class: int,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Create a grid for prediction visualization.

        Args:
            samples_by_class: Prediction data per class
            class_names: List of class names
            num_samples_per_class: Number of samples per class

        Returns:
            Figure and axes array
        """
        num_classes = len(class_names)
        figsize = self._calculate_optimal_figsize(
            num_rows=num_samples_per_class, num_cols=num_classes, scale_factor=1.3
        )

        fig, axes = plt.subplots(
            num_samples_per_class, num_classes, figsize=figsize, squeeze=False
        )

        fig.suptitle(
            "Model Predictions (Green=Correct, Red=Wrong)",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        return fig, axes

    def _add_border_to_image(
        self, image: np.ndarray, border_color: str, border_width: int = 6
    ) -> np.ndarray:
        """Add a colored border to an image.

        Args:
            image: Input image
            border_color: Border color ('green' or 'red')
            border_width: Width of the border in pixels

        Returns:
            Image with border
        """
        border_rgb = [0, 255, 0] if border_color == "green" else [255, 0, 0]

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

        # Set border pixels
        padded_image[:border_width, :] = border_rgb  # top
        padded_image[-border_width:, :] = border_rgb  # bottom
        padded_image[:, :border_width] = border_rgb  # left
        padded_image[:, -border_width:] = border_rgb  # right

        return padded_image

    def visualize_predictions(
        self,
        data_loader: DataLoader,
        model: nn.Module,
        num_samples_per_class: int = 5,
        use_amp: bool = False,
    ) -> None:
        """Visualize model predictions with automatic adjustment for many classes.

        Args:
            data_loader: DataLoader for inference
            model: Model to visualize predictions for
            num_samples_per_class: Number of samples per class to show
            use_amp: Whether to use automatic mixed precision
        """
        dataset = self._validate_dataset(data_loader)
        class_names = dataset.get_class_names()
        num_classes = len(class_names)

        # Get predictions
        samples_by_class = self._get_predictions(
            data_loader, model, num_classes, num_samples_per_class, use_amp
        )

        # Create visualization
        _, axes = self._create_prediction_grid(class_names, num_samples_per_class)

        # Plot each sample
        for col, class_idx in enumerate(range(num_classes)):
            class_data = samples_by_class[class_idx]
            num_samples = len(class_data["images"])

            for row in range(num_samples_per_class):
                ax = axes[row, col]

                if row < num_samples:
                    image = class_data["images"][row]
                    true_label = class_data["true_labels"][row]
                    pred_label = class_data["pred_labels"][row]

                    # Process image
                    image = self._denormalize_image(image)
                    is_correct = true_label == pred_label
                    border_color = "green" if is_correct else "red"

                    image_with_border = self._add_border_to_image(image, border_color)
                    ax.imshow(image_with_border)

                    # Add title with true and predicted labels
                    title_text = (
                        f"T: {class_names[true_label]}\nP: {class_names[pred_label]}"
                    )
                    ax.set_title(title_text, fontsize=10, color=border_color, pad=2)
                else:
                    ax.axis("off")

                ax.axis("off")

        plt.tight_layout()
        save_path = self.save_dir / "predictions.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("[viz] wrote %s", self._artifact_rel(save_path))

    def visualize_metrics(
        self,
        metrics: BaseMetrics,
        data_loader: DataLoader | None = None,
        model: nn.Module | None = None,
        use_amp: bool = False,
        figsize_curves: tuple[float, float] = (12, 5),
        figsize_cm: tuple[float, float] | None = None,
    ) -> None:
        """Visualize training metrics with improved confusion matrix for many classes.

        Args:
            metrics: Metrics object containing training history
            data_loader: Optional DataLoader for confusion matrix
            model: Optional model for confusion matrix
            use_amp: Whether to use automatic mixed precision
            figsize_curves: Figure size for loss/accuracy curves
            figsize_cm: Figure size for confusion matrix (auto-calculated if None)
        """
        # Plot training curves
        self._plot_loss_curves(metrics, figsize_curves)

        if hasattr(metrics, "accuracies"):
            self._plot_accuracy_curves(metrics, figsize_curves)

        # Plot confusion matrix if data and model are provided
        if data_loader is not None and model is not None:
            dataset = data_loader.dataset
            if isinstance(dataset, ClassificationDataset):
                num_classes = len(dataset.get_class_names())

                if figsize_cm is None:
                    cm_size = max(12, min(40, num_classes * 0.4))
                    figsize_cm = (cm_size, cm_size * 0.9)

                self._plot_confusion_matrix(data_loader, model, use_amp, figsize_cm)
            else:
                logger.warning(
                    "Skipping confusion matrix - dataset is not ClassificationDataset"
                )
        else:
            logger.debug(
                "[viz] skipping confusion matrix (no data_loader or model on this plot pass)"
            )

        logger.debug("[viz] metrics plots under %s", self._artifact_rel(self.save_dir))

    def visualize_per_class_metrics(
        self,
        data_loader: DataLoader,
        model: nn.Module,
        use_amp: bool = False,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Visualize per-class accuracy as a bar plot.

        Args:
            data_loader: DataLoader for evaluation
            model: Model to evaluate
            use_amp: Whether to use automatic mixed precision
            figsize: Figure size (auto-calculated if None)
        """
        dataset = self._validate_dataset(data_loader)
        class_names = dataset.get_class_names()
        num_classes = len(class_names)

        # Get predictions
        all_predictions, all_targets = self._get_all_predictions(
            data_loader, model, use_amp
        )

        # Calculate per-class accuracy
        per_class_acc = self._calculate_per_class_accuracy(
            all_targets, all_predictions, num_classes
        )

        # Create visualization
        self._plot_per_class_accuracy(per_class_acc, class_names, figsize)

    def _get_all_predictions(
        self,
        data_loader: DataLoader,
        model: nn.Module,
        use_amp: bool,
    ) -> tuple[list[int], list[int]]:
        """Get all predictions and targets from the model.

        Args:
            data_loader: DataLoader for inference
            model: Model to evaluate
            use_amp: Whether to use automatic mixed precision

        Returns:
            Tuple of (predictions, targets)
        """
        model = model.to(DEVICE)
        model.eval()

        all_predictions: list[int] = []
        all_targets: list[int] = []

        with torch.inference_mode():
            for data, target in iterate_with_progress(
                data_loader,
                description="Eval · per-class",
                transient=True,
            ):
                data, target = data.to(DEVICE), target.to(DEVICE)

                with autocast(device_type=DEVICE, enabled=use_amp):
                    output = model(data)

                pred = output.argmax(dim=1)
                all_predictions.extend(pred.cpu().numpy().tolist())
                all_targets.extend(target.cpu().numpy().tolist())

        return all_predictions, all_targets

    def _calculate_per_class_accuracy(
        self,
        targets: list[int],
        predictions: list[int],
        num_classes: int,
    ) -> np.ndarray:
        """Calculate per-class accuracy.

        Args:
            targets: True labels
            predictions: Predicted labels
            num_classes: Number of classes

        Returns:
            Array of per-class accuracies in percentage
        """
        cm = confusion_matrix(targets, predictions, labels=np.arange(num_classes))

        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_acc = cm.diagonal() / cm.sum(axis=1)
            per_class_acc = np.nan_to_num(per_class_acc, nan=0.0)

        return per_class_acc * 100.0

    def _plot_per_class_accuracy(
        self,
        per_class_acc: np.ndarray,
        class_names: list[str],
        figsize: tuple[float, float] | None,
    ) -> None:
        """Plot per-class accuracy bar chart.

        Args:
            per_class_acc: Per-class accuracy values
            class_names: List of class names
            figsize: Figure size (auto-calculated if None)
        """
        num_classes = len(class_names)

        if figsize is None:
            fig_width = max(12, min(40, num_classes * 0.4))
            figsize = (fig_width, 8)

        fig, ax = plt.subplots(figsize=figsize)
        colors = sns.color_palette("viridis", num_classes)
        bars = ax.bar(range(num_classes), per_class_acc, color=colors)

        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_ylim(0, 105)
        ax.set_title("Per-Class Accuracy", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Dynamic tick formatting
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=90, ha="right", fontsize=12)

        # Add value labels for reasonable number of classes
        if num_classes <= 30:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        plt.tight_layout()
        save_path = self.save_dir / "per_class_accuracy.png"
        plt.savefig(save_path)
        plt.close()
        logger.info("[viz] wrote %s", self._artifact_rel(save_path))

    def _plot_loss_curves(
        self, metrics: BaseMetrics, figsize: tuple[float, float]
    ) -> None:
        """Plot loss curves.

        Args:
            metrics: Metrics object containing loss history
            figsize: Figure size
        """
        _, ax = plt.subplots(figsize=figsize)

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
        plt.savefig(save_path)
        plt.close()
        logger.debug(f"Loss curves saved to {save_path}")

    def _plot_accuracy_curves(
        self, metrics: BaseMetrics, figsize: tuple[float, float]
    ) -> None:
        """Plot accuracy curves.

        Args:
            metrics: Metrics object containing accuracy history
            figsize: Figure size
        """
        _, ax = plt.subplots(figsize=figsize)

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
        plt.savefig(save_path)
        plt.close()
        logger.debug(f"Accuracy curves saved to {save_path}")

    def _plot_confusion_matrix(
        self,
        data_loader: DataLoader,
        model: nn.Module,
        use_amp: bool,
        figsize: tuple[float, float],
    ) -> None:
        """Plot confusion matrix with improved handling for many classes.

        Args:
            data_loader: DataLoader for evaluation
            model: Model to evaluate
            use_amp: Whether to use automatic mixed precision
            figsize: Figure size for the confusion matrix
        """
        dataset = self._validate_dataset(data_loader)
        class_names = dataset.get_class_names()
        num_classes = len(class_names)

        # Get predictions
        all_predictions, all_targets = self._get_all_predictions(
            data_loader, model, use_amp
        )

        # Create confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)

        # Create plot
        _, ax = plt.subplots(figsize=figsize)

        # For many classes, don't show annotations
        show_annot = num_classes <= 30

        sns.heatmap(
            cm,
            xticklabels=class_names,
            yticklabels=class_names,
            annot=show_annot,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar=True,
            square=True,
        )

        # Adjust tick labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right", fontsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

        plt.tight_layout()

        save_path_png = self.save_dir / "confusion_matrix.png"

        plt.savefig(save_path_png)
        plt.close()

        logger.info("[viz] wrote %s", self._artifact_rel(save_path_png))
