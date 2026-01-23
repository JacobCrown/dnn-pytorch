"""Visualizer for segmentation tasks."""

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.amp import autocast  # type: ignore[reportMissingImports]
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.utils import DEVICE
from src.core.metrics import BaseMetrics
from src.core.visualizer import BaseVisualizer
from src.segmentation.dataset import DataDistribution, SegmentationDataset

logger = logging.getLogger(__name__)


class SegmentationVisualizer(BaseVisualizer):
    """Visualizer class for segmentation tasks."""

    @staticmethod
    def _create_color_map(
        dataset: SegmentationDataset,
    ) -> dict[int, tuple[int, int, int]]:
        """Create a color map from class indices to RGB colors.

        Args:
            dataset: Segmentation dataset

        Returns:
            Dictionary mapping class indices to RGB colors
        """
        return dict(dataset.class_idx_to_rgb)

    @staticmethod
    def _mask_to_rgb(
        mask: np.ndarray, color_map: dict[int, tuple[int, int, int]]
    ) -> np.ndarray:
        """Convert a class mask to an RGB visualization.

        Args:
            mask: Class mask of shape (H, W) with class indices
            color_map: Dictionary mapping class indices to RGB colors

        Returns:
            RGB mask of shape (H, W, 3)
        """
        h, w = mask.shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for class_idx, color in color_map.items():
            rgb_mask[mask == class_idx] = color

        return rgb_mask

    @staticmethod
    def _overlay_mask_on_image(
        image: np.ndarray, mask: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        """Overlay a colored mask on an image.

        Args:
            image: RGB image of shape (H, W, 3)
            mask: RGB mask of shape (H, W, 3)
            alpha: Transparency of the mask overlay

        Returns:
            Blended image with mask overlay
        """
        return cv2.addWeighted(image, 1 - alpha, mask, alpha, 0)  # type: ignore

    @staticmethod
    def _get_sample_images(
        dataset: SegmentationDataset,
        num_samples: int,
        with_augmentations: bool = False,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Get sample images and masks.

        Args:
            dataset: Segmentation dataset
            num_samples: Number of samples to get
            with_augmentations: Whether to apply augmentations

        Returns:
            Tuple of (images, masks)
        """
        images = []
        masks = []

        num_samples = min(num_samples, len(dataset.samples))

        for i in range(num_samples):
            image_path, mask_path = dataset.samples[i]

            # Load image and mask without transforms
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: ignore
            else:
                continue

            mask = np.load(mask_path.with_suffix(".npy"))

            # Apply augmentations if requested
            if with_augmentations and dataset.transforms is not None:
                augmented = dataset.transforms(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

                # Convert tensor to numpy if needed
                if isinstance(image, torch.Tensor):
                    image = image.cpu().numpy()
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()

            images.append(image)
            masks.append(mask)

        return images, masks

    def visualize_example_images(
        self,
        data_loader: DataLoader,
        num_samples: int = 4,
        figsize: tuple[int, int] = (20, 10),
    ) -> None:
        """Visualize example training images with masks.

        Creates two plots:
        1. Images with masks without augmentations
        2. Images with masks with augmentations

        Args:
            data_loader: DataLoader containing the dataset
            num_samples: Number of samples to show
            figsize: Figure size for each plot
        """
        dataset = data_loader.dataset
        if not isinstance(dataset, SegmentationDataset):
            raise ValueError("Dataset must be a SegmentationDataset")

        color_map = SegmentationVisualizer._create_color_map(dataset)

        # Get images without augmentations
        images_no_aug, masks_no_aug = SegmentationVisualizer._get_sample_images(
            dataset, num_samples, with_augmentations=False
        )

        # Get images with augmentations
        images_aug, masks_aug = SegmentationVisualizer._get_sample_images(
            dataset, num_samples, with_augmentations=True
        )

        # Plot without augmentations
        SegmentationVisualizer._plot_images_with_masks(
            images_no_aug,
            masks_no_aug,
            color_map,
            "Example Images (No Augmentation)",
            self.save_dir / "example_images_no_aug.png",
            figsize,
        )

        # Plot with augmentations
        SegmentationVisualizer._plot_images_with_masks(
            images_aug,
            masks_aug,
            color_map,
            "Example Images (With Augmentation)",
            self.save_dir / "example_images_with_aug.png",
            figsize,
        )

        logger.info(f"Example images saved to {self.save_dir}")

    @staticmethod
    def _plot_images_with_masks(
        images: list[np.ndarray],
        masks: list[np.ndarray],
        color_map: dict[int, tuple[int, int, int]],
        title: str,
        save_path: Path,
        figsize: tuple[int, int],
    ) -> None:
        """Plot images with their corresponding masks.

        Args:
            images: List of images
            masks: List of masks
            color_map: Dictionary mapping class indices to RGB colors
            title: Title for the plot
            save_path: Path to save the plot
            figsize: Figure size
        """
        num_samples = len(images)
        fig, axes = plt.subplots(num_samples, 3, figsize=figsize)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Handle single sample case
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for idx in range(num_samples):
            # Denormalize image
            image = SegmentationVisualizer._denormalize_image(images[idx])

            # Handle mask - squeeze if it has an extra dimension
            mask = masks[idx]
            if mask.ndim == 3:
                mask = mask.squeeze()

            # Convert mask to RGB
            mask_rgb = SegmentationVisualizer._mask_to_rgb(mask, color_map)

            # Create overlay
            overlay = SegmentationVisualizer._overlay_mask_on_image(
                image, mask_rgb, alpha=0.4
            )

            # Plot original image
            axes[idx, 0].imshow(image)
            axes[idx, 0].set_title("Image")
            axes[idx, 0].axis("off")

            # Plot mask
            axes[idx, 1].imshow(mask_rgb)
            axes[idx, 1].set_title("Mask")
            axes[idx, 1].axis("off")

            # Plot overlay
            axes[idx, 2].imshow(overlay)
            axes[idx, 2].set_title("Overlay")
            axes[idx, 2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def visualize_data_distribution(
        self,
        data_loader: DataLoader,
        data_distribution: list[DataDistribution] | None = None,
        min_percentage: float = 0.1,
        figsize: tuple[int, int] = (14, 8),
    ) -> None:
        """Visualize the distribution of classes based on pixel counts.

        Args:
            data_loader: DataLoader containing the dataset
            data_distribution: Optional precomputed data distribution
            min_percentage: Minimum percentage threshold for plotting
            figsize: Figure size for the plot
        """
        dataset = data_loader.dataset
        if not isinstance(dataset, SegmentationDataset):
            raise ValueError("Dataset must be a SegmentationDataset")

        if data_distribution is None:
            logger.info("Using dataset data distribution for visualization...")
            data_distribution = dataset.get_data_distribution()
        if not data_distribution:
            logger.warning("No data distribution available to visualize.")
            return

        # Sort by pixel count
        sorted_distribution = sorted(
            data_distribution, key=lambda item: item.count, reverse=True
        )

        # Filter out classes with very few pixels
        filtered_distribution = [
            item for item in sorted_distribution if item.percentage >= min_percentage
        ]
        if not filtered_distribution:
            logger.warning(
                "No classes exceed min_percentage=%.2f%%; skipping plot.",
                min_percentage,
            )
            return

        filtered_class_names = [item.class_name for item in filtered_distribution]
        pixel_counts = [item.count for item in filtered_distribution]
        percentages = [item.percentage for item in filtered_distribution]

        # Create figure with one subplot
        _, ax = plt.subplots(1, 1, figsize=figsize)

        # Bar plot with pixel counts
        colors = sns.color_palette("husl", len(filtered_class_names))
        bars1 = ax.barh(range(len(filtered_class_names)), pixel_counts, color=colors)
        ax.set_xlabel("Pixel Count", fontsize=12)
        ax.set_ylabel("Class", fontsize=12)
        ax.set_title(
            "Class Distribution (Pixel Counts)", fontsize=14, fontweight="bold"
        )
        ax.set_yticks(range(len(filtered_class_names)))
        ax.set_yticklabels(filtered_class_names, fontsize=9)
        ax.grid(axis="x", alpha=0.3)

        # Add value labels on bars
        for _i, (bar, pct) in enumerate(zip(bars1, percentages, strict=False)):
            width = bar.get_width()
            ax.text(
                width,
                bar.get_y() + bar.get_height() / 2.0,
                f" {pct:.1f}%",
                ha="left",
                va="center",
                fontsize=8,
            )

        plt.tight_layout()
        save_path = self.save_dir / "data_distribution.png"
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Data distribution plot saved to {save_path}")

    def visualize_predictions(
        self,
        data_loader: DataLoader,
        model: nn.Module,
        use_amp: bool = False,
        num_samples: int = 4,
        figsize: tuple[int, int] = (20, 10),
    ) -> None:
        """Visualize model predictions on sample images.

        Args:
            data_loader: DataLoader to get samples from
            model: Trained model for predictions
            use_amp: Whether to use automatic mixed precision
            num_samples: Number of samples to visualize
            figsize: Figure size for the plot
        """
        dataset = data_loader.dataset
        if not isinstance(dataset, SegmentationDataset):
            raise ValueError("Dataset must be a SegmentationDataset")

        model = model.to(DEVICE)
        model.eval()

        color_map = SegmentationVisualizer._create_color_map(dataset)

        # Collect samples
        images = []
        true_masks = []
        pred_masks = []

        with torch.inference_mode():
            for data, target in tqdm(data_loader, desc="Collecting predictions"):
                if len(images) >= num_samples:
                    break

                data = data.to(DEVICE)
                target = target.to(DEVICE)

                with autocast(device_type=DEVICE, enabled=use_amp):
                    output = model(data)
                    pred = output.argmax(dim=1)

                # Store samples
                batch_images = data.cpu().numpy()
                batch_true = target.cpu().numpy()
                batch_pred = pred.cpu().numpy()

                for img, true_mask, pred_mask in zip(
                    batch_images, batch_true, batch_pred, strict=False
                ):
                    if len(images) < num_samples:
                        images.append(img)
                        true_masks.append(true_mask)
                        pred_masks.append(pred_mask)

        # Plot predictions
        fig, axes = plt.subplots(num_samples, 4, figsize=figsize)
        fig.suptitle(
            "Model Predictions (Image | Ground Truth | Prediction | Overlay)",
            fontsize=16,
            fontweight="bold",
        )

        # Handle single sample case
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for idx in range(num_samples):
            # Denormalize image
            image = SegmentationVisualizer._denormalize_image(images[idx])
            true_mask = true_masks[idx]
            pred_mask = pred_masks[idx]

            # Convert masks to RGB
            true_mask_rgb = SegmentationVisualizer._mask_to_rgb(true_mask, color_map)
            pred_mask_rgb = SegmentationVisualizer._mask_to_rgb(pred_mask, color_map)

            # Create overlay
            overlay = SegmentationVisualizer._overlay_mask_on_image(
                image, pred_mask_rgb, alpha=0.4
            )

            # Plot original image
            axes[idx, 0].imshow(image)
            axes[idx, 0].set_title("Image")
            axes[idx, 0].axis("off")

            # Plot ground truth mask
            axes[idx, 1].imshow(true_mask_rgb)
            axes[idx, 1].set_title("Ground Truth")
            axes[idx, 1].axis("off")

            # Plot predicted mask
            axes[idx, 2].imshow(pred_mask_rgb)
            axes[idx, 2].set_title("Prediction")
            axes[idx, 2].axis("off")

            # Plot overlay
            axes[idx, 3].imshow(overlay)
            axes[idx, 3].set_title("Overlay")
            axes[idx, 3].axis("off")

        plt.tight_layout()
        save_path = self.save_dir / "predictions.png"
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Predictions visualization saved to {save_path}")

        # Calculate and log IoU for visualized samples
        total_iou = 0.0
        for true_mask, pred_mask in zip(true_masks, pred_masks, strict=False):
            # Calculate IoU for this sample
            intersection = np.logical_and(
                true_mask == pred_mask, true_mask != 255
            ).sum()
            union = (true_mask != 255).sum()
            if union > 0:
                iou = intersection / union
                total_iou += iou

        avg_iou = total_iou / len(true_masks) if true_masks else 0
        logger.info(
            f"Average pixel accuracy on visualized samples: {avg_iou * 100:.2f}%"
        )

    def visualize_metrics(
        self,
        metrics: BaseMetrics,
        figsize: tuple[int, int] = (12, 5),
    ) -> None:
        """Visualize training metrics.

        Creates plots for:
        1. Loss curves (train and validation)
        2. IoU curves (train and validation) if available

        Args:
            metrics: Metrics object with training history (BaseMetrics or subclass)
            figsize: Figure size for the plots
        """
        # Plot 1: Loss curves
        self._plot_loss_curves(metrics, figsize)

        # Plot 2: IoU curves (if available)
        if hasattr(metrics, "ious"):
            self._plot_iou_curves(metrics, figsize)

        logger.info(f"All metrics visualizations saved to {self.save_dir}")

    def visualize_per_class_metrics(
        self,
        data_loader: DataLoader,
        model: nn.Module,
        use_amp: bool = False,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Visualize per-class mean Intersection over Union (mIoU).

        Args:
            data_loader: DataLoader for evaluation.
            model: The model to evaluate.
            use_amp: Whether to use automatic mixed precision.
            figsize: Figure size (auto-calculated if None).
        """
        dataset = data_loader.dataset
        if not isinstance(dataset, SegmentationDataset):
            raise ValueError("Dataset must be a SegmentationDataset")

        model = model.to(DEVICE)
        model.eval()

        class_names = dataset.get_class_names()
        num_classes = len(class_names)

        intersection_per_class = np.zeros(num_classes)
        union_per_class = np.zeros(num_classes)

        with torch.inference_mode():
            for data, target in tqdm(data_loader, desc="Evaluating mIoU per class"):
                data = data.to(DEVICE)
                target = target.to(DEVICE)

                with autocast(device_type=DEVICE, enabled=use_amp):
                    output = model(data)

                pred = output.argmax(dim=1)

                # Move to CPU for numpy operations
                pred_np = pred.cpu().numpy()
                target_np = target.cpu().numpy()

                for c in range(num_classes):
                    intersection = np.sum((pred_np == c) & (target_np == c))
                    union = np.sum((pred_np == c) | (target_np == c))
                    intersection_per_class[c] += intersection
                    union_per_class[c] += union

        with np.errstate(divide="ignore", invalid="ignore"):
            iou_per_class = intersection_per_class / union_per_class
            iou_per_class[np.isnan(iou_per_class)] = 0.0

        # Sort classes by mIoU for better visualization
        sorted_indices = np.argsort(iou_per_class)[::-1]
        sorted_iou_per_class = iou_per_class[sorted_indices]
        sorted_class_names = [class_names[i] for i in sorted_indices]

        # Dynamic figsize for horizontal bar plot
        if figsize is None:
            # Adjust height based on number of classes
            bar_height = max(10, num_classes * 0.35)
            bar_height = min(bar_height, 50)  # Max height
            figsize = (12, bar_height)

        _, ax = plt.subplots(figsize=figsize)
        colors = sns.color_palette("viridis", num_classes)
        # mypy doesn't like this, but it's valid
        sorted_colors = [colors[i] for i in sorted_indices]  # type: ignore

        # Horizontal bar plot
        bars = ax.barh(range(num_classes), sorted_iou_per_class, color=sorted_colors)

        ax.set_xlabel("Mean IoU", fontsize=12)
        ax.set_ylabel("Class", fontsize=12)
        ax.set_xlim(0, 1.05)
        ax.set_title(
            "Per-Class Mean Intersection over Union (mIoU)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)
        ax.set_yticks(range(num_classes))
        ax.invert_yaxis()  # Display highest IoU at the top

        ax.set_yticklabels(sorted_class_names, fontsize=9)

        # Add value labels on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                ha="left",
                va="center",
                fontsize=8,
            )

        plt.tight_layout()
        save_path = self.save_dir / "per_class_miou.png"
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Per-class mIoU plot saved to {save_path}")

    def _plot_loss_curves(self, metrics: BaseMetrics, figsize: tuple[int, int]) -> None:
        """Plot loss curves.

        Args:
            metrics: Metrics object
            figsize: Figure size
        """
        _, ax = plt.subplots(figsize=figsize)

        for key, values in metrics.losses.items():
            if values:  # Only plot if there are values
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

        logger.info(f"Loss curves saved to {save_path}")

    def _plot_iou_curves(self, metrics: BaseMetrics, figsize: tuple[int, int]) -> None:
        """Plot IoU/mIoU curves.

        Args:
            metrics: Metrics object (must have ious attribute)
            figsize: Figure size
        """
        _, ax = plt.subplots(figsize=figsize)

        # Access ious - will work for any metrics subclass that has it
        ious = getattr(metrics, "ious", {})
        for key, values in ious.items():
            if values:  # Only plot if there are values
                ax.plot(values, label=f"{key.capitalize()} mIoU", linewidth=2)

        ax.set_title("Training and Validation mIoU", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("mIoU", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / "iou_curves.png"
        plt.savefig(save_path)
        plt.close()

        logger.info(f"IoU curves saved to {save_path}")
