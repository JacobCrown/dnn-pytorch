"""Visualizer for segmentation tasks."""

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

from src.common.enums import TaskType
from src.core.metrics import BaseMetrics
from src.core.visualizer import BaseVisualizer
from src.segmentation.dataset import SegmentationDataset


class SegmentationVisualizer(BaseVisualizer):
    """Visualizer class for segmentation tasks."""

    _TASK_TYPE: TaskType = TaskType.SEGMENTATION

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
    def _create_color_map(dataset: SegmentationDataset) -> dict[int, tuple[int, int, int]]:
        """Create a color map from class indices to RGB colors.

        Args:
            dataset: Segmentation dataset

        Returns:
            Dictionary mapping class indices to RGB colors
        """
        color_map = {}
        for rgb_color, class_idx in dataset.rgb_to_idx.items():
            color_map[class_idx] = rgb_color
        return color_map

    @staticmethod
    def _mask_to_rgb(mask: np.ndarray, color_map: dict[int, tuple[int, int, int]]) -> np.ndarray:
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

            mask = cv2.imread(mask_path)
            if mask is not None:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # type: ignore
                # Convert RGB mask to class indices
                mask = dataset._convert_pixel_mask_to_class_mask(mask)
            else:
                continue

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

    @staticmethod
    def visualize_example_images(
        data_loader: DataLoader,
        save_dir: Path | None = None,
        num_samples: int = 4,
        figsize: tuple[int, int] = (20, 10),
    ) -> None:
        """Visualize example training images with masks.

        Creates two plots:
        1. Images with masks without augmentations
        2. Images with masks with augmentations

        Args:
            data_loader: DataLoader containing the dataset
            save_dir: Directory to save visualizations
            num_samples: Number of samples to show
            figsize: Figure size for each plot
        """
        if save_dir is None:
            save_dir = Path("visualizations")
        save_dir.mkdir(parents=True, exist_ok=True)

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
            save_dir / "example_images_no_aug.png",
            figsize,
        )

        # Plot with augmentations
        SegmentationVisualizer._plot_images_with_masks(
            images_aug,
            masks_aug,
            color_map,
            "Example Images (With Augmentation)",
            save_dir / "example_images_with_aug.png",
            figsize,
        )

        print(f"Example images saved to {save_dir}")

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
            overlay = SegmentationVisualizer._overlay_mask_on_image(image, mask_rgb, alpha=0.4)

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
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def visualize_data_distribution(
        data_loader: DataLoader,
        save_dir: Path | None = None,
        figsize: tuple[int, int] = (14, 8),
    ) -> None:
        """Visualize the distribution of classes based on pixel counts.

        Args:
            data_loader: DataLoader containing the dataset
            save_dir: Directory to save visualizations
            figsize: Figure size for the plot
        """
        if save_dir is None:
            save_dir = Path("visualizations")
        save_dir.mkdir(parents=True, exist_ok=True)

        dataset = data_loader.dataset
        if not isinstance(dataset, SegmentationDataset):
            raise ValueError("Dataset must be a SegmentationDataset")

        # Count pixels for each class
        class_pixel_counts = {}
        color_map = SegmentationVisualizer._create_color_map(dataset)
        
        # Get class names from the dataset
        class_names = dataset.get_class_names()
        
        # Initialize counters
        for class_idx in color_map.keys():
            class_pixel_counts[class_idx] = 0

        print("Computing pixel distribution...")
        for image_path, mask_path in tqdm(dataset.samples, desc="Processing masks"):
            mask = cv2.imread(mask_path)
            if mask is not None:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # type: ignore
                mask = dataset._convert_pixel_mask_to_class_mask(mask)
                
                # Count pixels for each class
                unique, counts = np.unique(mask, return_counts=True)
                for class_idx, count in zip(unique, counts, strict=False):
                    if class_idx in class_pixel_counts:
                        class_pixel_counts[class_idx] += count

        # Calculate percentages
        total_pixels = sum(class_pixel_counts.values())
        class_percentages = {
            cls: (count / total_pixels * 100) for cls, count in class_pixel_counts.items()
        }

        # Sort by pixel count
        sorted_classes = sorted(class_pixel_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Filter out classes with very few pixels (< 0.1%)
        filtered_classes = [(cls, count) for cls, count in sorted_classes 
                           if class_percentages[cls] >= 0.1]

        class_indices = [cls for cls, _ in filtered_classes]
        filtered_class_names = [class_names[cls] for cls in class_indices]
        pixel_counts = [count for _, count in filtered_classes]
        percentages = [class_percentages[cls] for cls in class_indices]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Bar plot with pixel counts
        colors = sns.color_palette("husl", len(filtered_class_names))
        bars1 = ax1.barh(range(len(filtered_class_names)), pixel_counts, color=colors)
        ax1.set_xlabel("Pixel Count", fontsize=12)
        ax1.set_ylabel("Class", fontsize=12)
        ax1.set_title("Class Distribution (Pixel Counts)", fontsize=14, fontweight="bold")
        ax1.set_yticks(range(len(filtered_class_names)))
        ax1.set_yticklabels(filtered_class_names, fontsize=9)
        ax1.grid(axis="x", alpha=0.3)

        # Add value labels on bars
        for i, (bar, pct) in enumerate(zip(bars1, percentages, strict=False)):
            width = bar.get_width()
            ax1.text(
                width,
                bar.get_y() + bar.get_height() / 2.0,
                f" {pct:.1f}%",
                ha="left",
                va="center",
                fontsize=8,
            )

        # Pie chart with percentages (top 10 classes)
        top_n = min(10, len(percentages))
        top_indices = np.argsort(percentages)[-top_n:][::-1]
        top_names = [filtered_class_names[i] for i in top_indices]
        top_percentages = [percentages[i] for i in top_indices]
        top_colors = [colors[i] for i in top_indices]

        ax2.pie(
            top_percentages,
            labels=top_names,
            autopct="%1.1f%%",
            colors=top_colors,
            startangle=90,
        )
        ax2.set_title(
            f"Top {top_n} Classes (Percentages)", fontsize=14, fontweight="bold"
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
        num_samples: int = 4,
        figsize: tuple[int, int] = (20, 10),
    ) -> None:
        """Visualize model predictions on sample images.

        Args:
            data_loader: DataLoader to get samples from
            model: Trained model for predictions
            save_dir: Directory to save visualizations
            device: Device to run model on
            use_amp: Whether to use automatic mixed precision
            num_samples: Number of samples to visualize
            figsize: Figure size for the plot
        """
        if model is None:
            raise ValueError("Model is required for visualizing predictions")

        if save_dir is None:
            save_dir = Path("visualizations")
        save_dir.mkdir(parents=True, exist_ok=True)

        dataset = data_loader.dataset
        if not isinstance(dataset, SegmentationDataset):
            raise ValueError("Dataset must be a SegmentationDataset")

        model = model.to(device)
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

                data = data.to(device)
                target = target.to(device)

                with autocast(device_type=device, enabled=use_amp):
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
        save_path = save_dir / "predictions.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Predictions visualization saved to {save_path}")

        # Calculate and print IoU for visualized samples
        total_iou = 0.0
        for true_mask, pred_mask in zip(true_masks, pred_masks, strict=False):
            # Calculate IoU for this sample
            intersection = np.logical_and(true_mask == pred_mask, true_mask != 255).sum()
            union = (true_mask != 255).sum()
            if union > 0:
                iou = intersection / union
                total_iou += iou

        avg_iou = total_iou / len(true_masks) if true_masks else 0
        print(f"Average pixel accuracy on visualized samples: {avg_iou * 100:.2f}%")

    @staticmethod
    def visualize_metrics(
        metrics: BaseMetrics,
        save_dir: Path | None = None,
        figsize: tuple[int, int] = (12, 5),
    ) -> None:
        """Visualize training metrics.

        Creates plots for:
        1. Loss curves (train and validation)
        2. IoU curves (train and validation) if available

        Args:
            metrics: Metrics object with training history (BaseMetrics or subclass)
            save_dir: Directory to save visualizations
            figsize: Figure size for the plots
        """
        if save_dir is None:
            save_dir = Path("visualizations")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: Loss curves
        SegmentationVisualizer._plot_loss_curves(metrics, save_dir, figsize)

        # Plot 2: IoU curves (if available)
        if hasattr(metrics, "ious"):
            SegmentationVisualizer._plot_iou_curves(metrics, save_dir, figsize)

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
    def _plot_iou_curves(
        metrics: BaseMetrics, save_dir: Path, figsize: tuple[int, int]
    ) -> None:
        """Plot IoU/mIoU curves.

        Args:
            metrics: Metrics object (must have ious attribute)
            save_dir: Directory to save the plot
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

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
        save_path = save_dir / "iou_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"IoU curves saved to {save_path}")

