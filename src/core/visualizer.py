"""Base Visualizer class for different task types."""

from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from src.core.metrics import BaseMetrics


class BaseVisualizer(ABC):
    """Base Visualizer class that can be extended for different task types.

    This class provides common functionality for visualizing data and metrics
    that can be shared across classification, segmentation, and other tasks.
    """

    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        plt.switch_backend("Agg")

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

        if image.dtype != np.uint8:
            # ImageNet normalization parameters
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            # Denormalize if image appears to be normalized
            if image.min() < 0 or image.max() < 1.5:
                image = image * std + mean

            image = np.clip(image * 255, 0, 255).astype(np.uint8)

        # Convert grayscale to RGB for consistent output
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2RGB)  # type: ignore[assignment]

        return image

    @staticmethod
    @abstractmethod
    def visualize_example_images(data_loader: DataLoader, **kwargs) -> None:
        """Visualize example training images."""
        pass

    @staticmethod
    @abstractmethod
    def visualize_data_distribution(data_loader: DataLoader, **kwargs) -> None:
        """Visualize data distribution of the dataset."""
        pass

    @staticmethod
    @abstractmethod
    def visualize_predictions(data_loader: DataLoader, **kwargs) -> None:
        """Visualize predictions."""
        pass

    @staticmethod
    @abstractmethod
    def visualize_metrics(metrics: BaseMetrics, **kwargs) -> None:
        """Visualize training and validation metrics."""
        pass
