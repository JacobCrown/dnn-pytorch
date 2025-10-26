"""Base Visualizer class for different task types."""

from abc import ABC, abstractmethod
from pathlib import Path

from torch.utils.data import DataLoader

from src.common.enums import TaskType
from src.core.metrics import BaseMetrics


class BaseVisualizer(ABC):
    """Base Visualizer class that can be extended for different task types.

    This class provides common functionality for visualizing data and metrics
    that can be shared across classification, segmentation, and other tasks.
    """

    _TASK_TYPE: TaskType = TaskType.UNDEFINED

    def __init__(self, vis_save_dir: Path):
        self.vis_save_dir = vis_save_dir

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
