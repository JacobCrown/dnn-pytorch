"""Utility functions for classification tasks."""

import random
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset

from src.classification.metrics import ClassificationMetrics
from src.segmentation.metrics import SegmentationMetrics
from src.common.config import BaseTrainingConfig
from src.core.metrics import BaseMetrics


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_data_loaders(
    config: BaseTrainingConfig,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Create data loaders for training, validation, and test sets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        test_dataset: Test dataset (optional)
        config: Training configuration

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

    return train_loader, val_loader, test_loader


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """Plot confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: Names of the classes
        save_path: Path to save the plot (optional)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_training_history(
    metrics: BaseMetrics,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> None:
    """Plot training history.

    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot (optional)
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot loss
    for key, value in metrics.losses.items():
        ax1.plot(value, label=f"{key} Loss")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

    # Plot accuracy
    if isinstance(metrics, ClassificationMetrics):
        for key, value in metrics.accuracies.items():
            ax2.plot(value, label=f"{key} Acc")
            ax2.set_title("Accuracy")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy (%)")
            ax2.legend()
            ax2.grid(True)

    if isinstance(metrics, SegmentationMetrics):
        for key, value in metrics.ious.items():
            ax2.plot(value, label=f"{key} IoU")
            ax2.set_title("IoU")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("IoU")
            ax2.legend()
            ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()
