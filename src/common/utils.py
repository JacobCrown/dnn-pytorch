"""Utility functions for classification tasks."""

import logging
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

logger = logging.getLogger(__name__)


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


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 8),
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


def get_opencv_interpolation_flag(name: str) -> int:
    mapping = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "bilinear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    result = mapping.get(name.lower())
    if result is None:
        logger.warning(
            f"Interpolation name {name} not found, using default interpolation flag cv2.INTER_LINEAR"
        )
        return cv2.INTER_LINEAR
    return result
