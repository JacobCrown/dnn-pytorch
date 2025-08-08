"""Utility functions for classification tasks."""

import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from PIL import Image
from torchvision import datasets


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
    train_dataset,
    val_dataset=None,
    test_dataset=None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Create data loaders for training, validation, and test sets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        test_dataset: Test dataset (optional)
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader


def split_dataset(
    dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset, torch.utils.data.Subset]:
    """Split a dataset into train, validation, and test sets.

    Args:
        dataset: Dataset to split
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_subset, val_subset, test_subset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1.0"
    )

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_subset, val_subset, test_subset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    return train_subset, val_subset, test_subset


def calculate_metrics(
    y_true: List[int], y_pred: List[int], class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes

    Returns:
        Dictionary containing metrics
    """
    accuracy = accuracy_score(y_true, y_pred)

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
    }


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
    history: Dict[str, List[float]],
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
    if "train_losses" in history and "val_losses" in history:
        ax1.plot(history["train_losses"], label="Train Loss")
        ax1.plot(history["val_losses"], label="Val Loss")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

    # Plot accuracy
    if "train_accs" in history and "val_accs" in history:
        ax2.plot(history["train_accs"], label="Train Acc")
        ax2.plot(history["val_accs"], label="Val Acc")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def save_class_mapping(class_to_idx: Dict[str, int], save_path: str) -> None:
    """Save class mapping to a JSON file.

    Args:
        class_to_idx: Mapping from class names to indices
        save_path: Path to save the mapping
    """
    import json

    with open(save_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)


def load_class_mapping(load_path: str) -> Dict[str, int]:
    """Load class mapping from a JSON file.

    Args:
        load_path: Path to load the mapping from

    Returns:
        Mapping from class names to indices
    """
    import json

    with open(load_path, "r") as f:
        return json.load(f)


def get_device() -> torch.device:
    """Get the best available device.

    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count the number of parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary containing parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
    }


def create_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """Create a directory for an experiment.

    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment

    Returns:
        Path to the experiment directory
    """
    experiment_dir = Path(base_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    (experiment_dir / "plots").mkdir(exist_ok=True)

    return experiment_dir
