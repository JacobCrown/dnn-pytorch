"""Base dataset class for different task types."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import numpy as np

from src.common.enums import TaskType


class BaseDataset(Dataset):
    """Base PyTorch Dataset class that can be extended for different task types.

    This class provides common functionality for image loading and preprocessing
    that can be shared across classification, segmentation, and other tasks.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        transforms: Any,
        extensions: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        task_type: TaskType = TaskType.UNDEFINED,
    ):
        """Initialize the base dataset.

        Args:
            data_root: Root directory containing the data
            transforms: Transforms to apply to images
            extensions: List of image file extensions to include
            max_samples: Maximum number of samples to load
            task_type: Type of task this dataset is for
        """
        self.data_root = Path(data_root)
        self.transforms = transforms
        self.extensions = extensions or [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        self.max_samples = max_samples
        self.task_type = task_type

        # Validate data root exists
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Data root directory does not exist: {self.data_root}"
            )

        # Build file paths and metadata
        self.samples = []
        self._build_samples()

    def _build_samples(self) -> None:
        """Build the list of samples. Override in subclasses for specific logic."""
        raise NotImplementedError("Subclasses must implement _build_samples")

    def _load_image(self, file_path: Union[str, Path]) -> torch.Tensor:
        """Load and preprocess an image.

        Args:
            file_path: Path to the image file

        Returns:
            Preprocessed image tensor
        """
        try:
            # Read image with OpenCV and convert BGR to RGB
            image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise RuntimeError(f"Failed to read image: {file_path}")
            if image.ndim == 2: # Grayscale image, needs expansion to 3 channels
                image = np.expand_dims(image, axis=-1)
            elif image.ndim == 3: # RGB image
                if image.shape[-1] == 3: # RGB image
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif image.shape[-1] == 4: # RGBA image
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                else:
                    raise ValueError(f"Invalid image shape: {image.shape}")

            # Apply transforms (expects HWC numpy)
            if self.transforms:
                transformed = self.transforms(image=image)
                # Handle both dict and direct tensor outputs
                if isinstance(transformed, dict):
                    image = transformed["image"]
                else:
                    image = transformed

            # Ensure output is a torch.Tensor
            if not isinstance(image, torch.Tensor):
                image = ToTensorV2()(image=image)["image"]

            return image

        except Exception as e:
            raise RuntimeError(f"Error loading image {file_path}: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Any:
        """Get a sample from the dataset. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement __getitem__")
