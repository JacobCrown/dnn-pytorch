"""Base dataset class for different task types."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Base PyTorch Dataset class that can be extended for different task types.

    This class provides common functionality for image loading and preprocessing
    that can be shared across classification, segmentation, and other tasks.
    """

    def __init__(
        self,
        data_root: str | Path,
        transforms: A.Compose,
        extensions: list[str] | None = None,
        max_samples: int | None = None,
    ):
        """Initialize the base dataset.

        Args:
            data_root: Root directory containing the data
            transforms: Albumentations transforms to apply to images
            extensions: List of image file extensions to include
            max_samples: Maximum number of samples to load per class
        """
        self.data_root = Path(data_root)
        self.transforms = transforms
        self.extensions = extensions or [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        self.max_samples = max_samples

        # Validate data root exists
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Data root directory does not exist: {self.data_root}"
            )

        # Build file paths and metadata
        self.samples = []
        self._build_samples()

    @staticmethod
    def _load_image(file_path: str | Path) -> np.ndarray:
        """Load and preprocess an image.

        Args:
            file_path: Path to the image file

        Returns:
            Image as numpy array
        """
        try:
            # Read image with OpenCV and convert BGR to RGB
            image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise RuntimeError(f"Failed to read image: {file_path}")
            if image.ndim == 2:  # Grayscale image, needs expansion to 3 channels
                image = np.expand_dims(image, axis=-1)
            elif image.ndim == 3:  # RGB image
                if image.shape[-1] == 3:  # RGB image
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif image.shape[-1] == 4:  # RGBA image
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                else:
                    raise ValueError(f"Invalid image shape: {image.shape}")

            return image

        except Exception as e:
            raise RuntimeError(f"Error loading image {file_path}: {e}") from e

    def __len__(self) -> int:
        return len(self.samples)

    # ================================ Abstract Methods ================================

    @abstractmethod
    def _build_samples(self) -> None: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Any: ...
