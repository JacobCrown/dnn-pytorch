"""Dataset for segmentation tasks."""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from core.dataset import BaseDataset
from src.common.enums import TaskType


class SegmentationDataset(BaseDataset):
    """A PyTorch Dataset for image segmentation tasks.

    This dataset expects a directory structure where images and masks are paired:
    data_root/
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── masks/
    │   ├── image1.png
    │   ├── image2.png
    │   └── ...
    └── ...

    Args:
        data_root (str): Root directory containing images and masks
        transforms (Compose): Transforms to apply to images and masks
        extensions (List[str]): List of image file extensions to include
        mask_extensions (List[str]): List of mask file extensions to include
        max_samples (Optional[int]): Maximum number of samples to load
        normalize_masks (bool): Whether to normalize mask values to [0, 1]
        binary_masks (bool): Whether masks are binary or multi-class
    """

    def __init__(
        self,
        data_root: str,
        transforms: Any,
        extensions: Optional[List[str]] = None,
        mask_extensions: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        normalize_masks: bool = True,
        binary_masks: bool = True,
    ):
        self.mask_extensions = mask_extensions or [".png", ".jpg", ".jpeg"]
        self.normalize_masks = normalize_masks
        self.binary_masks = binary_masks

        super().__init__(
            data_root=data_root,
            transforms=transforms,
            extensions=extensions,
            max_samples=max_samples,
            task_type=TaskType.SEGMENTATION,
        )

    def _build_samples(self) -> None:
        """Build the list of samples for segmentation."""
        images_dir = self.data_root / "images"
        masks_dir = self.data_root / "masks"

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

        # Get all image files
        image_files = []
        for ext in self.extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))

        # Build pairs of images and corresponding masks
        for image_file in image_files:
            # Try to find corresponding mask
            mask_file = self._find_mask_file(masks_dir, image_file)
            if mask_file is not None:
                self.samples.append((str(image_file), str(mask_file)))
                if self.max_samples and len(self.samples) >= self.max_samples:
                    break

        if not self.samples:
            raise RuntimeError("No valid image-mask pairs found")

    def _find_mask_file(self, masks_dir: Path, image_file: Path) -> Optional[Path]:
        """Find the corresponding mask file for an image file.

        Args:
            masks_dir: Directory containing mask files
            image_file: Path to the image file

        Returns:
            Path to the corresponding mask file, or None if not found
        """
        # Try different mask extensions
        for ext in self.mask_extensions:
            # Try exact name match with different extension
            mask_name = image_file.stem + ext
            mask_file = masks_dir / mask_name
            if mask_file.exists():
                return mask_file

            # Try with uppercase extension
            mask_name = image_file.stem + ext.upper()
            mask_file = masks_dir / mask_name
            if mask_file.exists():
                return mask_file

        return None

    def _load_mask(self, mask_path: Union[str, Path]) -> torch.Tensor:
        """Load and preprocess a segmentation mask.

        Args:
            mask_path: Path to the mask file

        Returns:
            Preprocessed mask tensor
        """
        try:
            # Read mask with OpenCV (grayscale)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Failed to read mask: {mask_path}")

            # Normalize to [0, 1] if requested
            if self.normalize_masks:
                if mask.max() > 1:
                    mask = mask.astype(np.float32) / 255.0
                else:
                    mask = mask.astype(np.float32)

            # Convert to tensor
            mask = torch.from_numpy(mask).float()

            # Add channel dimension if needed
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)  # (H, W) -> (1, H, W)

            return mask

        except Exception as e:
            raise RuntimeError(f"Error loading mask {mask_path}: {e}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (image, mask) pair
        """
        image_path, mask_path = self.samples[idx]

        # Load image and mask
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        # Ensure image and mask have the same spatial dimensions
        if image.shape[-2:] != mask.shape[-2:]:
            # Resize mask to match image dimensions
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0), size=image.shape[-2:], mode="nearest"
            ).squeeze(0)

        return image, mask

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset."""
        info = super().get_dataset_info()
        info.update(
            {
                "mask_extensions": self.mask_extensions,
                "normalize_masks": self.normalize_masks,
                "binary_masks": self.binary_masks,
            }
        )
        return info
