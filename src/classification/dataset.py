import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import torch
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


class ClassificationDataset(Dataset):
    """A PyTorch Dataset for image classification tasks.

    This dataset expects a directory structure where each subdirectory represents a class:
    data_root/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2/
    │   ├── image3.jpg
    │   └── ...
    └── ...

    Args:
        data_root (str): Root directory containing class-named subdirectories
        transforms (Compose): Torchvision transforms to apply to images
        extensions (List[str]): List of image file extensions to include
    """

    def __init__(
        self,
        data_root: str,
        transforms: Any,
        extensions: Optional[List[str]] = None,
    ):
        self.data_root = Path(data_root)
        self.transforms = transforms
        self.extensions = extensions or [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

        # Get class names (subdirectory names)
        self.class_names = sorted(
            [d.name for d in self.data_root.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.class_names)
        }

        # Build file paths and labels
        self.samples = []
        for class_name in self.class_names:
            class_dir = self.data_root / class_name
            class_idx = self.class_to_idx[class_name]

            for file_path in class_dir.iterdir():
                if file_path.suffix.lower() in self.extensions:
                    self.samples.append((str(file_path), class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            Tuple[torch.Tensor, int]: (image, label) pair
        """
        file_path, label = self.samples[idx]

        try:
            # Read image with OpenCV and convert BGR to RGB
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read image: {file_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply Albumentations transforms (expects HWC numpy)
            if self.transforms:
                image = self.transforms(image=image)["image"]

            # Ensure output is a torch.Tensor
            if not isinstance(image, torch.Tensor):
                image = ToTensorV2()(image=image)["image"]

            return image, label

        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            # Return a placeholder or raise the error
            raise

    def get_class_names(self) -> List[str]:
        """Get the list of class names."""
        return self.class_names

    def get_class_to_idx(self) -> Dict[str, int]:
        """Get the mapping from class names to indices."""
        return self.class_to_idx.copy()
