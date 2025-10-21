from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import albumentations as A

from src.common.enums import TaskType
from src.core.dataset import BaseDataset


@dataclass
class DataDistribution:
    class_name: str
    count: int
    percentage: float

class ClassificationDataset(BaseDataset):
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
        max_samples (Optional[int]): Maximum number of samples to load
    """

    def __init__(
        self,
        data_root: Path,
        transforms: A.Compose,
        extensions: List[str],
        max_samples: Optional[int] = None,
    ):
        super().__init__(
            data_root=data_root,
            transforms=transforms,
            extensions=extensions,
            max_samples=max_samples,
            task_type=TaskType.CLASSIFICATION,
        )

    def _build_samples(self) -> None:
        """Build the list of samples for classification."""
        # Get class names (subdirectory names)
        self.class_names = sorted(
            [d.name for d in self.data_root.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.class_names)
        }

        # Build file paths and labels
        for class_name in self.class_names:
            class_dir = self.data_root / class_name
            class_idx = self.class_to_idx[class_name]

            for file_path in class_dir.iterdir():
                if file_path.suffix.lower() in self.extensions:
                    self.samples.append((str(file_path), class_idx))
                    if self.max_samples and len(self.samples) >= self.max_samples:
                        break

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            Tuple[torch.Tensor, int]: (image, label) pair
        """
        file_path, label = self.samples[idx]

        # Use the base class image loading method
        image = self._load_image(file_path)
        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image_t: torch.Tensor = transformed["image"]
        return image_t, label

    def get_class_names(self) -> List[str]:
        """Get the list of class names."""
        return self.class_names

    def get_class_to_idx(self) -> Dict[str, int]:
        """Get the mapping from class names to indices."""
        return self.class_to_idx

    def get_data_distribution(self) -> List[DataDistribution]:
        """Get the distribution of the data."""
        data_distribution = [DataDistribution(class_name=class_name, count=0, percentage=0) for class_name in self.class_names]
        for _, label in self.samples:
            data_distribution[label].count += 1
        total_samples = len(self.samples)
        for data_dist in data_distribution:
            data_dist.percentage = (data_dist.count / total_samples) * 100
        return data_distribution