from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import torch

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
        max_samples (Optional[int]): Maximum number of samples to load per class
    """

    def __init__(
        self,
        data_root: Path,
        transforms: A.Compose,
        extensions: list[str],
        max_samples: int | None = None,
    ):
        super().__init__(
            data_root=data_root,
            transforms=transforms,
            extensions=extensions,
            max_samples=max_samples,
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
        self.idx_to_class = {
            class_name: idx for idx, class_name in self.class_to_idx.items()
        }

        # Build file paths and labels
        for class_name in self.class_names:
            class_samples = 0
            class_dir = self.data_root / class_name
            class_idx = self.class_to_idx[class_name]

            for file_path in class_dir.iterdir():
                if file_path.suffix.lower() in self.extensions:
                    self.samples.append((str(file_path), class_idx))
                    class_samples += 1
                    if self.max_samples and class_samples >= self.max_samples:
                        break

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
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

    def get_class_names(self) -> list[str]:
        """Get the list of class names."""
        return self.class_names

    def get_class_to_idx(self) -> dict[str, int]:
        """Get the mapping from class names to indices."""
        return self.class_to_idx

    def get_idx_to_class(self) -> dict[int, str]:
        """Get the mapping from indices to class names."""
        return self.idx_to_class

    def get_data_distribution(self) -> list[DataDistribution]:
        """Get the distribution of the data."""
        data_distribution = [
            DataDistribution(class_name=class_name, count=0, percentage=0)
            for class_name in self.class_names
        ]
        for _, label in self.samples:
            data_distribution[label].count += 1
        total_samples = len(self.samples)
        for data_dist in data_distribution:
            data_dist.percentage = (data_dist.count / total_samples) * 100
        return data_distribution
