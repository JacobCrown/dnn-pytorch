"""Dataset for segmentation tasks."""

from pathlib import Path
import logging

import albumentations as A
import torch
import numpy as np

from src.common.enums import TaskType
from src.core.dataset import BaseDataset
from src.segmentation.config import ClassConfig


logger = logging.getLogger(__name__)

class SegmentationDataset(BaseDataset):
    """A PyTorch Dataset for image segmentation tasks.
    
    This dataset expects a directory structure where images and masks are paired:
    data_root/
    ├── train/
    │   ├── images/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── masks/
    │   │   ├── mask1.png
    │   │   ├── mask2.png
    │   │   └── ...
    └── val/
    │   ├── images/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── masks/
    │   │   ├── mask1.png
    │   │   ├── mask2.png
    │   │   └── ...
    ...

    Args:
        data_root (Path): Root directory containing images and masks
        transforms (Compose): Transforms to apply to images and masks
        class_config (list[ClassConfig]): A list of class configurations.
        extensions (list[str]): List of image file extensions to include
        mask_extensions (list[str]): List of mask file extensions to include
        max_samples (int | None): Maximum number of samples to load
    """

    def __init__(
        self,
        data_root: Path,
        transforms: A.Compose,
        class_config: list[ClassConfig],
        extensions: list[str],
        mask_extensions: list[str],
        max_samples: int | None = None,
    ):
        self.mask_extensions = mask_extensions
        self.class_config = class_config
        self.rgb_to_idx = {tuple(c.pixel_value): idx for idx, c in enumerate(class_config)}
        self.idx_to_name = {idx: c.name for idx, c in enumerate(class_config)}
        self.lookup_table = self._create_lookup_table()

        super().__init__(
            data_root=data_root,
            transforms=transforms,
            extensions=extensions,
            max_samples=max_samples,
            task_type=TaskType.SEGMENTATION,
        )

    def _create_lookup_table(self) -> np.ndarray:
        """Create a lookup table for efficient mask conversion."""
        max_id = 0
        for color in self.rgb_to_idx.keys():
            color_id = color[0] * 256**2 + color[1] * 256 + color[2]
            if color_id > max_id:
                max_id = color_id

        lut = np.full(max_id + 1, 255, dtype=np.int64)

        for color, class_id in self.rgb_to_idx.items():
            color_id = color[0] * 256**2 + color[1] * 256 + color[2]
            lut[color_id] = class_id
            
        return lut

    def _convert_pixel_mask_to_class_mask(self, mask: np.ndarray) -> np.ndarray:
        """Converts an RGB mask to a class index mask using a lookup table.

        Args:
            mask (np.ndarray): The RGB mask as a NumPy array (H, W, 3).

        Returns:
            np.ndarray: A class index mask (H, W).
        """
        mask_int32 = mask.astype(np.int32)
        integer_mask = mask_int32.dot(np.array([256**2, 256, 1], dtype=np.int32))
        class_mask = self.lookup_table[integer_mask]

        return class_mask.astype(np.int64)

    def _build_samples(self) -> None:
        """Build the list of samples for segmentation."""
        images_dir = self.data_root / "images"
        masks_dir = self.data_root / "masks"

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

        image_files = []
        for ext in self.extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))

        for image_file in image_files:
            mask_file = self._find_mask_file(masks_dir, image_file)
            if mask_file is not None:
                self.samples.append((str(image_file), str(mask_file)))
                if self.max_samples and len(self.samples) >= self.max_samples:
                    break
            else:
                logger.warning(f"No mask found for image: {image_file}")

        if not self.samples:
            raise RuntimeError("No valid image-mask pairs found")

    def _find_mask_file(self, masks_dir: Path, image_file: Path) -> Path | None:
        """Find the corresponding mask file for an image file."""
        for mask_ext in self.mask_extensions:
            mask_file = masks_dir / f"{image_file.stem}{mask_ext}"
            if mask_file.exists():
                return mask_file
        
        logger.warning(f"No mask found for image with exact name match: {image_file.stem}")
        return None

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the transformed image
            (C, H, W) and mask (H, W).
        """
        image_path, mask_path = self.samples[idx]

        image = self._load_image(image_path)
        mask_rgb = self._load_image(mask_path)
        
        # Convert RGB mask to class indices
        mask = self._convert_pixel_mask_to_class_mask(mask_rgb)

        # Apply paired transforms
        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image_a: torch.Tensor = transformed["image"]
            mask_a: torch.Tensor = transformed["mask"]

        # The mask should be a LongTensor with shape (H, W) for CrossEntropyLoss
        if mask_a.ndim == 3 and mask_a.shape[0] == 1:
            mask_a = mask_a.squeeze(0)

        return image_a.float(), mask_a.long()

    def get_class_names(self) -> list[str]:
        """Get list of class names in order.

        Returns:
            List of class names
        """
        return [self.idx_to_name[i] for i in range(len(self.class_config))]