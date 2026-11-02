"""Dataset for segmentation tasks."""

import logging
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import numpy as np
import torch

from src.common.progress_tools import iterate_with_progress
from src.core.dataset import BaseDataset
from src.segmentation.config import ClassConfig, ClassMappingConfig
from src.segmentation.mask_utils import (
    build_lookup_table,
    find_mask_file,
    prepare_masks,
)

logger = logging.getLogger(__name__)


@dataclass
class DataDistribution:
    class_name: str
    count: int
    percentage: float


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
        class_mapping (list[ClassMappingConfig]): Optional mapping to merge classes.
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
        class_mapping: list[ClassMappingConfig] | None = None,
    ):
        self.mask_extensions = mask_extensions
        self.class_config = self._create_class_config(class_config, class_mapping)
        self.lookup_table = build_lookup_table(self.rgb_to_class_idx)
        self._data_distribution: list[DataDistribution] | None = None

        super().__init__(
            data_root=data_root,
            transforms=transforms,
            extensions=extensions,
            max_samples=max_samples,
        )

    def _create_class_config(
        self,
        class_config: list[ClassConfig],
        class_mapping: list[ClassMappingConfig] | None = None,
    ) -> list[ClassConfig]:
        """Create a final class configuration after applying merge/remap rules.

        This method:
        1. Validates class_config (no duplicate names/colors)
        2. Validates class_mapping (source classes exist, no double-mapping, etc.)
        3. Builds final class list (removes merged sources, adds new targets)
        4. Sets up rgb_to_class_idx mapping (original RGB -> final class index)

        Returns:
            Final list of ClassConfig after applying mappings.
        """
        final_classes, rgb_to_class_idx, class_idx_to_rgb, class_idx_to_name = (
            _build_class_mappings(class_config, class_mapping)
        )

        self.rgb_to_class_idx = rgb_to_class_idx
        self.class_idx_to_rgb = class_idx_to_rgb
        self.class_idx_to_name = class_idx_to_name

        return final_classes

    def _prepare_mask_files(self, image_files: list[Path]) -> None:
        """Find corresponding mask files for each image, preprocess them and save to a better .npy format"""
        masks_dir = self.data_root / "masks"
        prepare_masks(
            image_files=image_files,
            masks_dir=masks_dir,
            mask_extensions=self.mask_extensions,
            lookup_table=self.lookup_table,
        )

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

        self._prepare_mask_files(image_files)

        for image_file in image_files:
            mask_file = find_mask_file(masks_dir, image_file, self.mask_extensions)
            if mask_file is not None:
                self.samples.append((image_file, mask_file))
                if self.max_samples and len(self.samples) >= self.max_samples:
                    break
            else:
                logger.warning(f"No mask found for image: {image_file}")

        if not self.samples:
            raise RuntimeError("No valid image-mask pairs found")

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
        mask = np.load(mask_path.with_suffix(".npy"))

        # Apply paired transforms
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
        return [self.class_idx_to_name[i] for i in range(len(self.class_config))]

    def get_data_distribution(self, use_cache: bool = True) -> list[DataDistribution]:
        """
        Get the class pixel distribution across the dataset.

        Returns:
            List[DataDistribution]: For each class, shows the absolute pixel count and
            the percentage of pixels (across all masks) belonging to that class.
        """
        if use_cache and self._data_distribution is not None:
            return self._data_distribution

        num_classes = len(self.get_class_names())
        pixel_counts = np.zeros(num_classes, dtype=np.int64)
        total_pixels = 0

        for _, mask_path in iterate_with_progress(
            self.samples,
            description="Data distribution",
            transient=True,
        ):
            mask = np.load(mask_path.with_suffix(".npy"))
            # Ignore labels (e.g. 255) are outside [0, num_classes) and must be
            # excluded so np.bincount length matches pixel_counts.
            labeled = mask.ravel()
            labeled = labeled[labeled < num_classes]
            total_pixels += int(labeled.size)
            counts = np.bincount(labeled, minlength=num_classes)
            pixel_counts += counts

        data_distribution = []
        for class_idx, class_name in enumerate(self.get_class_names()):
            count = int(pixel_counts[class_idx])
            percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
            data_distribution.append(
                DataDistribution(
                    class_name=class_name, count=count, percentage=percentage
                )
            )
        self._data_distribution = data_distribution
        return data_distribution


def _validate_base_classes(
    class_config: list[ClassConfig],
) -> dict[str, ClassConfig]:
    if not class_config:
        raise ValueError("class_config cannot be empty for segmentation.")

    base_by_name: dict[str, ClassConfig] = {}
    rgb_to_name: dict[tuple[int, int, int], str] = {}
    for c in class_config:
        if c.name in base_by_name:
            raise ValueError(f"Duplicate class name in class_config: '{c.name}'")
        if len(c.pixel_value) != 3:
            raise ValueError(
                f"class_config[{c.name}]: pixel_value must have 3 elements"
            )
        rgb = (c.pixel_value[0], c.pixel_value[1], c.pixel_value[2])
        if rgb in rgb_to_name:
            raise ValueError(
                f"Duplicate pixel_value {list(rgb)} in class_config: "
                f"'{rgb_to_name[rgb]}' and '{c.name}'"
            )

        base_by_name[c.name] = c
        rgb_to_name[rgb] = c.name

    return base_by_name


def _validate_class_mapping(
    class_mapping: list[ClassMappingConfig] | None,
    base_by_name: dict[str, ClassConfig],
) -> tuple[dict[str, str], dict[str, ClassMappingConfig]]:
    if not class_mapping:
        return {}, {}

    source_to_target: dict[str, str] = {}
    target_configs: dict[str, ClassMappingConfig] = {}

    for mapping in class_mapping:
        if not mapping.map_classes:
            raise ValueError(
                f"class_mapping[{mapping.name}]: map_classes cannot be empty"
            )
        if mapping.name in target_configs:
            raise ValueError(
                f"Duplicate target name in class_mapping: '{mapping.name}'"
            )
        if mapping.name not in base_by_name:
            if not mapping.pixel_value:
                raise ValueError(
                    f"class_mapping[{mapping.name}]: must provide pixel_value for new class"
                )
            if len(mapping.pixel_value) != 3:
                raise ValueError(
                    f"class_mapping[{mapping.name}]: pixel_value must have 3 elements"
                )

        target_configs[mapping.name] = mapping

        for src_name in mapping.map_classes:
            if src_name not in base_by_name:
                raise ValueError(
                    f"class_mapping[{mapping.name}]: source class '{src_name}' not found in class_config"
                )
            if src_name in source_to_target:
                raise ValueError(
                    f"Source class '{src_name}' mapped multiple times: "
                    f"to '{source_to_target[src_name]}' and to '{mapping.name}'"
                )
            source_to_target[src_name] = mapping.name

    for target_name in target_configs:
        if target_name in source_to_target:
            mapped_to = source_to_target[target_name]
            if mapped_to != target_name:
                raise ValueError(
                    f"class_mapping target '{target_name}' is also a source in "
                    f"map_classes mapped to '{mapped_to}'; a label cannot be a "
                    f"merge target in one group and a remapped source in another. "
                    f"Listing '{target_name}' in map_classes together with other "
                    f"aliases ({target_name} -> {target_name}) is allowed."
                )

    return source_to_target, target_configs


def _build_final_classes(
    class_config: list[ClassConfig],
    base_by_name: dict[str, ClassConfig],
    source_to_target: dict[str, str],
    target_configs: dict[str, ClassMappingConfig],
) -> list[ClassConfig]:
    mapped_sources = set(source_to_target.keys())
    final_classes = [c for c in class_config if c.name not in mapped_sources]
    final_names = {c.name for c in final_classes}

    for target_name, mapping in target_configs.items():
        if target_name in final_names:
            continue
        if mapping.pixel_value:
            pixel_value = mapping.pixel_value
        else:
            pixel_value = base_by_name[target_name].pixel_value
        final_classes.append(ClassConfig(name=target_name, pixel_value=pixel_value))
        final_names.add(target_name)

    rgb_to_name: dict[tuple[int, int, int], str] = {}
    for c in final_classes:
        rgb = (c.pixel_value[0], c.pixel_value[1], c.pixel_value[2])
        if rgb in rgb_to_name:
            raise ValueError(
                f"Final classes have duplicate pixel_value {list(rgb)}: "
                f"'{rgb_to_name[rgb]}' and '{c.name}'"
            )
        rgb_to_name[rgb] = c.name

    return final_classes


def _build_class_mappings(
    class_config: list[ClassConfig],
    class_mapping: list[ClassMappingConfig] | None,
) -> tuple[
    list[ClassConfig],
    dict[tuple[int, int, int], int],
    dict[int, tuple[int, int, int]],
    dict[int, str],
]:
    base_by_name = _validate_base_classes(class_config)
    source_to_target, target_configs = _validate_class_mapping(
        class_mapping, base_by_name
    )
    final_classes = _build_final_classes(
        class_config, base_by_name, source_to_target, target_configs
    )

    final_name_to_idx = {c.name: idx for idx, c in enumerate(final_classes)}
    rgb_to_class_idx: dict[tuple[int, int, int], int] = {}

    for c in class_config:
        rgb = (c.pixel_value[0], c.pixel_value[1], c.pixel_value[2])
        if c.name in source_to_target:
            target_name = source_to_target[c.name]
            rgb_to_class_idx[rgb] = final_name_to_idx[target_name]
        else:
            rgb_to_class_idx[rgb] = final_name_to_idx[c.name]

    class_idx_to_rgb = {
        idx: (c.pixel_value[0], c.pixel_value[1], c.pixel_value[2])
        for idx, c in enumerate(final_classes)
    }
    class_idx_to_name = {idx: c.name for idx, c in enumerate(final_classes)}

    return final_classes, rgb_to_class_idx, class_idx_to_rgb, class_idx_to_name
