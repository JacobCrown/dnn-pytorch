import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import torch

from src.classification.config import ClassificationClassMappingConfig
from src.core.dataset import BaseDataset

logger = logging.getLogger(__name__)


@dataclass
class DataDistribution:
    class_name: str
    count: int
    percentage: float


@dataclass(frozen=True)
class MixedTrainClassCount:
    class_name: str
    real_count: int
    synthetic_count: int


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
        class_mapping: list[ClassificationClassMappingConfig] | None = None,
        canonical_class_to_idx: dict[str, int] | None = None,
    ):
        self.class_mapping = class_mapping
        self.canonical_class_to_idx = canonical_class_to_idx
        self.mixed_train_class_counts: list[MixedTrainClassCount] | None = None
        super().__init__(
            data_root=data_root,
            transforms=transforms,
            extensions=extensions,
            max_samples=max_samples,
        )

    def _build_samples(self) -> None:
        """Build the list of samples for classification."""
        source_class_names = sorted(
            [d.name for d in self.data_root.iterdir() if d.is_dir()]
        )
        if self.canonical_class_to_idx is None:
            self._build_samples_strict(source_class_names)
        else:
            self._build_samples_relaxed(source_class_names)

    def _build_samples_strict(self, source_class_names: list[str]) -> None:
        """Build samples from on-disk classes (typical for training; strict mapping)."""
        self.class_names, source_to_target = _build_final_classes(
            source_class_names, self.class_mapping
        )
        self._apply_class_indices()
        for source_class_name in source_class_names:
            target_class_name = source_to_target.get(
                source_class_name, source_class_name
            )
            class_idx = self.class_to_idx[target_class_name]
            self._append_class_samples(
                class_dir_name=source_class_name,
                class_idx=class_idx,
            )

    def _build_samples_relaxed(
        self,
        source_class_names: list[str],
    ) -> None:
        """Build samples for val/test using the same label space as training.

        Folders in ``class_mapping`` that are missing in this split are ignored.
        Folders on disk that are not part of the training label space are skipped
        with a warning.
        """
        assert self.canonical_class_to_idx is not None
        self.class_names = _class_names_in_index_order(self.canonical_class_to_idx)
        self.class_to_idx = dict(self.canonical_class_to_idx)
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        source_set = set(source_class_names)
        source_to_target = _source_to_target_for_split(source_set, self.class_mapping)
        for source_class_name in source_class_names:
            target_name = source_to_target.get(source_class_name, source_class_name)
            if target_name not in self.class_to_idx:
                logger.warning(
                    "Skipping %s path: folder '%s' resolves to unknown class label '%s' "
                    "(not in training label space)",
                    self.data_root,
                    source_class_name,
                    target_name,
                )
                continue
            self._append_class_samples(
                class_dir_name=source_class_name,
                class_idx=self.class_to_idx[target_name],
            )

    def _apply_class_indices(self) -> None:
        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.class_names)
        }
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

    def _append_class_samples(self, class_dir_name: str, class_idx: int) -> None:
        class_samples = 0
        class_dir = self.data_root / class_dir_name
        for file_path in sorted(class_dir.iterdir()):
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

        label_counts = Counter(label for _, label in self.samples)
        total_samples = len(self.samples)
        return [
            DataDistribution(
                class_name=class_name,
                count=label_counts.get(idx, 0),
                percentage=(label_counts.get(idx, 0) / total_samples * 100)
                if total_samples > 0
                else 0,
            )
            for idx, class_name in enumerate(self.class_names)
        ]


def _class_names_in_index_order(class_to_idx: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]


def _source_to_target_for_split(
    source_names_present: set[str],
    class_mapping: list[ClassificationClassMappingConfig] | None,
) -> dict[str, str]:
    """Apply ``class_mapping`` only for source folders that exist in this split."""
    if not class_mapping:
        return {}

    result: dict[str, str] = {}
    for mapping in class_mapping:
        if not mapping.map_classes:
            raise ValueError(
                f"class_mapping[{mapping.name}]: map_classes cannot be empty"
            )
        for source_name in mapping.map_classes:
            if source_name not in source_names_present:
                logger.debug(
                    "No folder %r for merge into %r in this split; skipping that source",
                    source_name,
                    mapping.name,
                )
                continue
            if source_name in result:
                raise ValueError(
                    f"Source class '{source_name}' mapped multiple times: "
                    f"to '{result[source_name]}' and to '{mapping.name}'"
                )
            result[source_name] = mapping.name
    return result


def _validate_class_mapping(
    source_class_names: list[str],
    class_mapping: list[ClassificationClassMappingConfig] | None,
) -> tuple[dict[str, str], list[str]]:
    if not class_mapping:
        return {}, []

    source_class_set = set(source_class_names)
    source_to_target: dict[str, str] = {}
    target_names: list[str] = []

    for mapping in class_mapping:
        if not mapping.map_classes:
            raise ValueError(
                f"class_mapping[{mapping.name}]: map_classes cannot be empty"
            )
        if mapping.name in target_names:
            raise ValueError(
                f"Duplicate target name in class_mapping: '{mapping.name}'"
            )
        target_names.append(mapping.name)

        for source_name in mapping.map_classes:
            if source_name not in source_class_set:
                raise ValueError(
                    f"class_mapping[{mapping.name}]: source class '{source_name}' not found"
                )
            if source_name in source_to_target:
                raise ValueError(
                    f"Source class '{source_name}' mapped multiple times: "
                    f"to '{source_to_target[source_name]}' and to '{mapping.name}'"
                )
            source_to_target[source_name] = mapping.name

    for target_name in target_names:
        if target_name in source_to_target:
            mapped_to = source_to_target[target_name]
            if mapped_to != target_name:
                raise ValueError(
                    f"class_mapping target '{target_name}' is also a source folder "
                    f"mapped to '{mapped_to}'; a label cannot be a merge target in one "
                    f"group and a remapped source folder in another. "
                    f"Listing '{target_name}' in map_classes together with other aliases "
                    f"({target_name} -> {target_name}) is allowed."
                )

    return source_to_target, target_names


def _build_final_classes(
    source_class_names: list[str],
    class_mapping: list[ClassificationClassMappingConfig] | None,
) -> tuple[list[str], dict[str, str]]:
    source_to_target, target_names = _validate_class_mapping(
        source_class_names, class_mapping
    )
    mapped_sources = set(source_to_target)

    final_class_names = [
        class_name
        for class_name in source_class_names
        if class_name not in mapped_sources
    ]
    final_class_names.extend(
        target_name
        for target_name in target_names
        if target_name not in final_class_names
    )

    return final_class_names, source_to_target


def _image_files_for_class(
    root: Path,
    source_class_name: str,
    extensions: list[str],
) -> list[Path]:
    class_dir = root / source_class_name
    if not class_dir.exists():
        return []
    return [
        file_path
        for file_path in sorted(class_dir.iterdir())
        if file_path.suffix.lower() in extensions
    ]


def build_mixed_train_dataset(
    *,
    real_root: Path,
    synthetic_root: Path,
    transforms: A.Compose,
    extensions: list[str],
    max_samples: int | None,
    class_mapping: list[ClassificationClassMappingConfig] | None,
    seed: int,
) -> ClassificationDataset:
    """Build a real-first classification dataset with synthetic fill samples."""
    real_source_names = sorted(d.name for d in real_root.iterdir() if d.is_dir())
    synthetic_source_names = sorted(d.name for d in synthetic_root.iterdir() if d.is_dir())
    source_class_names = sorted(set(real_source_names) | set(synthetic_source_names))
    class_names, source_to_target = _build_final_classes(source_class_names, class_mapping)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    rng = random.Random(seed)

    dataset = object.__new__(ClassificationDataset)
    dataset.data_root = synthetic_root
    dataset.transforms = transforms
    dataset.extensions = extensions
    dataset.max_samples = max_samples
    dataset.class_mapping = class_mapping
    dataset.canonical_class_to_idx = None
    dataset.samples = []
    dataset.class_names = class_names
    dataset.class_to_idx = class_to_idx
    dataset.idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    dataset.mixed_train_class_counts = []

    for class_name in class_names:
        source_names = [
            source_name
            for source_name in source_class_names
            if source_to_target.get(source_name, source_name) == class_name
        ]
        real_files: list[Path] = []
        synthetic_files: list[Path] = []
        for source_name in source_names:
            real_files.extend(_image_files_for_class(real_root, source_name, extensions))
            synthetic_files.extend(
                _image_files_for_class(synthetic_root, source_name, extensions)
            )

        synthetic_budget = (
            len(synthetic_files)
            if max_samples is None
            else max(0, max_samples - len(real_files))
        )
        selected_synthetic = (
            rng.sample(synthetic_files, synthetic_budget)
            if synthetic_budget < len(synthetic_files)
            else list(synthetic_files)
        )
        class_idx = class_to_idx[class_name]
        dataset.samples.extend((str(path), class_idx) for path in real_files)
        dataset.samples.extend((str(path), class_idx) for path in selected_synthetic)
        dataset.mixed_train_class_counts.append(
            MixedTrainClassCount(
                class_name=class_name,
                real_count=len(real_files),
                synthetic_count=len(selected_synthetic),
            )
        )

    return dataset
