"""Download CIFAR-10 and export it as class folders for classification training.

CIFAR-10 has a train split and a test split, and no validation split::

    data/classification/cifar-10/train/<class>/*.png
    data/classification/cifar-10/test/<class>/*.png

The default training config reads validation from ``test``::

    uv run python -m src.classification.main data.root=data/classification/cifar-10
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from PIL import Image
from torchvision.datasets import CIFAR10
from tqdm import tqdm

# Official host (cs.toronto.edu) serves this 163 MB archive at ~50 KB/s.
CIFAR10_URL = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "data" / "classification" / "cifar-10"
DOWNLOAD_ROOT = REPO_ROOT / "data" / ".cache" / "cifar-10"
SPLITS = (("train", True), ("test", False))


class _MirroredCIFAR10(CIFAR10):
    url = CIFAR10_URL


def export_split(
    samples: Iterable[tuple[Image.Image, int]],
    class_names: Sequence[str],
    destination: Path,
) -> int:
    """Save samples as ``destination/<class>/<index>.png``."""
    for class_name in class_names:
        (destination / class_name).mkdir(parents=True, exist_ok=True)

    written = 0
    for index, (image, label) in enumerate(samples):
        image.save(destination / class_names[label] / f"{index:05d}.png")
        written += 1
    return written


def split_is_ready(
    destination: Path, class_names: Sequence[str], image_count: int
) -> bool:
    """Return whether each class folder holds an equal share of ``image_count``."""
    if not class_names or image_count % len(class_names):
        return False
    per_class = image_count // len(class_names)
    return all(_png_count(destination / name) == per_class for name in class_names)


def export_cifar10(
    dataset_root: Path = DATASET_ROOT,
    download_root: Path = DOWNLOAD_ROOT,
) -> Path:
    """Download CIFAR-10 and export ``train`` and ``test`` class folders."""
    for split_name, train in SPLITS:
        dataset = _MirroredCIFAR10(root=download_root, train=train, download=True)
        destination = dataset_root / split_name
        class_names = list(dataset.classes)
        image_count = len(dataset)
        if split_is_ready(destination, class_names, image_count):
            print(f"{split_name}: already exported ({image_count} images)")
            continue

        written = export_split(
            _iter_samples(dataset, split_name),
            class_names,
            destination,
        )
        if written != image_count:
            raise RuntimeError(
                f"{split_name}: expected {image_count} images, wrote {written}"
            )
        print(f"{split_name}: wrote {written} images to {destination}")
    return dataset_root


def _iter_samples(
    dataset: CIFAR10, split_name: str
) -> Iterable[tuple[Image.Image, int]]:
    for index in tqdm(range(len(dataset)), desc=f"Exporting {split_name}"):
        yield dataset[index]


def _png_count(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(
        1
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )


def main() -> None:
    root = export_cifar10()
    print(f"CIFAR-10 ready at {root}")


if __name__ == "__main__":
    main()
