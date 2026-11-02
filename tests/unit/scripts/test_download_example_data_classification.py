"""Tests for exporting CIFAR-10 into the classification folder layout."""

from pathlib import Path

from PIL import Image
from torchvision.datasets import CIFAR10

from scripts.download_example_data_classification import (
    _MirroredCIFAR10,
    export_split,
    split_is_ready,
)


def _image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (32, 32), color)


def test_cifar10_download_uses_mirror_of_the_official_archive() -> None:
    assert _MirroredCIFAR10.url != CIFAR10.url
    assert _MirroredCIFAR10.tgz_md5 == CIFAR10.tgz_md5


def test_export_split_writes_one_folder_per_class(tmp_path: Path) -> None:
    samples = [
        (_image((255, 0, 0)), 0),
        (_image((0, 255, 0)), 1),
        (_image((0, 0, 255)), 0),
    ]

    written = export_split(samples, ["cat", "dog"], tmp_path)

    assert written == 3
    assert sorted(path.name for path in tmp_path.iterdir()) == ["cat", "dog"]
    assert (tmp_path / "cat" / "00000.png").is_file()
    assert (tmp_path / "dog" / "00001.png").is_file()
    assert (tmp_path / "cat" / "00002.png").is_file()
    with Image.open(tmp_path / "dog" / "00001.png") as image:
        assert image.size == (32, 32)


def test_split_is_ready_requires_expected_count_in_every_class(tmp_path: Path) -> None:
    class_names = ["cat", "dog"]

    assert not split_is_ready(tmp_path, class_names, image_count=2)

    export_split(
        [(_image((1, 2, 3)), 0), (_image((4, 5, 6)), 1)],
        class_names,
        tmp_path,
    )

    assert split_is_ready(tmp_path, class_names, image_count=2)
    assert not split_is_ready(tmp_path, class_names, image_count=4)
