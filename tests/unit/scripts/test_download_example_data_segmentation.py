"""Tests for exporting CamVid into the segmentation folder layout."""

from pathlib import Path

from PIL import Image

from scripts.download_example_data_segmentation import (
    camvid_pairs,
    export_split,
    split_is_ready,
)


def _png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color).save(path)


def test_export_split_pairs_image_with_32_class_label(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _png(source / "CamVid_RGB" / "0001TP_006690.png", (10, 20, 30))
    _png(source / "CamVid_Label" / "0001TP_006690_L.png", (128, 64, 128))
    split_file = source / "camvid_train.txt"
    split_file.write_text(
        "CamVid_RGB/0001TP_006690.png CamVidGray/0001TP_006690_L.png\n",
        encoding="utf-8",
    )

    written = export_split(camvid_pairs(source, split_file), tmp_path / "train")

    image = tmp_path / "train" / "images" / "0001TP_006690.png"
    mask = tmp_path / "train" / "masks" / "0001TP_006690.png"
    assert written == 1
    assert image.is_file()
    assert mask.is_file()
    with Image.open(mask) as opened:
        assert opened.getpixel((0, 0)) == (128, 64, 128)


def test_export_split_removes_files_outside_the_split(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _png(source / "CamVid_RGB" / "frame.png", (1, 2, 3))
    _png(source / "CamVid_Label" / "frame_L.png", (128, 64, 128))
    split_file = source / "camvid_train.txt"
    split_file.write_text(
        "CamVid_RGB/frame.png CamVidGray/frame_L.png\n",
        encoding="utf-8",
    )
    destination = tmp_path / "train"
    _png(destination / "images" / "stale.png", (9, 9, 9))
    _png(destination / "masks" / "stale.png", (9, 9, 9))
    (destination / "masks" / "stale.npy").write_bytes(b"old")
    (destination / "masks" / "frame.npy").write_bytes(b"old")

    export_split(camvid_pairs(source, split_file), destination)

    assert not (destination / "images" / "stale.png").exists()
    assert not (destination / "masks" / "stale.png").exists()
    assert not (destination / "masks" / "stale.npy").exists()
    assert not (destination / "masks" / "frame.npy").exists()
    assert split_is_ready(destination, image_count=1)


def test_split_is_ready_requires_a_mask_for_every_image(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _png(source / "CamVid_RGB" / "frame.png", (1, 2, 3))
    _png(source / "CamVid_Label" / "frame_L.png", (0, 0, 0))
    split_file = source / "camvid_train.txt"
    split_file.write_text(
        "CamVid_RGB/frame.png CamVidGray/frame_L.png\n",
        encoding="utf-8",
    )
    destination = tmp_path / "train"

    assert not split_is_ready(destination, image_count=1)

    export_split(camvid_pairs(source, split_file), destination)

    assert split_is_ready(destination, image_count=1)
    assert not split_is_ready(destination, image_count=2)
