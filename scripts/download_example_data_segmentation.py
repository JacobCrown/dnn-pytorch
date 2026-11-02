"""Download CamVid and lay it out for segmentation training.

The archive is the original 32-class RGB labels with the SegNet train/val/test
split. Masks are renamed to the image stem so each pair matches by filename::

    data/segmentation/CamVid/train/images/*.png
    data/segmentation/CamVid/train/masks/*.png
    data/segmentation/CamVid/val/images/*.png
    data/segmentation/CamVid/val/masks/*.png
    data/segmentation/CamVid/test/images/*.png
    data/segmentation/CamVid/test/masks/*.png

"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from tqdm import tqdm

# Pinned mirror. The Cambridge host for the original zips does not respond.
CAMVID_COMMIT = "79ae0071a75d90331a62b5a2aaf6d53cc69fd9f6"
CAMVID_URL = f"https://github.com/lih627/CamVid/archive/{CAMVID_COMMIT}.zip"
_LABEL_DIR = "CamVid_Label"
_IMAGE_DIR = "CamVid_RGB"
_SPLIT_FILES = {
    "train": "camvid_train.txt",
    "val": "camvid_val.txt",
    "test": "camvid_test.txt",
}
# SegNet split of the 701 labeled frames.
SPLIT_IMAGE_COUNTS = {"train": 367, "val": 101, "test": 233}

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "data" / "segmentation" / "CamVid"
DOWNLOAD_ROOT = REPO_ROOT / "data" / ".cache" / "camvid"


def camvid_pairs(source_root: Path, split_file: Path) -> list[tuple[Path, Path]]:
    """Pair each split image with its 32-class RGB label.

    Split files point at the 11-class ``CamVidGray`` masks. Training here uses
    the original labels in ``CamVid_Label``, which share the ``<stem>_L.png`` name.
    """
    pairs: list[tuple[Path, Path]] = []
    for line_number, line in enumerate(
        split_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        image_rel, _label_rel = line.split()
        image_path = source_root / image_rel
        mask_path = source_root / _LABEL_DIR / f"{Path(image_rel).stem}_L.png"
        if not image_path.is_file():
            raise FileNotFoundError(
                f"{split_file}:{line_number}: missing image {image_path}"
            )
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"{split_file}:{line_number}: missing mask {mask_path}"
            )
        pairs.append((image_path, mask_path))
    return pairs


def export_split(pairs: Iterable[tuple[Path, Path]], destination: Path) -> int:
    """Copy pairs into ``destination/images`` and ``destination/masks``."""
    images_dir = destination / "images"
    masks_dir = destination / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    written_stems: set[str] = set()
    for image_path, mask_path in pairs:
        stem = image_path.stem
        shutil.copy2(image_path, images_dir / f"{stem}{image_path.suffix.lower()}")
        mask_target = masks_dir / f"{stem}.png"
        shutil.copy2(mask_path, mask_target)
        cached_mask = mask_target.with_suffix(".npy")
        if cached_mask.exists():
            cached_mask.unlink()
        written_stems.add(stem)
    _remove_other_files(images_dir, written_stems, {".png"})
    _remove_other_files(masks_dir, written_stems, {".png", ".npy"})
    return len(written_stems)


def split_is_ready(destination: Path, image_count: int) -> bool:
    """Return whether every image has a mask with the same stem."""
    images_dir = destination / "images"
    masks_dir = destination / "masks"
    image_stems = _png_stems(images_dir)
    mask_stems = _png_stems(masks_dir)
    return len(image_stems) == image_count and image_stems == mask_stems


def export_camvid(
    dataset_root: Path = DATASET_ROOT,
    download_root: Path = DOWNLOAD_ROOT,
) -> Path:
    """Download CamVid and export ``train``, ``val``, and ``test``."""
    if all(
        split_is_ready(dataset_root / split_name, image_count)
        for split_name, image_count in SPLIT_IMAGE_COUNTS.items()
    ):
        print(f"CamVid already exported at {dataset_root}")
        return dataset_root

    source_root = _extract_archive(
        _download_archive(download_root), download_root / "src"
    )
    for split_name, split_filename in _SPLIT_FILES.items():
        destination = dataset_root / split_name
        image_count = SPLIT_IMAGE_COUNTS[split_name]
        if split_is_ready(destination, image_count):
            print(f"{split_name}: already exported ({image_count} images)")
            continue
        written = export_split(
            tqdm(
                camvid_pairs(source_root, source_root / split_filename),
                desc=f"Exporting {split_name}",
                total=image_count,
            ),
            destination,
        )
        if written != image_count:
            raise RuntimeError(
                f"{split_name}: expected {image_count} images, wrote {written}"
            )
        print(f"{split_name}: wrote {written} images to {destination}")
    return dataset_root


def _download_archive(download_root: Path) -> Path:
    archive = download_root / "camvid.zip"
    if archive.is_file() and zipfile.is_zipfile(archive):
        return archive

    download_root.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(".zip.part")
    request = urllib.request.Request(CAMVID_URL, headers={"User-Agent": "dnn-pytorch"})
    with (
        urllib.request.urlopen(request) as response,
        partial.open("wb") as handle,
    ):
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header else None
        with tqdm(
            total=total, unit="B", unit_scale=True, desc="Downloading CamVid"
        ) as progress:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                progress.update(len(chunk))
    partial.replace(archive)
    return archive


def _extract_archive(archive: Path, destination: Path) -> Path:
    if _archive_is_extracted(destination):
        return destination

    with zipfile.ZipFile(archive) as zip_file:
        extracted = 0
        for info in zip_file.infolist():
            relative = _member_target(info.filename)
            if relative is None or info.is_dir():
                continue
            target = destination / Path(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            extracted += 1
    if extracted == 0:
        raise RuntimeError(f"CamVid archive contains no images or labels: {archive}")
    return destination


def _archive_is_extracted(destination: Path) -> bool:
    image_count = sum(SPLIT_IMAGE_COUNTS.values())
    return (
        _png_count(destination / _IMAGE_DIR) == image_count
        and _png_count(destination / _LABEL_DIR) == image_count
        and all((destination / name).is_file() for name in _SPLIT_FILES.values())
    )


def _member_target(filename: str) -> PurePosixPath | None:
    relative = PurePosixPath(filename)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2:
        return None
    inner = PurePosixPath(*relative.parts[1:])
    if inner.parts[0] in {_IMAGE_DIR, _LABEL_DIR}:
        return inner
    if len(inner.parts) == 1 and inner.name in _SPLIT_FILES.values():
        return inner
    return None


def _remove_other_files(
    directory: Path, keep_stems: set[str], suffixes: set[str]
) -> None:
    for path in directory.iterdir():
        if (
            path.is_file()
            and path.suffix.lower() in suffixes
            and path.stem not in keep_stems
        ):
            path.unlink()


def _png_stems(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }


def _png_count(directory: Path) -> int:
    return len(_png_stems(directory))


def main() -> None:
    root = export_camvid()
    print(f"CamVid ready at {root}")


if __name__ == "__main__":
    main()
