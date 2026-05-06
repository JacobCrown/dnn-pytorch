import hashlib
import json
import logging
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from src.core.dataset import BaseDataset
from src.road_sign_classification.config import DEFAULT_ROAD_SIGN_BACKGROUNDS_DIR

logger = logging.getLogger(__name__)
BACKGROUND_INDEX_CACHE_DIR = (
    Path.home() / ".cache" / "dnn-pytorch" / "road-sign-backgrounds"
)


def sign_family_prefix(class_name: str) -> str:
    """Return the family prefix, for example `A-7 -> A`."""
    name = class_name.strip()
    if not name:
        return "misc"
    if "-" in name:
        prefix = name.split("-", 1)[0].strip()
        return prefix or "misc"
    return name[0].upper()


def _normalize_extensions(extensions: list[str]) -> set[str]:
    return {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions
    }


def _list_image_paths(
    directory: Path,
    extensions: list[str],
    *,
    recursive: bool,
) -> list[Path]:
    if not directory.exists():
        return []

    valid_extensions = _normalize_extensions(extensions)
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in valid_extensions
    )


def _background_cache_key(
    directory: Path,
    extensions: list[str],
    stride: int,
) -> str:
    payload = json.dumps(
        {
            "directory": str(directory.resolve()),
            "extensions": sorted(_normalize_extensions(extensions)),
            "stride": stride,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _background_cache_path(
    directory: Path,
    extensions: list[str],
    stride: int,
) -> Path:
    return (
        BACKGROUND_INDEX_CACHE_DIR
        / f"{_background_cache_key(directory, extensions, stride)}.json"
    )


def _load_cached_background_paths(
    directory: Path,
    extensions: list[str],
    stride: int,
) -> list[Path] | None:
    cache_path = _background_cache_path(directory, extensions, stride)
    if not cache_path.exists():
        return None

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read background cache %s: %s", cache_path, exc)
        return None

    cached_paths = [Path(path_str) for path_str in payload.get("paths", [])]
    if not cached_paths:
        return None

    existing_paths = [path for path in cached_paths if path.exists()]
    if len(existing_paths) != len(cached_paths):
        logger.warning(
            "Background cache %s is stale; rebuilding manifest.",
            cache_path,
        )
        return None

    logger.info(
        "Loaded %d sampled backgrounds from cache %s",
        len(existing_paths),
        cache_path,
    )
    return existing_paths


def _save_cached_background_paths(
    directory: Path,
    extensions: list[str],
    stride: int,
    background_paths: list[Path],
) -> None:
    if not background_paths:
        return

    cache_path = _background_cache_path(directory, extensions, stride)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "directory": str(directory.resolve()),
        "stride": stride,
        "extensions": sorted(_normalize_extensions(extensions)),
        "paths": [str(path) for path in background_paths],
    }
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_sampled_background_paths(
    directory: Path,
    extensions: list[str],
    stride: int,
) -> list[Path]:
    cached_paths = _load_cached_background_paths(directory, extensions, stride)
    if cached_paths is not None:
        return cached_paths

    logger.info("Scanning background directory %s", directory)
    all_background_paths = _list_image_paths(
        directory,
        extensions,
        recursive=True,
    )
    sampled_background_paths = _sample_background_paths(all_background_paths, stride)
    if not sampled_background_paths and all_background_paths:
        sampled_background_paths = all_background_paths
    _save_cached_background_paths(
        directory,
        extensions,
        stride,
        sampled_background_paths,
    )
    return sampled_background_paths


def _sample_background_paths(background_paths: list[Path], stride: int) -> list[Path]:
    if stride <= 1:
        return background_paths
    return [
        path
        for index, path in enumerate(background_paths, start=1)
        if index % stride == 0
    ]


def load_pil_image(file_path: str | Path, mode: str | None = None) -> Image.Image:
    path = Path(file_path)
    try:
        with Image.open(path) as image:
            image.load()
            loaded = image.convert(mode) if mode is not None else image.copy()
        return loaded
    except Exception as exc:
        raise RuntimeError(f"Error loading image {path}: {exc}") from exc


def build_sign_augmentation_pipeline() -> A.Compose:
    # return A.Compose(
    #     [
    #         A.LongestMaxSize(
    #             max_size=256,
    #             p=1.0,
    #             interpolation=cv2.INTER_AREA,
    #             mask_interpolation=cv2.INTER_CUBIC,
    #         ),
    #         A.RandomScale(scale_limit=(-0.9, 0.0), p=1.0, interpolation=cv2.INTER_AREA, mask_interpolation=cv2.INTER_CUBIC),
    #         A.OneOf(
    #             [
    #                 A.Affine(
    #                     scale={"x": (0.7, 1), "y": (0.9, 1)},
    #                     rotate=(-5, 5),
    #                     interpolation=cv2.INTER_AREA,
    #                     mask_interpolation=cv2.INTER_CUBIC,
    #                     shear=(-5, 5),
    #                     fit_output=True,
    #                     p=1.0,
    #                 ),
    #             ],
    #             p=0.6,
    #         ),
    #         A.Perspective(scale=(0.01, 0.03), keep_size=False, fit_output=True, p=1.0),
    #         A.OneOf(
    #             [
    #                 A.HueSaturationValue(
    #                     hue_shift_limit=5,
    #                     sat_shift_limit=10,
    #                     val_shift_limit=20,
    #                     p=1.0,
    #                 ),
    #                 A.RandomBrightnessContrast(
    #                     brightness_limit=(-0.4, 0.2),
    #                     contrast_limit=(-0.4, 0.2),
    #                     p=1.0,
    #                 ),
    #                 A.RandomToneCurve(scale=0.3, p=1.0),
    #             ],
    #             p=1.0,
    #         ),
    #         A.OneOf(
    #             [
    #                 A.Spatter(
    #                     mean=0.1,
    #                     std=0.2,
    #                     gauss_sigma=2,
    #                     cutout_threshold=0.4,
    #                     intensity=0.5,
    #                     mode="mud",
    #                     p=1.0,
    #                 ),
    #             ],
    #             p=0.3,
    #         ),
    #         A.OneOf(
    #             [
    #                 A.ImageCompression(quality_range=(5, 30), p=1.0),
    #             ],
    #             p=1.0,
    #         ),
    #     ]
    # )
    return A.Compose(
        [
            A.LongestMaxSize(
                max_size=256,
                p=1.0,
                interpolation=cv2.INTER_AREA,
                mask_interpolation=cv2.INTER_CUBIC,
            ),
            A.OneOf(
                [
                    A.Affine(
                        scale={"x": (0.5, 1), "y": (0.9, 1)},
                        rotate=(-10, 10),
                        interpolation=cv2.INTER_AREA,
                        mask_interpolation=cv2.INTER_CUBIC,
                        shear=(-5, 5),
                        fit_output=True,
                        p=1.0,
                    ),
                ],
                p=1.0,
            ),
            A.HueSaturationValue(
                hue_shift_limit=7,
                sat_shift_limit=10,
                val_shift_limit=20,
                p=1.0,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.4, 0.2),
                        contrast_limit=(-0.4, 0.2),
                        p=1.0,
                    ),
                    A.RandomToneCurve(scale=0.3, p=1.0),
                ],
                p=1.0,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(
                        blur_limit=(5, 9),
                        sigma_limit=(1, 3),
                        p=1.0,
                    ),
                    A.MotionBlur(allow_shifted=False, p=1.0),
                    A.GaussNoise(
                        std_range=(0.01, 0.05),
                        mean_range=(-0.01, 0.01),
                        p=1.0,
                    ),
                ],
                p=0.8,
            ),
            A.GaussNoise(
                std_range=(0.01, 0.05),
                mean_range=(-0.01, 0.01),
                p=0.3,
            ),
            A.OneOf(
                [
                    A.Spatter(
                        mean=0.1,
                        std=0.2,
                        gauss_sigma=2,
                        cutout_threshold=0.4,
                        intensity=0.5,
                        mode="mud",
                        p=1.0,
                    ),
                ],
                p=1.0,
            ),
            A.OneOf(
                [
                    A.ImageCompression(quality_range=(5, 20), p=1.0),
                    A.RandomScale(
                        scale_limit=(-0.9, 0.0),
                        p=1.0,
                        interpolation=cv2.INTER_AREA,
                        mask_interpolation=cv2.INTER_CUBIC,
                    ),
                ],
                p=1.0,
            ),
        ]
    )


@dataclass(frozen=True)
class DataDistribution:
    class_name: str
    count: int
    percentage: float


@dataclass(frozen=True)
class SyntheticResourceIndex:
    data_root: Path
    class_names: list[str]
    class_to_idx: dict[str, int]
    idx_to_class: dict[int, str]
    template_paths_by_class: dict[str, list[Path]]
    background_paths: list[Path]

    @classmethod
    def build(
        cls,
        *,
        data_root: Path,
        extensions: list[str],
        max_samples: int | None = None,
        backgrounds_dir: Path | None = None,
        background_sample_stride: int = 25,
    ) -> "SyntheticResourceIndex":
        resolved_backgrounds_dir = (
            Path(backgrounds_dir)
            if backgrounds_dir is not None
            else Path(DEFAULT_ROAD_SIGN_BACKGROUNDS_DIR)
        )
        class_names = sorted(
            [path.name for path in data_root.iterdir() if path.is_dir()]
        )
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

        template_paths_by_class: dict[str, list[Path]] = {}
        total_templates = 0
        for class_name in class_names:
            class_dir = data_root / class_name
            template_paths = _list_image_paths(
                class_dir,
                extensions,
                recursive=False,
            )
            if max_samples is not None:
                template_paths = template_paths[:max_samples]
            if not template_paths:
                raise RuntimeError(
                    f"No valid templates found for class '{class_name}' in {class_dir}"
                )
            template_paths_by_class[class_name] = template_paths
            total_templates += len(template_paths)

        background_paths = get_sampled_background_paths(
            resolved_backgrounds_dir,
            extensions,
            background_sample_stride,
        )
        if not background_paths:
            raise RuntimeError(
                f"No valid background images found in {resolved_backgrounds_dir}"
            )

        logger.info(
            "Indexed %d classes, %d templates, %d sampled backgrounds",
            len(class_names),
            total_templates,
            len(background_paths),
        )
        return cls(
            data_root=data_root,
            class_names=class_names,
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
            template_paths_by_class=template_paths_by_class,
            background_paths=background_paths,
        )


class ImageCache:
    """Small per-worker cache for decoded PIL images."""

    def __init__(self, max_size: int, *, mode: str | None = None):
        self.max_size = max(1, max_size)
        self.mode = mode
        self._cache: OrderedDict[str, Image.Image] = OrderedDict()

    def get(self, file_path: str | Path) -> Image.Image:
        key = str(file_path)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        image = load_pil_image(file_path, mode=self.mode)
        self._cache[key] = image
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return image


class SyntheticSignGenerator:
    """Reusable synthetic sign compositor shared by training and export."""

    def __init__(
        self,
        transforms: A.Compose,
        sign_transforms: A.Compose | None = None,
        *,
        transforms_by_family: dict[str, A.Compose] | None = None,
        sign_transforms_by_family: dict[str, A.Compose] | None = None,
    ):
        self.transforms = transforms
        self.transforms_sign = sign_transforms or build_sign_augmentation_pipeline()
        self.transforms_by_family = transforms_by_family or {}
        self.sign_transforms_by_family = sign_transforms_by_family or {}

    def _tile_transforms_for_family(self, family: str | None) -> A.Compose:
        if family is None:
            return self.transforms
        return self.transforms_by_family.get(family, self.transforms)

    def _sign_transforms_for_family(self, family: str | None) -> A.Compose:
        if family is None:
            return self.transforms_sign
        return self.sign_transforms_by_family.get(family, self.transforms_sign)

    def generate_synthetic_sign(
        self,
        sign_template: Image.Image,
        background: Image.Image,
        *,
        family: str | None = None,
    ) -> Any:
        img_augmented = self.augment_sign(sign_template, family=family)
        img_background = self.paste_image_on_background(img_augmented, background)
        img_background_aug = self._tile_transforms_for_family(family)(
            image=np.array(img_background)
        )
        return img_background_aug["image"]

    def generate_synthetic_image(
        self,
        sign_template: Image.Image,
        background: Image.Image,
        *,
        family: str | None = None,
    ) -> np.ndarray:
        image = self.generate_synthetic_sign(sign_template, background, family=family)
        if isinstance(image, torch.Tensor):
            raise TypeError(
                "SyntheticSignGenerator expected image-space transforms, got tensor output."
            )
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def augment_sign(
        self, img: Image.Image, *, family: str | None = None
    ) -> Image.Image:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        img_array = np.array(img)
        rgb = img_array[:, :, :3]
        alpha = img_array[:, :, 3:4]

        img_transformed = self._sign_transforms_for_family(family)(
            image=rgb, mask=alpha
        )
        augmented_rgb = img_transformed["image"]
        augmented_alpha = img_transformed["mask"]
        augmented_array = np.dstack([augmented_rgb, augmented_alpha])
        return Image.fromarray(augmented_array, "RGBA")

    def paste_image_on_background(
        self, img: Image.Image, background: Image.Image
    ) -> Image.Image:
        """Paste a sign on a random ROI from the chosen background."""
        bg_w, bg_h = background.size
        icon_w, icon_h = img.size

        # margin_width_right = random.randint(0, int(icon_w * 0.05))
        # margin_width_left = random.randint(0, int(icon_w * 0.05))
        # margin_height_bottom = random.randint(0, int(icon_h * 0.05))
        # margin_height_top = random.randint(0, int(icon_h * 0.05))
        margin_width_right = 0
        margin_width_left = 0
        margin_height_bottom = 0
        margin_height_top = 0
        target_w, target_h = img.size

        max_x = max(margin_width_left, bg_w - target_w - margin_width_right)
        max_y = max(margin_height_top, bg_h - target_h - margin_height_bottom)
        paste_x = random.randint(margin_width_left, max_x)
        paste_y = random.randint(margin_height_top, max_y)

        roi_left = paste_x - margin_width_left
        roi_upper = paste_y - margin_height_top
        roi_right = paste_x + target_w + margin_width_right
        roi_lower = paste_y + target_h + margin_height_bottom

        roi = background.crop((roi_left, roi_upper, roi_right, roi_lower))
        if random.random() < 0.5:
            roi = ImageOps.mirror(roi)

        roi.paste(img, (margin_width_left, margin_height_top), img)
        return roi


class ClassificationDataset(BaseDataset):
    """Balanced road sign dataset with on-the-fly synthetic composition."""

    def __init__(
        self,
        data_root: Path,
        transforms: A.Compose,
        extensions: list[str],
        max_samples: int | None = None,
        samples_per_class: int = 5,
        backgrounds_dir: Path | None = None,
        background_sample_stride: int = 20,
        preload_backgrounds: bool = True,
        transforms_by_family: dict[str, A.Compose] | None = None,
        sign_transforms_by_family: dict[str, A.Compose] | None = None,
    ):
        self.samples_by_class: dict[int, list[Image.Image]] = {}
        self.class_family_by_idx: dict[int, str] = {}
        self.samples_per_class = samples_per_class
        self.backgrounds_dir = (
            Path(backgrounds_dir)
            if backgrounds_dir is not None
            else Path(DEFAULT_ROAD_SIGN_BACKGROUNDS_DIR)
        )
        self.background_sample_stride = background_sample_stride
        self.preload_backgrounds = preload_backgrounds
        self.background_paths: list[Path] = []
        self.backgrounds: list[Image.Image] = []
        self.synthetic_generator = SyntheticSignGenerator(
            transforms,
            transforms_by_family=transforms_by_family,
            sign_transforms_by_family=sign_transforms_by_family,
        )

        super().__init__(
            data_root=data_root,
            transforms=transforms,
            extensions=extensions,
            max_samples=max_samples,
        )

    def _build_samples(self) -> None:
        resource_index = SyntheticResourceIndex.build(
            data_root=self.data_root,
            extensions=self.extensions,
            max_samples=self.max_samples,
            backgrounds_dir=self.backgrounds_dir,
            background_sample_stride=self.background_sample_stride,
        )
        self.class_names = resource_index.class_names
        self.class_to_idx = resource_index.class_to_idx
        self.idx_to_class = resource_index.idx_to_class
        self.background_paths = resource_index.background_paths

        logger.info("Loading %d sampled background images", len(self.background_paths))
        if self.preload_backgrounds:
            self.backgrounds = [
                load_pil_image(background_path, mode="RGB")
                for background_path in self.background_paths
            ]
            logger.info("Loaded %d backgrounds into memory", len(self.backgrounds))

        total_images = 0
        for class_name in self.class_names:
            class_idx = self.class_to_idx[class_name]
            self.class_family_by_idx[class_idx] = sign_family_prefix(class_name)
            templates = [
                load_pil_image(template_path)
                for template_path in resource_index.template_paths_by_class[class_name]
            ]
            self.samples_by_class[class_idx] = templates
            total_images += len(templates)

        logger.info(
            "Loaded %d images across %d classes",
            total_images,
            len(self.class_names),
        )
        logger.info(
            "Each epoch will have %d samples (%d per class)",
            len(self.class_names) * self.samples_per_class,
            self.samples_per_class,
        )

    @staticmethod
    def _load_image(file_path: str | Path) -> Image.Image:
        return load_pil_image(file_path)

    def __len__(self) -> int:
        return len(self.class_names) * self.samples_per_class

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        class_idx = random.randint(0, len(self.class_names) - 1)
        class_images = self.samples_by_class[class_idx]
        image = random.choice(class_images)
        family = self.class_family_by_idx.get(class_idx)
        if self.preload_backgrounds:
            background = random.choice(self.backgrounds)
        else:
            background = load_pil_image(
                random.choice(self.background_paths), mode="RGB"
            )

        synthetic_sign = self.generate_synthetic_sign(image, background, family=family)
        return synthetic_sign, class_idx

    def get_class_names(self) -> list[str]:
        return self.class_names

    def get_class_to_idx(self) -> dict[str, int]:
        return self.class_to_idx

    def get_idx_to_class(self) -> dict[int, str]:
        return self.idx_to_class

    def get_data_distribution(self) -> list[DataDistribution]:
        total_images = sum(len(imgs) for imgs in self.samples_by_class.values())
        distributions: list[DataDistribution] = []
        for class_name in self.class_names:
            class_idx = self.class_to_idx[class_name]
            count = len(self.samples_by_class[class_idx])
            percentage = (count / total_images) * 100 if total_images > 0 else 0.0
            distributions.append(
                DataDistribution(
                    class_name=class_name,
                    count=count,
                    percentage=percentage,
                )
            )
        return distributions

    def generate_synthetic_sign(
        self,
        sign_template: Image.Image,
        background: Image.Image,
        *,
        family: str | None = None,
    ) -> Any:
        return self.synthetic_generator.generate_synthetic_sign(
            sign_template,
            background,
            family=family,
        )

    def augment_sign(
        self, img: Image.Image, *, family: str | None = None
    ) -> Image.Image:
        return self.synthetic_generator.augment_sign(img, family=family)

    def paste_image_on_background(
        self, img: Image.Image, background: Image.Image
    ) -> Image.Image:
        return self.synthetic_generator.paste_image_on_background(img, background)
