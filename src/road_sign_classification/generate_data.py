"""Generate a synthetic image dataset for road sign classification (parallel export to disk).

Loads real class templates and random backgrounds, composites them, applies the Hydra
``augmentations`` pipeline, and saves RGB PNGs per class. By default output resolution
matches the augment pipeline (variable sizes are preserved). Set
``generation.resize_to_model=true`` to also ``Resize`` to ``model.width`` x ``model.height``
after augmentations. Designed to be light on workers (no per-worker full dataset or
PyTorch tensors — see ``SyntheticSignGenerator`` in ``dataset.py``).
"""

import multiprocessing as mp
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import hydra
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from src.common.augmentations import get_augmentations
from src.common.logging import setup_logging
from src.common.utils import set_seed
from src.road_sign_classification.config import ClassificationConfig
from src.road_sign_classification.dataset import (
    ImageCache,
    SyntheticResourceIndex,
    SyntheticSignGenerator,
)
from src.road_sign_classification.model_manager import (
    build_classification_preprocessing,
)

logger = setup_logging(__file__)

OUTPUT_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
OUTPUT_INDEX_PATTERN = re.compile(r"_(\d+)$")


@dataclass(frozen=True)
class OutputClassState:
    existing_count: int
    max_index: int


@dataclass(frozen=True)
class GenerationTask:
    task_id: int
    class_name: str
    start_index: int
    count: int
    seed: int


worker_resources: SyntheticResourceIndex | None = None
worker_generator: SyntheticSignGenerator | None = None
worker_background_cache: ImageCache | None = None
worker_template_cache: ImageCache | None = None
worker_output_dir: Path | None = None
worker_png_compress_level: int = 1
worker_failed_background_paths: set[str] | None = None


def resolve_project_path(original_cwd: Path, raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else (original_cwd / path).resolve()


def resolve_backgrounds_dir(config: ClassificationConfig, original_cwd: Path) -> Path:
    return (
        resolve_project_path(
            original_cwd,
            config.generation.backgrounds_dir or config.data.backgrounds_dir,
        )
        or original_cwd
    )


def resolve_background_stride(config: ClassificationConfig) -> int:
    return (
        config.generation.background_sample_stride
        if config.generation.background_sample_stride is not None
        else config.data.background_sample_stride
    )


def resolve_num_workers(config: ClassificationConfig, task_count: int) -> int:
    if task_count <= 0:
        return 1
    if config.generation.num_workers is not None:
        return max(1, min(config.generation.num_workers, task_count))
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count // 2, 8, task_count))


def resolve_start_method(config: ClassificationConfig) -> str:
    if config.generation.start_method:
        return config.generation.start_method
    if sys.platform.startswith("linux"):
        return "fork"
    return "spawn"


def prepare_output_dir(
    output_dir: Path,
    class_names: list[str],
    *,
    incremental: bool,
) -> None:
    if output_dir.exists() and not incremental:
        logger.info("Removing existing output directory: %s", output_dir)
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    for class_name in class_names:
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)


def scan_output_dir(
    output_dir: Path, class_names: list[str]
) -> dict[str, OutputClassState]:
    states: dict[str, OutputClassState] = {}
    for class_name in class_names:
        class_dir = output_dir / class_name
        existing_count = 0
        max_index = -1

        if class_dir.exists():
            for file_path in class_dir.iterdir():
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in OUTPUT_IMAGE_EXTENSIONS:
                    continue

                existing_count += 1
                match = OUTPUT_INDEX_PATTERN.search(file_path.stem)
                if match is None:
                    continue
                max_index = max(max_index, int(match.group(1)))

        states[class_name] = OutputClassState(
            existing_count=existing_count,
            max_index=max_index,
        )
    return states


def build_generation_tasks(
    output_states: dict[str, OutputClassState],
    *,
    target_per_class: int,
    chunk_size: int,
    seed: int = 1337,
) -> tuple[list[GenerationTask], dict[str, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    tasks: list[GenerationTask] = []
    missing_by_class: dict[str, int] = {}
    rng = random.Random(seed)

    for class_name, state in output_states.items():
        missing = max(0, target_per_class - state.existing_count)
        missing_by_class[class_name] = missing
        if missing == 0:
            continue

        start_index = state.max_index + 1
        for offset in range(0, missing, chunk_size):
            tasks.append(
                GenerationTask(
                    task_id=len(tasks),
                    class_name=class_name,
                    start_index=start_index + offset,
                    count=min(chunk_size, missing - offset),
                    seed=rng.randint(0, 2**31 - 1),
                )
            )

    return tasks, missing_by_class


def build_export_transforms(config: ClassificationConfig) -> A.Compose:
    train_augs = get_augmentations(config.augmentations)
    if not config.generation.resize_to_model:
        return train_augs
    export_preprocessing = build_classification_preprocessing(
        config.model,
        include_normalize=False,
        include_tensor=False,
    )
    return A.Compose(train_augs.transforms + export_preprocessing.transforms)


def init_worker(
    config_dict: dict[str, Any],
    resource_index: SyntheticResourceIndex,
    output_dir: Path,
) -> None:
    global worker_resources
    global worker_generator
    global worker_background_cache
    global worker_template_cache
    global worker_failed_background_paths
    global worker_output_dir
    global worker_png_compress_level

    config = ClassificationConfig.from_dict(config_dict)
    worker_resources = resource_index
    worker_generator = SyntheticSignGenerator(build_export_transforms(config))
    worker_background_cache = ImageCache(
        config.generation.background_cache_size,
        mode="RGB",
    )
    worker_template_cache = ImageCache(
        config.generation.template_cache_size,
        mode="RGBA",
    )
    worker_failed_background_paths = set()
    worker_output_dir = output_dir
    worker_png_compress_level = max(0, min(9, config.generation.png_compress_level))


def get_random_background() -> Image.Image:
    if (
        worker_resources is None
        or worker_background_cache is None
        or worker_failed_background_paths is None
    ):
        raise RuntimeError("Worker background resources are not initialized.")

    candidate_paths = worker_resources.background_paths
    if not candidate_paths:
        raise RuntimeError("No background paths available for generation.")

    max_attempts = min(len(candidate_paths), 32)
    for _ in range(max_attempts):
        background_path = random.choice(candidate_paths)
        key = str(background_path)
        if key in worker_failed_background_paths:
            continue

        try:
            return worker_background_cache.get(background_path)
        except RuntimeError:
            worker_failed_background_paths.add(key)

    raise RuntimeError(
        "Unable to load any usable background image; all attempted background loads failed."
    )


def generate_images_worker(task: GenerationTask) -> dict[str, int | str]:
    if (
        worker_resources is None
        or worker_generator is None
        or worker_background_cache is None
        or worker_template_cache is None
        or worker_failed_background_paths is None
        or worker_output_dir is None
    ):
        raise RuntimeError("Worker resources are not initialized.")

    random.seed(task.seed)
    np.random.seed(task.seed % (2**32))

    template_paths = worker_resources.template_paths_by_class[task.class_name]
    class_dir = worker_output_dir / task.class_name
    generated_count = 0

    for offset in range(task.count):
        template = worker_template_cache.get(random.choice(template_paths))
        background = get_random_background()
        synthetic_image = worker_generator.generate_synthetic_image(
            template, background
        )
        image_path = (
            class_dir / f"{task.class_name}_{task.start_index + offset:08d}.png"
        )
        Image.fromarray(synthetic_image).save(
            image_path,
            format="PNG",
            optimize=False,
            compress_level=worker_png_compress_level,
        )
        generated_count += 1

    return {
        "task_id": task.task_id,
        "class_name": task.class_name,
        "generated_count": generated_count,
    }


@hydra.main(
    version_base=None,
    config_path="../../configs/road_sign_classification",
    config_name="main.yaml",
)
def main(cfg: dict) -> None:
    config = ClassificationConfig.from_dict(dict(cfg))
    logger.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    set_seed()

    original_cwd = Path(hydra.utils.get_original_cwd())
    data_root = resolve_project_path(original_cwd, config.data.root)
    if data_root is None:
        raise RuntimeError("config.data.root must not be null")

    train_root = (data_root / config.data.train_dir).resolve()
    backgrounds_dir = resolve_backgrounds_dir(config, original_cwd)
    output_dir = resolve_project_path(original_cwd, config.generation.output_dir)
    if output_dir is None:
        raise RuntimeError("config.generation.output_dir must not be null")

    resource_index = SyntheticResourceIndex.build(
        data_root=train_root,
        extensions=config.data.extensions,
        max_samples=config.data.max_samples,
        backgrounds_dir=backgrounds_dir,
        background_sample_stride=resolve_background_stride(config),
    )
    class_names = resource_index.class_names
    prepare_output_dir(
        output_dir,
        class_names,
        incremental=config.generation.incremental,
    )

    output_states = scan_output_dir(output_dir, class_names)
    tasks, missing_by_class = build_generation_tasks(
        output_states,
        target_per_class=config.generation.target_per_class,
        chunk_size=config.generation.chunk_size,
    )
    total_images_to_generate = sum(missing_by_class.values())

    for class_name in class_names:
        current_count = output_states[class_name].existing_count
        missing = missing_by_class[class_name]
        if missing > 0:
            logger.info(
                "Class '%s': has %d, needs %d more",
                class_name,
                current_count,
                missing,
            )

    if total_images_to_generate == 0:
        logger.info(
            "All classes already satisfy target_per_class=%d.",
            config.generation.target_per_class,
        )
        return

    num_workers = resolve_num_workers(config, len(tasks))
    start_method = resolve_start_method(config)
    logger.info(
        "Generating %d images across %d tasks with %d workers (%s).",
        total_images_to_generate,
        len(tasks),
        num_workers,
        start_method,
    )

    ctx = mp.get_context(start_method)
    pending_tasks = {task.task_id: task for task in tasks}
    pool = None
    try:
        pool = ctx.Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(dict(cfg), resource_index, output_dir),
        )
        with tqdm(total=total_images_to_generate, desc="Generating images") as pbar:
            async_result = pool.imap_unordered(
                generate_images_worker, tasks, chunksize=1
            )
            completed_tasks = 0
            while completed_tasks < len(tasks):
                try:
                    result = async_result.next(timeout=30)
                except mp.TimeoutError:
                    remaining = list(pending_tasks.values())[:5]
                    logger.warning(
                        "No completed tasks in the last 30s. Remaining tasks: %d. Sample classes: %s",
                        len(pending_tasks),
                        ", ".join(task.class_name for task in remaining),
                    )
                    continue

                completed_tasks += 1
                pending_tasks.pop(int(result["task_id"]), None)
                pbar.update(int(result["generated_count"]))
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        logger.error("Interrupted by user.")
        if pool is not None:
            pool.terminate()
            pool.join()
        raise SystemExit(1) from None
    except Exception:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise

    logger.info(
        "Generation complete. Total images saved: %d",
        total_images_to_generate,
    )


if __name__ == "__main__":
    main()
