"""Main training pipeline for image classification with Hydra configuration."""

import json
import logging
from pathlib import Path

import albumentations as A
import hydra
from omegaconf import OmegaConf
from torchinfo import summary

from src.common.augmentations import get_augmentations
from src.common.data_utils import create_data_loaders
from src.common.exceptions import DataLoaderException
from src.common.experiment_paths import road_sign_classification_experiment_root
from src.common.hydra_resolvers import register_hydra_resolvers
from src.common.logging import configure_run_logging
from src.common.losses import get_loss_function
from src.common.optimizers import get_optimizer
from src.common.schedulers import get_scheduler
from src.common.utils import DEVICE, set_seed
from src.road_sign_classification.config import ClassificationConfig
from src.road_sign_classification.dataset import ClassificationDataset
from src.road_sign_classification.model_manager import ClassificationModelManager
from src.road_sign_classification.trainer import ClassificationTrainer
from task.aug_pipelines import (
    AugParams,
    PipelineStructure,
    build_sign_compose,
    build_tile_compose,
)

register_hydra_resolvers()

logger = logging.getLogger(__name__)


def _load_family_search_results(
    search_results_dir: Path | None,
    *,
    data_preproc: A.Compose,
) -> tuple[dict[str, A.Compose], dict[str, A.Compose]]:
    """Load per-family Stage A and Stage C pipelines from aug_search JSON results."""
    if search_results_dir is None or not search_results_dir.is_dir():
        return {}, {}

    sign_transforms_by_family: dict[str, A.Compose] = {}
    tile_transforms_by_family: dict[str, A.Compose] = {}
    for result_path in sorted(search_results_dir.glob("best_aug_config_*.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        family = payload.get("family")
        if not family:
            family = result_path.stem.removeprefix("best_aug_config_")
        params = AugParams.from_jsonable(payload["params"])
        structure = PipelineStructure.from_jsonable(payload["structure"])
        sign_transforms_by_family[family] = build_sign_compose(params, structure)
        tile_only = build_tile_compose(params, structure)
        tile_transforms_by_family[family] = A.Compose(
            tile_only.transforms + data_preproc.transforms
        )

    logger.info(
        "Loaded per-family search results for %d families from %s",
        len(sign_transforms_by_family),
        search_results_dir,
    )
    return sign_transforms_by_family, tile_transforms_by_family


@hydra.main(
    version_base=None,
    config_path="../../configs/road_sign_classification",
    config_name="main.yaml",
)
def main(cfg: dict) -> None:
    config = ClassificationConfig.from_dict(dict(cfg))
    experiment_dir = road_sign_classification_experiment_root(
        train_root_dir=config.train.root_dir,
        data_root=config.data.root,
        model_name=config.model.name,
        height=config.model.height,
        width=config.model.width,
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
    ).resolve()
    configure_run_logging(experiment_directory=experiment_dir)
    logger.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # ---------------------------------------------------------------------
    # Seed & device
    # ---------------------------------------------------------------------
    set_seed()
    logger.info("Device: %s", DEVICE)

    # ---------------------------------------------------------------------
    # Model and data preprocessing
    # ---------------------------------------------------------------------
    classification_model_manager = ClassificationModelManager(config.model)
    model = classification_model_manager.model
    data_preproc = classification_model_manager.preprocessing

    logger.debug(
        "Model summary:\n%s",
        summary(
            model,
            input_size=(
                1,
                config.model.in_channels,
                config.model.height,
                config.model.width,
            ),
            col_names=("input_size", "output_size", "num_params", "trainable"),
            verbose=0,
        ),
    )

    # ---------------------------------------------------------------------
    # Augmentations + Preprocessing
    # ---------------------------------------------------------------------
    logger.info("Building augmentation and preprocessing pipelines…")

    train_augs = get_augmentations(config.augmentations)

    train_transforms = A.Compose(train_augs.transforms + data_preproc.transforms)
    logger.debug("Train transforms: %s", train_transforms)

    val_transforms = A.Compose(data_preproc.transforms)
    logger.debug("Val transforms: %s", val_transforms)
    family_search_results_dir = (
        Path(config.family_search_results_dir)
        if config.family_search_results_dir is not None
        else None
    )
    sign_transforms_by_family, tile_transforms_by_family = _load_family_search_results(
        family_search_results_dir,
        data_preproc=data_preproc,
    )

    # ---------------------------------------------------------------------
    # Datasets & loaders
    # ---------------------------------------------------------------------
    data_config = config.data
    train_dataset = ClassificationDataset(
        data_root=Path(data_config.root) / data_config.train_dir,
        transforms=train_transforms,
        extensions=data_config.extensions,
        max_samples=data_config.max_samples,
        samples_per_class=data_config.samples_per_class,
        backgrounds_dir=Path(data_config.backgrounds_dir),
        background_sample_stride=data_config.background_sample_stride,
        preload_backgrounds=data_config.preload_backgrounds,
        transforms_by_family=tile_transforms_by_family,
        sign_transforms_by_family=sign_transforms_by_family,
    )
    val_dataset = ClassificationDataset(
        data_root=Path(data_config.root) / data_config.val_dir,
        transforms=train_transforms,
        extensions=data_config.extensions,
        max_samples=data_config.max_samples,
        samples_per_class=data_config.samples_per_class,
        backgrounds_dir=Path(data_config.backgrounds_dir),
        background_sample_stride=data_config.background_sample_stride,
        preload_backgrounds=data_config.preload_backgrounds,
        transforms_by_family=tile_transforms_by_family,
        sign_transforms_by_family=sign_transforms_by_family,
    )
    if data_config.test_dir is None:
        logger.warning("No test directory found, skipping final evaluation")
        test_dataset = None
    else:
        test_dataset = ClassificationDataset(
            data_root=Path(data_config.root) / data_config.test_dir,
            transforms=val_transforms,
            extensions=data_config.extensions,
            max_samples=data_config.max_samples,
            samples_per_class=data_config.samples_per_class,
            backgrounds_dir=Path(data_config.backgrounds_dir),
            background_sample_stride=data_config.background_sample_stride,
            preload_backgrounds=data_config.preload_backgrounds,
            transforms_by_family=tile_transforms_by_family,
            sign_transforms_by_family=sign_transforms_by_family,
        )

    train_loader, val_loader, test_loader = create_data_loaders(
        config.train,
        train_dataset,
        val_dataset,
        test_dataset,
    )

    data_distribution = train_dataset.get_data_distribution()
    logger.info(
        "Data distribution (%d classes; per-class counts at DEBUG)",
        len(data_distribution),
    )
    for data_dist in data_distribution:
        logger.debug(
            "\t%s: %d (%.2f%%)",
            data_dist.class_name,
            data_dist.count,
            data_dist.percentage,
        )

    if train_loader is None or val_loader is None:
        logger.error("No train or val loader found")
        raise DataLoaderException(
            "Cannot continue training without train and val data loaders"
        )
    if test_loader is None:
        logger.warning("No test loader found, skipping final evaluation")

    logger.info("Datasets and loaders ready.")
    logger.info("\tTrain samples: %d", len(train_dataset))
    logger.info("\tVal samples: %d", len(val_dataset))
    if test_dataset is not None:
        logger.info("\tTest samples: %d", len(test_dataset))

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------
    logger.info("Creating model %s", config.model.name)
    logger.info(
        "Model created — total parameters: %d",
        sum(p.numel() for p in model.parameters()),
    )

    # ---------------------------------------------------------------------
    # Optimizer & scheduler
    # ---------------------------------------------------------------------
    logger.info("Setting up optimizer and scheduler…")
    optimizer = get_optimizer(config.train.optimizer, model.parameters())
    scheduler = get_scheduler(config.train.scheduler, optimizer)

    # ---------------------------------------------------------------------
    # Loss function
    # ---------------------------------------------------------------------
    criterion = get_loss_function(config.train.loss_function)

    # ---------------------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------------------
    trainer = ClassificationTrainer(
        model_manager=classification_model_manager,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=config,
    )

    # ---------------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------------
    logger.info("Starting training for %d epochs", config.train.epochs)
    trainer.train()


if __name__ == "__main__":
    import cProfile
    import pstats

    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()
    stats = pstats.Stats(pr).sort_stats("cumtime")  # Sort po kumulatywnym czasie
    stats.print_stats(20)  # Top 20 funkcji
    stats.dump_stats("profile.prof")
