"""Main training pipeline for image segmentation with Hydra configuration."""

import logging
from pathlib import Path

import albumentations as A
import hydra
from omegaconf import OmegaConf
from torchinfo import summary

from src.common.augmentations import get_augmentations
from src.common.data_utils import create_data_loaders
from src.common.exceptions import DataLoaderException
from src.common.experiment_paths import segmentation_experiment_root
from src.common.hydra_resolvers import register_hydra_resolvers
from src.common.logging import configure_run_logging
from src.common.losses import get_loss_function
from src.common.optimizers import get_optimizer
from src.common.schedulers import get_scheduler
from src.common.utils import DEVICE, set_seed
from src.segmentation.config import SegmentationConfig
from src.segmentation.dataset import SegmentationDataset
from src.segmentation.model_manager import SegmentationModelManager
from src.segmentation.trainer import SegmentationTrainer

register_hydra_resolvers()

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../configs/segmentation",
    config_name="main.yaml",
)
def main(cfg: dict) -> None:
    config = SegmentationConfig.from_dict(dict(cfg))
    experiment_dir = segmentation_experiment_root(
        train_root_dir=config.train.root_dir,
        data_root=config.data.root,
        architecture=config.model.architecture,
        encoder_name=config.model.encoder_name,
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
    segmentation_model_manager = SegmentationModelManager(config.model)
    model = segmentation_model_manager.model
    data_preproc = segmentation_model_manager.preprocessing

    logger.debug(
        "Model summary:\n%s",
        summary(
            model,
            input_size=(
                4,
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

    # ---------------------------------------------------------------------
    # Datasets & loaders
    # ---------------------------------------------------------------------
    data_config = config.data
    train_dataset = SegmentationDataset(
        data_root=Path(data_config.root) / data_config.train_dir,
        transforms=train_transforms,
        class_config=data_config.class_config,
        class_mapping=data_config.class_mapping,
        extensions=data_config.extensions,
        mask_extensions=data_config.mask_extensions,
        max_samples=data_config.max_samples,
    )
    val_dataset = SegmentationDataset(
        data_root=Path(data_config.root) / data_config.val_dir,
        transforms=val_transforms,
        class_config=data_config.class_config,
        class_mapping=data_config.class_mapping,
        extensions=data_config.extensions,
        mask_extensions=data_config.mask_extensions,
        max_samples=data_config.max_samples,
    )
    if data_config.test_dir is None:
        logger.warning("No test directory found, skipping final evaluation")
        test_dataset = None
    else:
        test_dataset = SegmentationDataset(
            data_root=Path(data_config.root) / data_config.test_dir,
            transforms=val_transforms,
            class_config=data_config.class_config,
            class_mapping=data_config.class_mapping,
            extensions=data_config.extensions,
            mask_extensions=data_config.mask_extensions,
            max_samples=data_config.max_samples,
        )

    train_loader, val_loader, test_loader = create_data_loaders(
        config.train,
        train_dataset,
        val_dataset,
        test_dataset,
    )

    if train_loader is None or val_loader is None:
        raise DataLoaderException(
            "Cannot continue training without train and val data loaders"
        )
    if test_loader is None:
        logger.warning("No test loader found, skipping final evaluation")

    data_distribution = train_dataset.get_data_distribution()
    logger.info(
        "Data distribution (%d classes; detail at DEBUG)", len(data_distribution)
    )
    for data_dist in data_distribution:
        logger.debug(
            "\t%s: %.2f%% of pixels",
            data_dist.class_name,
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
    logger.info(
        "Creating model %s with encoder %s",
        config.model.architecture,
        config.model.encoder_name,
    )
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

    trainer = SegmentationTrainer(
        model_manager=segmentation_model_manager,
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
    # import cProfile
    # import pstats

    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # stats = pstats.Stats(pr).sort_stats("tottime")  # Sort po kumulatywnym czasie
    # stats.print_stats(20)  # Top 20 funkcji
    # stats.dump_stats("profile.prof")
