"""Main training pipeline for image classification with Hydra configuration."""

from pathlib import Path

import albumentations as A
import hydra
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary

from src.classification.config import ClassificationConfig
from src.classification.dataset import ClassificationDataset
from src.classification.model_manager import ClassificationModelManager
from src.classification.trainer import ClassificationTrainer
from src.common.augmentations import get_augmentations
from src.common.data_utils import create_data_loaders
from src.common.exceptions import DataLoaderException
from src.common.logging import setup_logging
from src.common.losses import get_loss_function
from src.common.optimizers import get_optimizer
from src.common.schedulers import get_scheduler
from src.common.utils import DEVICE, set_seed

# Note: real log file will be set inside main() after Hydra changes cwd
logger = setup_logging(__file__)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="classification/main.yaml",
)
def main(cfg: DictConfig) -> None:
    # Load config
    root_cfg = cfg.classification
    config = ClassificationConfig.from_dict(root_cfg)
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

    logger.info(
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
    # Rebuild using augmentations but avoid duplicate ToTensorV2
    train_augs = get_augmentations(config.augmentations, append_to_tensor=False)

    train_transforms = A.Compose(train_augs.transforms + data_preproc.transforms)
    logger.info("Train transforms: %s", train_transforms)

    val_transforms = A.Compose(data_preproc.transforms)
    logger.info("Val transforms: %s", val_transforms)

    # ---------------------------------------------------------------------
    # Datasets & loaders
    # ---------------------------------------------------------------------
    data_config = config.data
    train_dataset = ClassificationDataset(
        data_root=Path(data_config.root) / data_config.train_dir,
        transforms=train_transforms,
        extensions=data_config.extensions,
        max_samples=data_config.max_samples,
    )
    val_dataset = ClassificationDataset(
        data_root=Path(data_config.root) / data_config.val_dir,
        transforms=val_transforms,
        extensions=data_config.extensions,
        max_samples=data_config.max_samples,
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
        )

    train_loader, val_loader, test_loader = create_data_loaders(
        config.train,
        train_dataset,
        val_dataset,
        test_dataset,
    )

    data_distribution = train_dataset.get_data_distribution()
    logger.info("Data distribution:")
    for data_dist in data_distribution:
        logger.info(
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
    logger.info("Creating model %s", root_cfg.model.name)
    logger.info(
        "Model created — total params: %d", sum(p.numel() for p in model.parameters())
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
    main()
