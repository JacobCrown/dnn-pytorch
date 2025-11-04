"""Main training pipeline for image segmentation with Hydra configuration."""

import logging
from pathlib import Path

import albumentations as A
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary

from src.segmentation.config import SegmentationConfig
from src.segmentation.dataset import SegmentationDataset
from src.segmentation.trainer import SegmentationTrainer
from src.common.augmentations import get_augmentations
from src.common.exceptions import DataLoaderException
from src.common.logging import setup_logging
from src.common.losses import get_loss_function
from src.segmentation.model_loader import get_model
from src.common.optimizers import get_optimizer
from src.common.schedulers import get_scheduler
from src.common.utils import set_seed
from src.common.data_utils import create_data_loaders

# Note: real log file will be set inside main() after Hydra changes cwd
logger = setup_logging(__file__)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="segmentation/main.yaml",
)
def main(cfg: DictConfig) -> None:
    # Load config
    config = SegmentationConfig.from_dict(cfg.segmentation)
    logging.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # ---------------------------------------------------------------------
    # Seed & device
    # ---------------------------------------------------------------------
    # Experiment block optional; default seed=42
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)


    model, data_preproc = get_model(config.model)
    model.to(device)
    logger.info(
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
    train_dataset = SegmentationDataset(
        data_root=Path(data_config.root) / data_config.train_dir,
        transforms=train_transforms,
        class_config=data_config.class_config,
        extensions=data_config.extensions,
        mask_extensions=data_config.mask_extensions,
        max_samples=data_config.max_samples,
    )
    val_dataset = SegmentationDataset(
        data_root=Path(data_config.root) / data_config.val_dir,
        transforms=val_transforms,
        class_config=data_config.class_config,
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

    logging.info("Datasets and loaders ready.")
    logger.info("\tTrain samples: %d", len(train_dataset))
    logger.info("\tVal samples: %d", len(val_dataset))
    if test_dataset is not None:
        logger.info("\tTest samples: %d", len(test_dataset))

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------
    logging.info(
        "Creating model %s with encoder %s",
        config.model.architecture,
        config.model.params["encoder_name"],
    )
    logging.info(
        "Model created — total params: %d", sum(p.numel() for p in model.parameters())
    )

    # ---------------------------------------------------------------------
    # Optimizer & scheduler
    # ---------------------------------------------------------------------
    logging.info("Setting up optimizer and scheduler…")
    optimizer = get_optimizer(config.train.optimizer, model.parameters())
    scheduler = get_scheduler(config.train.scheduler, optimizer)

    # ---------------------------------------------------------------------
    # Loss function
    # ---------------------------------------------------------------------
    criterion = get_loss_function(config.train.loss_function)

    # ---------------------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------------------

    num_classes = config.model.classes
    trainer = SegmentationTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=config,
        num_classes=num_classes,
        test_loader=test_loader,
        device=device,
    )

    # ---------------------------------------------------------------------
    # Resume from checkpoint if provided
    # ---------------------------------------------------------------------
    resume_path = config.train.resume
    if resume_path:
        logging.info("Resuming training from %s", resume_path)
        trainer.load_checkpoint(resume_path)

    # ---------------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------------
    logging.info("Starting training for %d epochs", config.train.epochs)
    trainer.train()

    # ---------------------------------------------------------------------
    # Visualizations: training curves and predictions on validation set
    # ---------------------------------------------------------------------
    logging.info("Generating visualizations…")
    trainer.visualize()


if __name__ == "__main__":
    main()
