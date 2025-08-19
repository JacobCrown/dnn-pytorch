"""Main training pipeline for image classification with Hydra configuration."""

import logging
from pathlib import Path

import albumentations as A
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.classification.config import ClassificationConfig
from src.classification.dataset import ClassificationDataset
from src.classification.trainer import ClassificationTrainer
from src.common.augmentations import get_augmentations
from src.common.exceptions import DataLoaderException
from src.common.logging import setup_logging
from src.common.losses import get_loss_function
from src.common.model_loader import get_model
from src.common.optimizers import get_optimizer
from src.common.schedulers import get_scheduler
from src.common.utils import set_seed
from src.common.data_utils import create_data_loaders

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
    logging.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # ---------------------------------------------------------------------
    # Seed & device
    # ---------------------------------------------------------------------
    # Experiment block optional; default seed=42
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, data_preproc = get_model(config.model)
    model.to(device)

    # ---------------------------------------------------------------------
    # Augmentations + Preprocessing
    # ---------------------------------------------------------------------
    logging.info("Building augmentation and preprocessing pipelines…")
    # Rebuild using augmentations but avoid duplicate ToTensorV2
    train_augs = get_augmentations(config.augmentations, append_to_tensor=False)

    train_transforms = A.Compose(train_augs.transforms + data_preproc.transforms)

    val_transforms = A.Compose(data_preproc.transforms)

    # ---------------------------------------------------------------------
    # Datasets & loaders
    # ---------------------------------------------------------------------
    data_root = Path(root_cfg.data.root)
    train_dataset = ClassificationDataset(
        data_root / root_cfg.data.train_dir,
        transforms=train_transforms,
        extensions=root_cfg.data.extensions,
        max_samples=root_cfg.data.max_samples,
    )
    val_dataset = ClassificationDataset(
        data_root / root_cfg.data.val_dir,
        transforms=val_transforms,
        extensions=root_cfg.data.extensions,
        max_samples=root_cfg.data.max_samples,
    )
    try:
        test_dataset = ClassificationDataset(
            data_root / root_cfg.data.test_dir,
            transforms=val_transforms,
            extensions=root_cfg.data.extensions,
            max_samples=root_cfg.data.max_samples,
        )
    except Exception as e:
        logger.error(f"Error creating test dataset: {e}")
        test_dataset = None

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
    logging.info("Creating model %s", root_cfg.model.name)
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

    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_cfg=config.train,
        root_dir="output",  # TODO: make this configurable
        device=device,
        save_dir_name="checkpoints",
        log_dir_name="logs",
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
    # Visualizations: training curves and confusion matrix on validation set
    # ---------------------------------------------------------------------
    logging.info("Visualizing metrics…")
    trainer.visualize_metrics()


if __name__ == "__main__":
    main()
