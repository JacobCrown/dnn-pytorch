"""Main training pipeline for image classification with Hydra configuration."""
# pyright: reportMissingImports=false

import logging
import os
from pathlib import Path

import albumentations as A
import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.common.losses import get_loss_function
from src.common.optimizers import get_optimizer
from src.common.schedulers import get_scheduler
from src.classification.config import ClassificationConfig
from src.common.augmentations import get_augmentations
from src.common.model_loader import get_model
from src.common.logging import setup_logging
from src.classification.dataset import ClassificationDataset
from src.classification.trainer import ClassificationTrainer
from src.common.utils import (
    create_data_loaders,
    set_seed,
    plot_training_history,
    plot_confusion_matrix,
)
from src.common.exceptions import DataLoaderException
from sklearn.metrics import confusion_matrix

# Note: real log file will be set inside main() after Hydra changes cwd
logger = setup_logging(__file__)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="classification/main.yaml",
)
def main(cfg: DictConfig) -> None:
    # Load config
    # aug_cfg = cfg.get("classification").get("augmentations", [])
    # for aug in aug_cfg:
    #     print(aug)
    root_cfg = cfg.classification
    config = ClassificationConfig.from_dict(root_cfg)
    logging.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # ---------------------------------------------------------------------
    # Seed & device
    # ---------------------------------------------------------------------
    # Experiment block optional; default seed=42
    seed_val = 42
    set_seed(seed_val)
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
    )
    val_dataset = ClassificationDataset(
        data_root / root_cfg.data.val_dir,
        transforms=val_transforms,
        extensions=root_cfg.data.extensions,
    )
    try:
        test_dataset = ClassificationDataset(
            data_root / root_cfg.data.test_dir,
            transforms=val_transforms,
            extensions=root_cfg.data.extensions,
        )
    except FileNotFoundError:
        logger.warning("No test directory found, skipping test dataset")
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
    optimizer = get_optimizer(config.train.optimizer)
    scheduler = get_scheduler(config.train.scheduler)

    # ---------------------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------------------
    # Loss function
    criterion = get_loss_function(config.train.loss_function)

    # Pyright sometimes narrows scheduler to ReduceLROnPlateau which is not a subclass of _LRScheduler
    # Our trainer supports both types, so type: ignore on this argument
    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,  # type: ignore[arg-type]
        cfg=root_cfg,
        train_cfg=train_cfg,
        device=device,
        save_dir=str(Path(os.getcwd()) / "checkpoints"),
        log_dir=str(log_dir),
    )

    # Resume from checkpoint if provided
    resume_path = getattr(train_cfg, "resume", None)
    if resume_path:
        logging.info("Resuming training from %s", resume_path)
        trainer.load_checkpoint(resume_path)

    # ---------------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------------
    logging.info("Starting training for %d epochs", num_epochs)
    history = trainer.train(
        num_epochs=num_epochs,
        save_freq=int(getattr(train_cfg, "save_freq", 1)),
        early_stopping_patience=getattr(train_cfg, "early_stopping_patience", None),
    )

    # ---------------------------------------------------------------------
    # Final evaluation
    # ---------------------------------------------------------------------
    best_ckpt = Path(os.getcwd()) / "checkpoints" / "best.pth"
    if best_ckpt.exists():
        logging.info("Loading best checkpoint for final validation evaluation")
        trainer.load_checkpoint(str(best_ckpt))
        val_loss, val_acc = trainer.validate()
        logging.info(
            "Validation after training — loss: %.4f, acc: %.2f%%", val_loss, val_acc
        )

    # Visualizations: training curves and confusion matrix on validation set
    plots_dir = Path(os.getcwd()) / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_training_history(history, save_path=str(plots_dir / "training_curves.png"))

    # Confusion matrix using validation set
    assert val_loader is not None, (
        "Validation loader must not be None to plot confusion matrix"
    )
    y_true, y_pred = trainer.predict_labels(val_loader)
    cm = confusion_matrix(y_true, y_pred)
    class_names = val_dataset.get_class_names()
    plot_confusion_matrix(
        cm, class_names, save_path=str(plots_dir / "confusion_matrix.png")
    )


if __name__ == "__main__":
    main()
