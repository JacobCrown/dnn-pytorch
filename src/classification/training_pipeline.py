"""Orchestration for classification training (datasets → trainer → checkpoints)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
from torch.utils.data import DataLoader
from torchinfo import summary

from src.classification.config import (
    ClassificationConfig,
    ClassificationModelConfig,
)
from src.classification.dataset import ClassificationDataset
from src.classification.model_manager import ClassificationModelManager
from src.classification.trainer import ClassificationTrainer
from src.common.augmentations import get_augmentations
from src.common.data_utils import create_data_loaders
from src.common.experiment_paths import classification_experiment_root
from src.common.losses import get_loss_function
from src.common.optimizers import get_optimizer
from src.common.schedulers import get_scheduler

logger = logging.getLogger(__name__)


@dataclass
class ClassificationDataPipeline:
    train_dataset: ClassificationDataset
    train_loader: DataLoader
    val_dataset: ClassificationDataset
    val_loader: DataLoader
    test_dataset: ClassificationDataset | None
    test_loader: DataLoader | None


def apply_num_classes_from_train_dataset(
    config: ClassificationConfig, train_dataset: ClassificationDataset
) -> int:
    """Set ``config.model.num_classes`` from the resolved training label space."""
    derived = len(train_dataset.get_class_names())
    config.model.num_classes = derived
    logger.info("Derived num_classes=%d from training dataset", derived)
    return derived


def build_classification_data_pipeline(
    config: ClassificationConfig,
    data_preproc: A.Compose,
) -> ClassificationDataPipeline:
    logger.info("Building augmentation and preprocessing pipelines...")

    train_augs = get_augmentations(config.augmentations)
    train_transforms = A.Compose(train_augs.transforms + data_preproc.transforms)
    logger.debug("Train transforms: %s", train_transforms)

    eval_transforms = A.Compose(data_preproc.transforms)
    logger.debug("Eval transforms: %s", eval_transforms)

    data_config = config.data
    train_dataset = ClassificationDataset(
        data_root=Path(data_config.root) / data_config.train_dir,
        transforms=train_transforms,
        extensions=data_config.extensions,
        max_samples=data_config.max_samples,
        class_mapping=data_config.class_mapping,
    )
    canonical = train_dataset.get_class_to_idx()
    val_dataset = ClassificationDataset(
        data_root=Path(data_config.root) / data_config.val_dir,
        transforms=eval_transforms,
        extensions=data_config.extensions,
        max_samples=data_config.max_samples,
        class_mapping=data_config.class_mapping,
        canonical_class_to_idx=canonical,
    )
    if data_config.test_dir is None:
        logger.warning("No test directory found, skipping final evaluation")
        test_dataset = None
    else:
        test_dataset = ClassificationDataset(
            data_root=Path(data_config.root) / data_config.test_dir,
            transforms=eval_transforms,
            extensions=data_config.extensions,
            max_samples=data_config.max_samples,
            class_mapping=data_config.class_mapping,
            canonical_class_to_idx=canonical,
        )

    train_loader, val_loader, test_loader = create_data_loaders(
        config.train,
        train_dataset,
        val_dataset,
        test_dataset,
        allow_empty_val_test=True,
    )

    return ClassificationDataPipeline(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )


def log_data_pipeline_summary(
    config: ClassificationConfig,
    data_pipeline: ClassificationDataPipeline,
) -> None:
    data_config = config.data
    train_dataset = data_pipeline.train_dataset

    source_class_count = len(
        [
            d
            for d in (Path(data_config.root) / data_config.train_dir).iterdir()
            if d.is_dir()
        ]
    )
    derived_num_classes = apply_num_classes_from_train_dataset(config, train_dataset)
    train_class_names = train_dataset.get_class_names()
    logger.info(
        "Classification labels resolved: %d source folders -> %d final classes",
        source_class_count,
        derived_num_classes,
    )
    logger.info(
        "Final class names: %d classes (full list at DEBUG)", len(train_class_names)
    )
    logger.debug("Final class names: %s", train_class_names)

    if data_config.class_mapping:
        for mapping in data_config.class_mapping:
            logger.debug("Mapped %s -> %s", mapping.map_classes, mapping.name)

    data_distribution = train_dataset.get_data_distribution()
    logger.info(
        "Data distribution: %d classes (per-class counts at DEBUG)",
        len(data_distribution),
    )
    for data_dist in data_distribution:
        logger.debug(
            "\t%s: %d (%.2f%%)",
            data_dist.class_name,
            data_dist.count,
            data_dist.percentage,
        )

    if data_pipeline.test_loader is None:
        logger.warning("No test loader found, skipping final evaluation")

    logger.info("Datasets and loaders ready.")
    logger.info("\tTrain samples: %d", len(train_dataset))
    logger.info("\tVal samples: %d", len(data_pipeline.val_dataset))
    if data_pipeline.test_dataset is not None:
        logger.info("\tTest samples: %d", len(data_pipeline.test_dataset))


def log_model_summary(
    model_manager: ClassificationModelManager,
    model_cfg: ClassificationModelConfig,
) -> None:
    model = model_manager.model
    logger.info("Creating model %s", model_cfg.name)
    logger.info(
        "Model created — total parameters: %d",
        sum(p.numel() for p in model.parameters()),
    )
    logger.info(
        "Trainable parameters: %d",
        sum(p.numel() for p in model_manager.get_trainable_parameters()),
    )
    logger.debug(
        "Model summary:\n%s",
        summary(
            model,
            input_size=(
                1,
                model_cfg.in_channels,
                model_cfg.height,
                model_cfg.width,
            ),
            col_names=("input_size", "output_size", "num_params", "trainable"),
            verbose=0,
        ),
    )


def run_classification_training(
    config: ClassificationConfig,
) -> None:
    """Wire datasets, model, optimizer and run training. Callable without Hydra."""
    run_directory=classification_experiment_root(
        config=config,
    )
    data_preproc = get_augmentations(config.model.preprocessing)
    data_pipeline = build_classification_data_pipeline(config, data_preproc)
    log_data_pipeline_summary(config, data_pipeline)

    if config.model.num_classes is None:
        raise RuntimeError(
            "num_classes was not set after building the training dataset; this is a bug."
        )

    classification_model_manager = ClassificationModelManager(config.model)
    log_model_summary(classification_model_manager, config.model)

    logger.info("Setting up optimizer and scheduler...")
    optimizer = get_optimizer(
        config.train.optimizer,
        classification_model_manager.get_trainable_parameters(),
    )
    scheduler = get_scheduler(config.train.scheduler, optimizer)

    criterion = get_loss_function(config.train.loss_function)

    trainer = ClassificationTrainer(
        model_manager=classification_model_manager,
        train_loader=data_pipeline.train_loader,
        val_loader=data_pipeline.val_loader,
        test_loader=data_pipeline.test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=config,
        run_directory=run_directory,
    )

    logger.info("Starting training for %d epochs", config.train.epochs)

    trainer.train()
