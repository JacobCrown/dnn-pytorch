"""Orchestration for classification training (datasets → trainer → checkpoints)."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import albumentations as A
from torch.utils.data import DataLoader
from torchinfo import summary

from src.classification.config import (
    ClassificationConfig,
    ClassificationDataConfig,
    ClassificationModelConfig,
)
from src.classification.dataset import ClassificationDataset
from src.classification.model_manager import ClassificationModelManager
from src.classification.trainer import ClassificationTrainer
from src.common.augmentations import get_augmentations
from src.common.data_utils import create_data_loaders
from src.common.experiment_paths import classification_experiment_root
from src.common.experiment_registry import write_run_record
from src.common.logging import configure_run_logging
from src.common.losses import get_loss_function
from src.common.optimizers import get_optimizer
from src.common.schedulers import get_scheduler
from src.common.tracking import build_tracker
from src.common.training_contracts import TrainingArtifacts

logger = logging.getLogger(__name__)


@dataclass
class ClassificationDataPipeline:
    train_dataset: ClassificationDataset
    val_dataset: ClassificationDataset
    test_dataset: ClassificationDataset | None
    train_loader: DataLoader
    val_loader: DataLoader | None
    test_loader: DataLoader | None


@dataclass(frozen=True)
class ClassificationTrainingMetrics:
    """Metric history collected by ``ClassificationTrainer``."""

    losses: dict[str, list[float]]
    accuracies: dict[str, list[float]]
    best_val_loss: float | None
    best_val_accuracy: float | None
    final_train_loss: float | None
    final_train_accuracy: float | None
    final_val_loss: float | None
    final_val_accuracy: float | None
    test_loss: float | None
    test_accuracy: float | None


@dataclass(frozen=True)
class ClassificationTrainingOutcome:
    """Paths after a completed training run (Hydra cwd is typically outputs/...)."""

    artifacts: TrainingArtifacts
    metrics: ClassificationTrainingMetrics
    used_validation: bool

    @property
    def checkpoints_dir(self) -> Path:
        return self.artifacts.checkpoints_dir

    @property
    def best_checkpoint_path(self) -> Path | None:
        return self.artifacts.best_checkpoint_path

    @property
    def predictor_checkpoint_path(self) -> Path | None:
        return self.artifacts.predictor_checkpoint_path


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

    if data_pipeline.val_loader is None:
        logger.warning(
            "No validation data loader; training will not use a validation set"
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
    configure_run_logging(
        experiment_directory=classification_experiment_root(
            train_root_dir=config.train.root_dir,
            data_root=config.data.root,
            model_name=config.model.name,
            height=config.model.height,
            width=config.model.width,
            epochs=config.train.epochs,
            batch_size=config.train.batch_size,
        ),
        use_rich_console=True,
        quiet_third_party_loggers=True,
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

    experiment_root = classification_experiment_root(
        train_root_dir=config.train.root_dir,
        data_root=config.data.root,
        model_name=config.model.name,
        height=config.model.height,
        width=config.model.width,
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
    )
    tracker = build_tracker(
        config=config.train.tracking,
        run_dir=experiment_root,
        task_name="classification",
    )

    trainer = ClassificationTrainer(
        model_manager=classification_model_manager,
        train_loader=data_pipeline.train_loader,
        val_loader=data_pipeline.val_loader,
        test_loader=data_pipeline.test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=config,
        tracker=tracker,
    )

    logger.info("Starting training for %d epochs", config.train.epochs)

    trainer.train()

    outcome = ClassificationTrainingOutcome(
        artifacts=TrainingArtifacts.from_completed_run(
            trainer.root_dir,
            checkpoints_dir_name=config.train.checkpoints_dir,
            logs_dir_name=config.train.logs_dir,
            visualizations_dir_name=config.train.visualize_dir,
        ),
        metrics=_collect_training_metrics(trainer),
        used_validation=data_pipeline.val_loader is not None,
    )
    write_run_record(
        task_name="classification",
        artifacts=outcome.artifacts,
        metrics=outcome.metrics,
        config=config.to_dict(),
        used_validation=outcome.used_validation,
    )
    tracker.log_metrics(asdict(outcome.metrics))
    if outcome.best_checkpoint_path is not None:
        tracker.log_artifact("best_checkpoint", outcome.best_checkpoint_path)
    tracker.close()
    return outcome
