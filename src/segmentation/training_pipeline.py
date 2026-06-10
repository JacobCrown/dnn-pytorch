"""Orchestration for segmentation training (datasets -> trainer -> checkpoints)."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import albumentations as A
from torch.utils.data import DataLoader
from torchinfo import summary

from src.common.augmentations import get_augmentations
from src.common.data_utils import create_data_loaders
from src.common.exceptions import DataLoaderException
from src.common.experiment_paths import segmentation_experiment_root
from src.common.experiment_registry import write_run_record
from src.common.logging import configure_run_logging
from src.common.losses import get_loss_function
from src.common.optimizers import get_optimizer
from src.common.schedulers import get_scheduler
from src.common.tracking import build_tracker
from src.common.training_contracts import TrainingArtifacts
from src.segmentation.config import SegmentationConfig, SegmentationModelConfig
from src.segmentation.dataset import SegmentationDataset
from src.segmentation.model_manager import SegmentationModelManager
from src.segmentation.trainer import SegmentationTrainer

logger = logging.getLogger(__name__)


class ClassNamedDataset(Protocol):
    def get_class_names(self) -> list[str]: ...


@dataclass
class SegmentationDataPipeline:
    train_dataset: SegmentationDataset
    val_dataset: SegmentationDataset
    test_dataset: SegmentationDataset | None
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader | None


@dataclass(frozen=True)
class SegmentationTrainingMetrics:
    """Metric history collected by ``SegmentationTrainer``."""

    losses: dict[str, list[float]]
    ious: dict[str, list[float]]
    best_val_loss: float | None
    best_val_iou: float | None
    final_train_loss: float | None
    final_train_iou: float | None
    final_val_loss: float | None
    final_val_iou: float | None
    test_loss: float | None
    test_iou: float | None


@dataclass(frozen=True)
class SegmentationTrainingOutcome:
    """Paths and metrics after a completed segmentation training run."""

    artifacts: TrainingArtifacts
    metrics: SegmentationTrainingMetrics
    used_validation: bool

    @property
    def checkpoints_dir(self) -> Path:
        return self.artifacts.checkpoints_dir

    @property
    def best_checkpoint_path(self) -> Path | None:
        return self.artifacts.best_checkpoint_path


def apply_classes_from_train_dataset(
    config: SegmentationConfig,
    train_dataset: ClassNamedDataset,
) -> int:
    """Set ``config.model.classes`` from the resolved training label space."""
    derived = len(train_dataset.get_class_names())
    config.model.classes = derived
    logger.info("Derived segmentation classes=%d from training dataset", derived)
    return derived


def build_segmentation_data_pipeline(
    config: SegmentationConfig,
    data_preproc: A.Compose,
) -> SegmentationDataPipeline:
    logger.info("Building augmentation and preprocessing pipelines...")

    train_augs = get_augmentations(config.augmentations)
    train_transforms = A.Compose(train_augs.transforms + data_preproc.transforms)
    logger.debug("Train transforms: %s", train_transforms)

    eval_transforms = A.Compose(data_preproc.transforms)
    logger.debug("Eval transforms: %s", eval_transforms)

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
        transforms=eval_transforms,
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
            transforms=eval_transforms,
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

    return SegmentationDataPipeline(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )


def log_data_pipeline_summary(data_pipeline: SegmentationDataPipeline) -> None:
    train_dataset = data_pipeline.train_dataset
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

    if data_pipeline.test_loader is None:
        logger.warning("No test loader found, skipping final evaluation")

    logger.info("Datasets and loaders ready.")
    logger.info("\tTrain samples: %d", len(train_dataset))
    logger.info("\tVal samples: %d", len(data_pipeline.val_dataset))
    if data_pipeline.test_dataset is not None:
        logger.info("\tTest samples: %d", len(data_pipeline.test_dataset))


def log_model_summary(
    model_manager: SegmentationModelManager,
    model_cfg: SegmentationModelConfig,
) -> None:
    model = model_manager.model
    logger.info(
        "Creating model %s with encoder %s",
        model_cfg.architecture,
        model_cfg.encoder_name,
    )
    logger.info(
        "Model created - total parameters: %d",
        sum(p.numel() for p in model.parameters()),
    )
    logger.debug(
        "Model summary:\n%s",
        summary(
            model,
            input_size=(
                4,
                model_cfg.in_channels,
                model_cfg.height,
                model_cfg.width,
            ),
            col_names=("input_size", "output_size", "num_params", "trainable"),
            verbose=0,
        ),
    )


def _last(values: list[float]) -> float | None:
    return float(values[-1]) if values else None


def _best_iou_for_min_loss(losses: list[float], ious: list[float]) -> float | None:
    if not losses or not ious:
        return None
    best_idx = min(range(len(losses)), key=losses.__getitem__)
    if best_idx >= len(ious):
        return None
    return float(ious[best_idx])


def _collect_training_metrics(
    trainer: SegmentationTrainer,
) -> SegmentationTrainingMetrics:
    losses = {
        split: [float(value) for value in values]
        for split, values in trainer.metrics.losses.items()
    }
    ious = {
        split: [float(value) for value in values]
        for split, values in trainer.metrics.ious.items()
    }
    val_losses = losses.get("val", [])
    val_ious = ious.get("val", [])
    return SegmentationTrainingMetrics(
        losses=losses,
        ious=ious,
        best_val_loss=min(val_losses) if val_losses else None,
        best_val_iou=_best_iou_for_min_loss(val_losses, val_ious),
        final_train_loss=_last(losses.get("train", [])),
        final_train_iou=_last(ious.get("train", [])),
        final_val_loss=_last(val_losses),
        final_val_iou=_last(val_ious),
        test_loss=_last(losses.get("test", [])),
        test_iou=_last(ious.get("test", [])),
    )


def run_segmentation_training(
    config: SegmentationConfig,
) -> SegmentationTrainingOutcome:
    """Wire datasets, model, optimizer and run training. Callable without Hydra."""
    configure_run_logging(
        experiment_directory=segmentation_experiment_root(
            train_root_dir=config.train.root_dir,
            data_root=config.data.root,
            architecture=config.model.architecture,
            encoder_name=config.model.encoder_name,
            height=config.model.height,
            width=config.model.width,
            epochs=config.train.epochs,
            batch_size=config.train.batch_size,
        ),
    )
    data_preproc = get_augmentations(config.model.preprocessing)
    data_pipeline = build_segmentation_data_pipeline(config, data_preproc)
    apply_classes_from_train_dataset(config, data_pipeline.train_dataset)
    log_data_pipeline_summary(data_pipeline)

    segmentation_model_manager = SegmentationModelManager(config.model)
    log_model_summary(segmentation_model_manager, config.model)

    logger.info("Setting up optimizer and scheduler...")
    optimizer = get_optimizer(config.train.optimizer, segmentation_model_manager.model.parameters())
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    criterion = get_loss_function(config.train.loss_function)

    trainer = SegmentationTrainer(
        model_manager=segmentation_model_manager,
        train_loader=data_pipeline.train_loader,
        val_loader=data_pipeline.val_loader,
        test_loader=data_pipeline.test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=config,
    )

    logger.info("Starting training for %d epochs", config.train.epochs)
    trainer.train()

    outcome = SegmentationTrainingOutcome(
        artifacts=TrainingArtifacts.from_completed_run(
            trainer.root_dir,
            checkpoints_dir_name=config.train.checkpoints_dir,
            logs_dir_name=config.train.logs_dir,
            visualizations_dir_name=config.train.visualize_dir,
        ),
        metrics=_collect_training_metrics(trainer),
        used_validation=True,
    )
    write_run_record(
        task_name="segmentation",
        artifacts=outcome.artifacts,
        metrics=outcome.metrics,
        config=config.to_dict(),
        used_validation=outcome.used_validation,
    )
    tracker = build_tracker(
        config=config.train.tracking,
        run_dir=outcome.artifacts.root_dir,
        task_name="segmentation",
    )
    tracker.log_metrics(asdict(outcome.metrics))
    if outcome.best_checkpoint_path is not None:
        tracker.log_artifact("best_checkpoint", outcome.best_checkpoint_path)
    tracker.close()
    return outcome
