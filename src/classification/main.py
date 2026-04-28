"""Main training pipeline for image classification."""

from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchinfo import summary

from src.classification.config import ClassificationConfig, ClassificationModelConfig
from src.classification.dataset import ClassificationDataset
from src.classification.model_manager import ClassificationModelManager
from src.classification.trainer import ClassificationTrainer
from src.common.augmentations import get_augmentations
from src.common.data_utils import create_data_loaders
from src.common.logging import setup_logging
from src.common.losses import get_loss_function
from src.common.optimizers import get_optimizer
from src.common.schedulers import get_scheduler
from src.common.utils import DEVICE, set_seed

# Note: real log file will be set inside main() after Hydra changes cwd
logger = setup_logging(__file__)


@dataclass
class ClassificationDataPipeline:
    train_dataset: ClassificationDataset
    val_dataset: ClassificationDataset
    test_dataset: ClassificationDataset | None
    train_loader: DataLoader
    val_loader: DataLoader | None
    test_loader: DataLoader | None


def _sync_num_classes_with_dataset(
    config: ClassificationConfig, train_dataset: ClassificationDataset
) -> int:
    """Update model.num_classes from the resolved training labels."""
    derived_num_classes = len(train_dataset.get_class_names())
    if derived_num_classes != config.model.num_classes:
        logger.warning(
            "Overriding model.num_classes from %d to %d based on training dataset",
            config.model.num_classes,
            derived_num_classes,
        )
    config.model.num_classes = derived_num_classes
    return derived_num_classes


def _build_classification_data_pipeline(
    config: ClassificationConfig,
    data_preproc: A.Compose,
) -> ClassificationDataPipeline:
    logger.info("Building augmentation and preprocessing pipelines...")

    train_augs = get_augmentations(config.augmentations)
    train_transforms = A.Compose(train_augs.transforms + data_preproc.transforms)
    logger.info("Train transforms: %s", train_transforms)

    eval_transforms = A.Compose(data_preproc.transforms)
    logger.info("Eval transforms: %s", eval_transforms)

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


def _log_data_pipeline_summary(
    config: ClassificationConfig,
    data_pipeline: ClassificationDataPipeline,
) -> list[str]:
    data_config = config.data
    train_dataset = data_pipeline.train_dataset

    source_class_count = len(
        [
            d
            for d in (Path(data_config.root) / data_config.train_dir).iterdir()
            if d.is_dir()
        ]
    )
    derived_num_classes = _sync_num_classes_with_dataset(config, train_dataset)
    train_class_names = train_dataset.get_class_names()
    logger.info(
        "Classification labels resolved: %d source folders -> %d final classes",
        source_class_count,
        derived_num_classes,
    )
    logger.info("Final class names: %s", train_class_names)

    if data_config.class_mapping:
        for mapping in data_config.class_mapping:
            logger.info("Mapped %s -> %s", mapping.map_classes, mapping.name)

    data_distribution = train_dataset.get_data_distribution()
    logger.info("Data distribution:")
    for data_dist in data_distribution:
        logger.info(
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

    return train_class_names


def _log_model_summary(
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
    logger.info(
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


@hydra.main(
    version_base=None,
    config_path="../../configs/classification",
    config_name="main.yaml",
)
def main(cfg: dict) -> None:
    config = ClassificationConfig.from_dict(dict(cfg))
    logger.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    set_seed()
    logger.info("Device: %s", DEVICE)

    data_preproc = get_augmentations(config.model.preprocessing)
    data_pipeline = _build_classification_data_pipeline(config, data_preproc)
    _log_data_pipeline_summary(config, data_pipeline)

    classification_model_manager = ClassificationModelManager(config.model)
    _log_model_summary(classification_model_manager, config.model)

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
    )

    logger.info("Starting training for %d epochs", config.train.epochs)
    trainer.train()


if __name__ == "__main__":
    main()
