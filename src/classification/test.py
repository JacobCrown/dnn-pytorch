"""Test-time evaluation entrypoint for image classification."""

import logging
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchinfo import summary

from src.classification.config import ClassificationTestConfig
from src.classification.dataset import ClassificationDataset
from src.classification.model_manager import ClassificationModelManager
from src.classification.trainer import evaluate_classification_model
from src.classification.visualizer import ClassificationVisualizer
from src.common.augmentations import get_augmentations
from src.common.hydra_resolvers import register_hydra_resolvers
from src.common.logging import configure_run_logging
from src.common.losses import get_loss_function
from src.common.utils import DEVICE, set_seed

register_hydra_resolvers()

logger = logging.getLogger(__name__)


def _resolve_checkpoint_path(checkpoint_path: str) -> Path:
    resolved_path = Path(to_absolute_path(checkpoint_path)).resolve()
    logger.info("Resolved evaluation checkpoint: %s", resolved_path)
    return resolved_path


def _validate_checkpoint_class_names(checkpoint: dict) -> list[str]:
    checkpoint_class_names = checkpoint.get("class_names")
    if checkpoint_class_names is None:
        raise ValueError("Checkpoint does not contain class_names")
    if not checkpoint_class_names:
        raise ValueError("Checkpoint contains an empty class_names list")
    return checkpoint_class_names


def _build_test_dataset(
    config: ClassificationTestConfig,
    class_names: list[str],
    data_preproc,
) -> ClassificationDataset:
    if config.data.test_dir is None:
        raise ValueError("Classification test config requires data.test_dir to be set")

    canonical_class_to_idx = {
        class_name: idx for idx, class_name in enumerate(class_names)
    }
    return ClassificationDataset(
        data_root=Path(config.data.root) / config.data.test_dir,
        transforms=data_preproc,
        extensions=config.data.extensions,
        max_samples=config.data.max_samples,
        class_mapping=config.data.class_mapping,
        canonical_class_to_idx=canonical_class_to_idx,
    )


def _create_test_loader(
    config: ClassificationTestConfig,
    test_dataset: ClassificationDataset,
) -> DataLoader:
    if len(test_dataset) == 0:
        raise ValueError(
            "Test dataset is empty — cannot run classification evaluation."
        )

    common_kwargs = {
        "num_workers": config.train.num_workers,
        "pin_memory": config.train.pin_memory,
    }
    if config.train.num_workers > 0:
        common_kwargs["persistent_workers"] = config.train.persistent_workers
        if config.train.prefetch_factor is not None:
            common_kwargs["prefetch_factor"] = config.train.prefetch_factor

    return DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )


def _log_test_dataset_summary(test_dataset: ClassificationDataset) -> None:
    logger.info("Test dataset ready.")
    logger.info("\tTest samples: %d", len(test_dataset))
    names = test_dataset.get_class_names()
    logger.info(
        "Final class names: %d classes (see log file for full list)", len(names)
    )
    logger.debug("Final class names: %s", names)

    data_distribution = test_dataset.get_data_distribution()
    logger.info(
        "Test data distribution (%d classes; per-class detail at DEBUG)",
        len(data_distribution),
    )
    for data_dist in data_distribution:
        logger.debug(
            "\t%s: %d (%.2f%%)",
            data_dist.class_name,
            data_dist.count,
            data_dist.percentage,
        )


def _log_model_summary(model_manager: ClassificationModelManager) -> None:
    model = model_manager.model
    model_cfg = model_manager.model_cfg
    logger.info("Loaded model %s", model_cfg.name)
    logger.info(
        "Model parameters: %d",
        sum(param.numel() for param in model.parameters()),
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


def _run_test_visualizations(
    config: ClassificationTestConfig,
    test_loader: DataLoader,
    model_manager: ClassificationModelManager,
) -> None:
    visualize_dir = Path(config.train.visualize_dir)
    visualize_dir.mkdir(parents=True, exist_ok=True)
    visualizer = ClassificationVisualizer(visualize_dir)

    logger.info("Saving test visualizations to %s", visualize_dir.resolve())
    visualizer.visualize_example_images(
        data_loader=test_loader,
        num_samples_per_class=5,
    )
    visualizer.visualize_data_distribution(data_loader=test_loader)
    visualizer.visualize_predictions(
        data_loader=test_loader,
        model=model_manager.model,
        use_amp=config.train.use_amp,
        num_samples_per_class=5,
    )


@hydra.main(
    version_base=None,
    config_path="../../configs/classification",
    config_name="test.yaml",
)
def main(cfg: dict) -> None:
    experiment_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_log_path = configure_run_logging(
        experiment_directory=experiment_dir,
        use_rich_console=True,
        quiet_third_party_loggers=True,
    )
    logger.info("Full run log (all levels): %s", run_log_path)
    config = ClassificationTestConfig.from_dict(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)  # type: ignore[arg-type]
    )
    logger.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    set_seed()
    logger.info("Device: %s", DEVICE)

    if config.eval_checkpoint is None:
        raise ValueError("eval_checkpoint must be set for classification test runs")

    checkpoint_path = _resolve_checkpoint_path(config.eval_checkpoint)
    checkpoint = torch.load(
        str(checkpoint_path), map_location=DEVICE, weights_only=False
    )
    checkpoint_class_names = _validate_checkpoint_class_names(checkpoint)

    model_manager = ClassificationModelManager.from_checkpoint_path(checkpoint_path)
    _log_model_summary(model_manager)

    data_preproc = get_augmentations(config.model.preprocessing)
    logger.debug("Test preprocessing from config: %s", data_preproc)

    test_dataset = _build_test_dataset(
        config,
        checkpoint_class_names,
        data_preproc,
    )
    if test_dataset.get_class_names() != checkpoint_class_names:
        raise ValueError(
            "Test dataset class names do not match checkpoint class names. "
            f"Checkpoint: {checkpoint_class_names}; dataset: {test_dataset.get_class_names()}"
        )

    _log_test_dataset_summary(test_dataset)
    test_loader = _create_test_loader(config, test_dataset)
    criterion = get_loss_function(config.train.loss_function)
    _run_test_visualizations(config, test_loader, model_manager)

    logger.info("Starting test evaluation using checkpoint %s", checkpoint_path)
    evaluate_classification_model(
        model=model_manager.model,
        data_loader=test_loader,
        criterion=criterion,
        use_amp=config.train.use_amp,
        class_names=checkpoint_class_names,
        split_name="Test",
    )


if __name__ == "__main__":
    main()
