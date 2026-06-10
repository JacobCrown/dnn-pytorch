"""Test-time evaluation entrypoint for semantic segmentation."""

import logging
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.common.augmentations import get_augmentations
from src.common.hydra_resolvers import register_hydra_resolvers
from src.common.logging import configure_run_logging
from src.common.losses import get_loss_function
from src.common.utils import DEVICE, set_seed
from src.segmentation.config import SegmentationTestConfig
from src.segmentation.dataset import SegmentationDataset
from src.segmentation.model_manager import SegmentationModelManager
from src.segmentation.trainer import evaluate_segmentation_model

register_hydra_resolvers()

logger = logging.getLogger(__name__)


def _resolve_checkpoint_path(checkpoint_path: str) -> Path:
    resolved_path = Path(to_absolute_path(checkpoint_path)).resolve()
    logger.info("Resolved evaluation checkpoint: %s", resolved_path)
    return resolved_path


def _build_test_dataset(
    config: SegmentationTestConfig,
    eval_transforms,
) -> SegmentationDataset:
    if config.data.test_dir is None:
        raise ValueError("Segmentation test config requires data.test_dir to be set")

    return SegmentationDataset(
        data_root=Path(config.data.root) / config.data.test_dir,
        transforms=eval_transforms,
        class_config=config.data.class_config,
        class_mapping=config.data.class_mapping,
        extensions=config.data.extensions,
        mask_extensions=config.data.mask_extensions,
        max_samples=config.data.max_samples,
    )


def _create_test_loader(
    config: SegmentationTestConfig,
    test_dataset: SegmentationDataset,
) -> DataLoader:
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty — cannot run segmentation evaluation.")

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
        **common_kwargs,
    )


@hydra.main(
    version_base=None,
    config_path="../../configs/segmentation",
    config_name="test.yaml",
)
def main(cfg: dict) -> None:
    experiment_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    configure_run_logging(experiment_directory=experiment_dir)
    config = SegmentationTestConfig.from_dict(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)  # type: ignore[arg-type]
    )
    logger.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    set_seed()
    logger.info("Device: %s", DEVICE)

    if config.eval_checkpoint is None:
        raise ValueError("eval_checkpoint must be set for segmentation test runs")

    checkpoint_path = _resolve_checkpoint_path(config.eval_checkpoint)
    checkpoint = torch.load(
        str(checkpoint_path), map_location=DEVICE, weights_only=False
    )
    checkpoint_class_names = checkpoint.get("class_names")
    if not checkpoint_class_names:
        raise ValueError("Checkpoint does not contain class_names")

    model_manager = SegmentationModelManager.from_checkpoint_path(checkpoint_path)
    eval_transforms = get_augmentations(config.model.preprocessing)
    test_dataset = _build_test_dataset(config, eval_transforms)
    if test_dataset.get_class_names() != checkpoint_class_names:
        raise ValueError(
            "Test dataset class names do not match checkpoint class names. "
            f"Checkpoint: {checkpoint_class_names}; dataset: {test_dataset.get_class_names()}"
        )

    test_loader = _create_test_loader(config, test_dataset)
    criterion = get_loss_function(config.train.loss_function)

    logger.info("Starting test evaluation using checkpoint %s", checkpoint_path)
    evaluate_segmentation_model(
        model=model_manager.model,
        data_loader=test_loader,
        criterion=criterion,
        use_amp=config.train.use_amp,
        num_classes=config.model.classes,
        split_name="Test",
    )


if __name__ == "__main__":
    main()
