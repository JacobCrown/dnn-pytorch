"""Hydra entry point for image segmentation training."""

import logging

import hydra
from omegaconf import OmegaConf

from src.common.experiment_paths import segmentation_experiment_root
from src.common.hydra_resolvers import register_hydra_resolvers
from src.common.logging import configure_run_logging
from src.common.utils import DEVICE, set_seed
from src.segmentation.config import SegmentationConfig
from src.segmentation.training_pipeline import run_segmentation_training

register_hydra_resolvers()

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../configs/segmentation",
    config_name="main.yaml",
)
def main(cfg: dict) -> None:
    config = SegmentationConfig.from_dict(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)  # type: ignore[arg-type]
    )
    experiment_dir = segmentation_experiment_root(
        train_root_dir=config.train.root_dir,
        data_root=config.data.root,
        architecture=config.model.architecture,
        encoder_name=config.model.encoder_name,
        height=config.model.height,
        width=config.model.width,
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
    ).resolve()
    run_log_path = configure_run_logging(
        run_directory=experiment_dir,
        use_rich_console=True,
        quiet_third_party_loggers=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("Full run log (all levels): %s", run_log_path)
    logger.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    set_seed()
    logger.info("Device: %s", DEVICE)

    run_segmentation_training(config)


if __name__ == "__main__":
    main()
