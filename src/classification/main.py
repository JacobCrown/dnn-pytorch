"""Hydra entry point for image classification training."""

import logging

import hydra
from omegaconf import OmegaConf

from src.classification.config import ClassificationConfig
from src.classification.training_pipeline import run_classification_training
from src.common.experiment_paths import classification_experiment_root
from src.common.hydra_resolvers import register_hydra_resolvers
from src.common.logging import configure_run_logging
from src.common.utils import DEVICE, set_seed

register_hydra_resolvers()


@hydra.main(
    version_base=None,
    config_path="../../configs/classification",
    config_name="main.yaml",
)
def main(cfg: dict) -> None:
    config = ClassificationConfig.from_dict(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)  # type: ignore[arg-type]
    )
    experiment_dir = classification_experiment_root(
        train_root_dir=config.train.root_dir,
        data_root=config.data.root,
        model_name=config.model.name,
        height=config.model.height,
        width=config.model.width,
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
    ).resolve()
    run_log_path = configure_run_logging(
        experiment_directory=experiment_dir,
        use_rich_console=True,
        quiet_third_party_loggers=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("Full run log (all levels): %s", run_log_path)
    logger.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    set_seed()
    logger.info("Device: %s", DEVICE)

    run_classification_training(config)


if __name__ == "__main__":
    main()
