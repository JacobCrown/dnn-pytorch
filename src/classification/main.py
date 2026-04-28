"""Hydra entry point for image classification training."""

import hydra
from omegaconf import OmegaConf

from src.classification.config import ClassificationConfig
from src.classification.training_pipeline import run_classification_training
from src.common.logging import setup_logging
from src.common.utils import DEVICE, set_seed

logger = setup_logging(__file__)


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

    run_classification_training(config)


if __name__ == "__main__":
    main()
