"""Hydra entry point for image classification training."""

import hydra
from omegaconf import OmegaConf

from src.classification.config import ClassificationConfig
from src.classification.training_pipeline import run_classification_training
from src.common.experiment_paths import classification_experiment_root
from src.common.hydra_resolvers import register_hydra_resolvers
from src.common.logging import configure_run_logging, run_train_info_log
from src.common.utils import set_seed
from src.common.paths import get_config_path

register_hydra_resolvers()


@hydra.main(
    version_base=None,
    config_path=str(get_config_path() / "classification"),
    config_name="main.yaml",
)
def main(cfg: dict) -> None:
    config = ClassificationConfig.from_dict(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)  # type: ignore[arg-type]
    )
    experiment_dir = classification_experiment_root(config=config).resolve()
    run_log_path = configure_run_logging(
        run_directory=experiment_dir,
        use_rich_console=True,
        quiet_third_party_loggers=True,
    )
    resolved_cfg = OmegaConf.to_yaml(cfg, resolve=True)

    run_train_info_log("Classification", run_log_path, resolved_cfg)
    set_seed()
    run_classification_training(config)


if __name__ == "__main__":
    main()
