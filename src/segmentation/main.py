"""Hydra entry point for image segmentation training."""

import hydra
from omegaconf import OmegaConf

from src.common.experiment_paths import segmentation_experiment_root
from src.common.hydra_resolvers import register_hydra_resolvers
from src.common.logging import configure_run_logging, run_train_info_log
from src.common.paths import get_config_path
from src.common.utils import set_seed
from src.segmentation.config import SegmentationConfig
from src.segmentation.training_pipeline import run_segmentation_training

register_hydra_resolvers()


@hydra.main(
    version_base=None,
    config_path=str(get_config_path() / "segmentation"),
    config_name="main.yaml",
)
def main(cfg: dict) -> None:
    config = SegmentationConfig.from_dict(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)  # type: ignore[arg-type]
    )
    experiment_dir = segmentation_experiment_root(config=config).resolve()
    run_log_path = configure_run_logging(
        run_directory=experiment_dir,
        use_rich_console=True,
        quiet_third_party_loggers=True,
    )
    resolved_cfg = OmegaConf.to_yaml(cfg, resolve=True)

    run_train_info_log("Segmentation", run_log_path, resolved_cfg)
    set_seed()
    run_segmentation_training(config)


if __name__ == "__main__":
    main()
