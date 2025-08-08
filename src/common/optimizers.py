from src.common.config import OptimizerConfig
import torch


def get_optimizer(config: OptimizerConfig) -> torch.optim.Optimizer:
    """Get optimizer from config."""
    try:
        return getattr(torch.optim, config.name)(**config.params)
    except AttributeError as e:
        raise ValueError(f"Unsupported optimizer: {config.name}") from e
