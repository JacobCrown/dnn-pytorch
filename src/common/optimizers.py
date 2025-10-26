from collections.abc import Iterable

import torch

from src.common.config import OptimizerConfig


def get_optimizer(
    config: OptimizerConfig, params: Iterable[torch.nn.Parameter]
) -> torch.optim.Optimizer:
    """Get optimizer from config."""
    try:
        return getattr(torch.optim, config.name)(params, **config.params)
    except AttributeError as e:
        raise ValueError(f"Unsupported optimizer: {config.name}") from e
