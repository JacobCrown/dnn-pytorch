import torch.nn as nn

from src.common.config import LossFunctionConfig


def get_loss_function(config: LossFunctionConfig) -> nn.Module:
    """Get loss function from config."""
    try:
        return getattr(nn, config.name)(**config.params)
    except AttributeError as e:
        raise ValueError(f"Unsupported loss function: {config.name}") from e
