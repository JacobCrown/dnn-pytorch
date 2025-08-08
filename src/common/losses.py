from src.common.config import LossFunctionConfig
import torch.nn as nn
import torch.nn.functional as F


def get_loss_function(config: LossFunctionConfig) -> nn.Module:
    """Get loss function from config."""
    try:
        return getattr(F, config.name)(**config.params)
    except AttributeError as e:
        raise ValueError(f"Unsupported loss function: {config.name}") from e
