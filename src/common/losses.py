import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from src.common.config import LossFunctionConfig


class CombinedLoss(nn.Module):
    """Combines multiple loss functions with optional weights.

    Args:
        losses: List of (loss_function, weight) tuples.
    """

    def __init__(self, losses: list[tuple[nn.Module, float]]):
        super().__init__()
        self.losses = nn.ModuleList([loss for loss, _ in losses])
        self.weights = [weight for _, weight in losses]

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the combined weighted loss."""
        total_loss = torch.tensor(0.0, device=input.device, dtype=input.dtype)
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss = total_loss + weight * loss_fn(input, target)
        return total_loss

    def __repr__(self) -> str:
        loss_strs = [
            f"{loss.__class__.__name__} (weight={weight})"
            for loss, weight in zip(self.losses, self.weights)
        ]
        return f"CombinedLoss([{', '.join(loss_strs)}])"


def _create_single_loss(config: LossFunctionConfig) -> nn.Module:
    """Create a single loss function from config."""
    try:
        return getattr(smp.losses, config.name)(**config.params)
    except AttributeError as e:
        raise ValueError(f"Unsupported loss function: {config.name}") from e


def get_loss_function(
    config: LossFunctionConfig | list[LossFunctionConfig],
) -> nn.Module:
    """Get loss function(s) from config.

    Args:
        config: Single LossFunctionConfig or list of LossFunctionConfig objects.
                When a list is provided, losses are combined with their weights.

    Returns:
        A single nn.Module (either a single loss or CombinedLoss).
    """
    # Handle single config (backward compatible)
    if isinstance(config, LossFunctionConfig):
        return _create_single_loss(config)

    # Handle list of configs
    if not config:
        raise ValueError("Loss function config list cannot be empty")

    # If only one loss in list, return it directly
    if len(config) == 1:
        return _create_single_loss(config[0])

    # Create combined loss
    losses = [(_create_single_loss(cfg), cfg.weight) for cfg in config]
    return CombinedLoss(losses)
