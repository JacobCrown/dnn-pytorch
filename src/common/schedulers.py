from src.common.config import SchedulerConfig
import torch.optim.lr_scheduler


def get_scheduler(config: SchedulerConfig) -> torch.optim.lr_scheduler._LRScheduler:
    """Get scheduler from config."""
    try:
        return getattr(torch.optim.lr_scheduler, config.name)(**config.params)
    except AttributeError as e:
        raise ValueError(f"Unsupported scheduler: {config.name}") from e
