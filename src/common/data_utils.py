from torch.utils.data import DataLoader, Dataset

from src.common.config import BaseTrainingConfig


def create_data_loaders(
    config: BaseTrainingConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Create data loaders for training, validation, and test sets.
    Args:
        config: Training configuration.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
    Returns:
        A tuple containing the training, validation, and test data loaders.
    """
    common_kwargs: dict = {
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
    }
    if config.num_workers > 0:
        # Only valid when using worker processes
        common_kwargs["persistent_workers"] = config.persistent_workers
        if config.prefetch_factor is not None:
            common_kwargs["prefetch_factor"] = config.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        drop_last=config.drop_last,
        **common_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            **common_kwargs,
        )

    return train_loader, val_loader, test_loader
