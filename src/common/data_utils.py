import logging
from collections.abc import Sized
from typing import cast

from torch.utils.data import DataLoader, Dataset

from src.common.config import BaseTrainingConfig

logger = logging.getLogger(__name__)


def create_data_loaders(
    config: BaseTrainingConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset | None = None,
    *,
    allow_empty_val_test: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Create data loaders for training, validation, and test sets.
    Args:
        config: Training configuration.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        allow_empty_val_test: If True, skip building loaders for val/test splits
            with zero samples (returns None) instead of failing in DataLoader.
    Returns:
        A tuple containing the training, validation, and test data loaders.
    """
    if len(cast(Sized, train_dataset)) == 0:
        raise ValueError(
            "Training dataset is empty — add images or fix data paths before training."
        )

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
        if allow_empty_val_test and len(cast(Sized, test_dataset)) == 0:
            logger.warning(
                "Test dataset is empty — final test evaluation will be skipped."
            )
        else:
            test_loader = DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                drop_last=False,
                **common_kwargs,
            )

    return train_loader, val_loader, test_loader
