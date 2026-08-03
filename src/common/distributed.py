"""Distributed training strategy helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from src.common.config import DistributedTrainingConfig


@dataclass(frozen=True)
class DistributedState:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> DistributedState:
        source = os.environ if env is None else env
        return cls(
            rank=int(source.get("RANK", "0")),
            local_rank=int(source.get("LOCAL_RANK", "0")),
            world_size=int(source.get("WORLD_SIZE", "1")),
        )

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


class TrainingStrategy:
    def __init__(
        self,
        *,
        config: DistributedTrainingConfig,
        state: DistributedState,
    ) -> None:
        self.config = config
        self.state = state

    @property
    def is_main_process(self) -> bool:
        return self.state.is_main_process

    def build_sampler(
        self,
        dataset: Dataset,
        *,
        shuffle: bool,
    ) -> DistributedSampler | None:
        return None


class DDPTrainingStrategy(TrainingStrategy):
    def build_sampler(
        self,
        dataset: Dataset,
        *,
        shuffle: bool,
    ) -> DistributedSampler:
        return DistributedSampler(
            dataset,
            num_replicas=self.state.world_size,
            rank=self.state.rank,
            shuffle=shuffle,
        )

    def wrap_model(self, model):
        return DistributedDataParallel(
            model,
            device_ids=[self.state.local_rank],
            find_unused_parameters=self.config.find_unused_parameters,
        )


def build_training_strategy(
    config: DistributedTrainingConfig,
    *,
    state: DistributedState | None = None,
) -> TrainingStrategy:
    resolved_state = DistributedState.from_env() if state is None else state
    if config.strategy == "single":
        return TrainingStrategy(config=config, state=resolved_state)
    if config.strategy == "ddp":
        return DDPTrainingStrategy(config=config, state=resolved_state)
    raise ValueError(f"Unsupported distributed strategy: {config.strategy}")
