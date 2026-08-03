import torch
from torch.utils.data import TensorDataset

from src.common.config import DistributedTrainingConfig
from src.common.distributed import DistributedState, build_training_strategy


def test_distributed_state_defaults_to_single_process() -> None:
    state = DistributedState.from_env({})

    assert state.rank == 0
    assert state.local_rank == 0
    assert state.world_size == 1
    assert state.is_distributed is False
    assert state.is_main_process is True


def test_build_training_strategy_uses_single_process_by_default() -> None:
    strategy = build_training_strategy(DistributedTrainingConfig())

    assert strategy.state.world_size == 1
    assert strategy.is_main_process is True


def test_ddp_strategy_builds_distributed_sampler() -> None:
    strategy = build_training_strategy(
        DistributedTrainingConfig(strategy="ddp"),
        state=DistributedState(rank=1, local_rank=1, world_size=2),
    )
    dataset = TensorDataset(torch.arange(8))

    sampler = strategy.build_sampler(dataset, shuffle=True)

    assert sampler is not None
    assert sampler.num_replicas == 2
    assert sampler.rank == 1
