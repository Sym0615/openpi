import dataclasses

import jax
import numpy as np

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import profiling as _profiling


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_reassembles_worker_micro_batches():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=8,
        worker_batch_size=2,
        num_batches=2,
        num_workers=0,
        in_order=False,
    )
    batches = list(loader)

    assert len(batches) == 2
    assert all(all(x.shape[0] == 8 for x in jax.tree.leaves(batch)) for batch in batches)


def test_contiguous_distributed_sampler_preserves_blocks():
    dataset = np.arange(40)
    rank0 = _data_loader.ContiguousDistributedSampler(dataset, num_replicas=2, rank=0)
    rank1 = _data_loader.ContiguousDistributedSampler(dataset, num_replicas=2, rank=1)

    assert list(rank0) == list(range(20))
    assert list(rank1) == list(range(20, 40))


def test_torch_data_loader_parallel_returns_worker_profiling():
    was_enabled = _profiling.enabled()
    _profiling.set_enabled(enabled=True)
    _profiling.snapshot(reset=True)
    try:
        config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
        dataset = _data_loader.FakeDataset(config, 10)
        loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=1, num_workers=2)

        _ = next(iter(loader))
        events = _profiling.snapshot(reset=True)

        assert events["dataloader.collate"]["count"] == 1
        assert events["dataloader.collate"]["total_ms"] >= 0
    finally:
        _profiling.snapshot(reset=True)
        _profiling.set_enabled(enabled=was_enabled)


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
