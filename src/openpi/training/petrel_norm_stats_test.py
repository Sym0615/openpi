import numpy as np

from openpi.training import petrel_norm_stats


def test_reservoir_rows_per_group_is_independent_of_the_worker_count():
    """Splitting the budget across workers starves groups that live in one shard."""
    assert petrel_norm_stats.reservoir_rows_per_group(65536, 8) == 8192
    assert petrel_norm_stats.reservoir_rows_per_group(1024, 8) == 1024


def test_reservoir_is_allocated_only_once_a_group_is_observed():
    accumulator = petrel_norm_stats._Accumulator(dim=3, reservoir_rows=8, seed=1)  # noqa: SLF001

    assert accumulator.reservoir is None

    accumulator.update(np.ones((2, 3), dtype=np.float32))

    assert accumulator.reservoir.shape == (8, 3)
    assert accumulator.reservoir_count == 2


def test_merging_an_untouched_worker_accumulator_is_a_no_op():
    merged = petrel_norm_stats._Accumulator(dim=2, reservoir_rows=4, seed=1)  # noqa: SLF001
    merged.update(np.zeros((3, 2), dtype=np.float32))

    merged.merge(petrel_norm_stats._Accumulator(dim=2, reservoir_rows=4, seed=2))  # noqa: SLF001

    assert merged.reservoir_count == 3
    assert merged.count == 3


def test_single_shard_group_keeps_the_full_reservoir_through_a_merge():
    capacity = petrel_norm_stats.reservoir_rows_per_group(1024, 1)
    worker = petrel_norm_stats._Accumulator(dim=2, reservoir_rows=capacity, seed=1)  # noqa: SLF001
    worker.update(np.arange(4000, dtype=np.float32).reshape(2000, 2))
    merged = petrel_norm_stats._Accumulator(dim=2, reservoir_rows=capacity, seed=2)  # noqa: SLF001

    merged.merge(worker)

    assert merged.reservoir_count == capacity
    assert merged.finalize().q01.shape == (2,)


def test_action_reservoir_keeps_the_horizon_axis():
    accumulator = petrel_norm_stats._HorizonAccumulator(  # noqa: SLF001
        horizon=2,
        dim=3,
        reservoir_rows=8,
        seed=1,
    )

    accumulator.update(np.ones((5, 2, 3), dtype=np.float32))

    assert accumulator.reservoir.shape == (4, 2, 3)
    assert accumulator.finalize().q99.shape == (2, 3)
