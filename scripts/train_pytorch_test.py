import threading
from types import SimpleNamespace

import pytest
import torch

from scripts import train_pytorch


def test_training_optimizations_default_to_disabled(monkeypatch):
    monkeypatch.delenv("OPENPI_BF16_AUTOCAST", raising=False)
    monkeypatch.delenv("OPENPI_COMPILE_TRANSFORMER_LAYERS", raising=False)
    monkeypatch.delenv("OPENPI_GRADIENT_CHECKPOINTING", raising=False)

    assert train_pytorch.training_optimization_config() == {
        "bf16_autocast": False,
        "compile_transformer_layers": False,
        "gradient_checkpointing": False,
    }


def test_training_optimizations_enable_stable_path(monkeypatch):
    monkeypatch.setenv("OPENPI_BF16_AUTOCAST", "1")
    monkeypatch.setenv("OPENPI_COMPILE_TRANSFORMER_LAYERS", "true")
    monkeypatch.setenv("OPENPI_GRADIENT_CHECKPOINTING", "on")

    assert train_pytorch.training_optimization_config() == {
        "bf16_autocast": True,
        "compile_transformer_layers": True,
        "gradient_checkpointing": True,
    }


def test_zero_optimizer_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("OPENPI_ZERO_OPTIMIZER", raising=False)
    assert train_pytorch.zero_optimizer_enabled() is False


def test_zero_optimizer_requires_multiple_ranks():
    with pytest.raises(ValueError, match="requires at least two distributed ranks"):
        train_pytorch.validate_zero_optimizer_configuration(enabled=True, world_size=1)


def test_zero_optimizer_accepts_multi_rank_ddp():
    train_pytorch.validate_zero_optimizer_configuration(enabled=True, world_size=8)


def _mixed_dtype_parameters():
    torch.manual_seed(0)
    # Interleave the dtypes so a wrapper that assumed contiguous buckets would fail.
    return [
        torch.nn.Parameter(torch.randn(2, 2, dtype=torch.bfloat16)),
        torch.nn.Parameter(torch.randn(2, 2)),
        torch.nn.Parameter(torch.randn(2, dtype=torch.bfloat16)),
    ]


def _stepped_adamw(parameters):
    optimizer = torch.optim.AdamW(parameters, lr=0.1)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    return optimizer


def _dtype_partitioned_optimizer(parameters):
    """Stand in for the real wrapper; consolidated ZeRO shards use this layout."""
    parameters_by_dtype = {}
    global_indices = {}
    for index, parameter in enumerate(parameters):
        parameters_by_dtype.setdefault(str(parameter.dtype), []).append(parameter)
        global_indices.setdefault(str(parameter.dtype), []).append(index)
    return train_pytorch.DtypePartitionedZeroOptimizer(
        {dtype: torch.optim.AdamW(group, lr=0.1) for dtype, group in parameters_by_dtype.items()},
        global_indices,
    )


def test_zero_optimizer_writes_a_checkpoint_plain_adamw_can_resume():
    parameters = _mixed_dtype_parameters()
    zero = _dtype_partitioned_optimizer(parameters)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    zero.step()

    state_dict = zero.state_dict()
    assert state_dict["param_groups"][0]["params"] == [0, 1, 2]

    resumed = torch.optim.AdamW(parameters, lr=0.9)
    resumed.load_state_dict(state_dict)

    assert resumed.param_groups[0]["lr"] == 0.1
    for parameter in parameters:
        torch.testing.assert_close(
            resumed.state[parameter]["exp_avg"],
            zero.optimizers_by_dtype[str(parameter.dtype)].state[parameter]["exp_avg"],
        )


def test_zero_optimizer_resumes_from_a_plain_adamw_checkpoint():
    parameters = _mixed_dtype_parameters()
    reference = _stepped_adamw(parameters)
    reference_state = reference.state_dict()

    zero = _dtype_partitioned_optimizer(parameters)
    zero.load_state_dict(reference_state)

    for parameter in parameters:
        loaded = zero.optimizers_by_dtype[str(parameter.dtype)].state[parameter]
        torch.testing.assert_close(loaded["exp_avg"], reference.state[parameter]["exp_avg"])
    # Round-tripping back out reproduces the plain layout it came from.
    assert zero.state_dict()["param_groups"][0]["params"] == reference_state["param_groups"][0]["params"]


def test_zero_optimizer_param_groups_stay_live_after_a_resume():
    """The LR schedule writes through optim.param_groups on every step.

    torch.optim.Optimizer.load_state_dict replaces its group dicts, so a list
    captured at construction would silently freeze the schedule after --resume.
    """
    parameters = _mixed_dtype_parameters()
    zero = _dtype_partitioned_optimizer(parameters)
    zero.load_state_dict(_stepped_adamw(parameters).state_dict())

    for param_group in zero.param_groups:
        param_group["lr"] = 0.042

    assert [group["lr"] for optimizer in zero.optimizers_by_dtype.values() for group in optimizer.param_groups] == [
        0.042,
        0.042,
    ]


def test_zero_optimizer_checkpoint_consolidates_on_all_ranks():
    optimizer = _dtype_partitioned_optimizer(_mixed_dtype_parameters())
    consolidated_to = []
    for shard in optimizer.optimizers_by_dtype.values():
        shard.consolidate_state_dict = lambda *, to, _sink=consolidated_to: _sink.append(to)

    train_pytorch.save_checkpoint(
        torch.nn.Linear(2, 2),
        optimizer,
        global_step=10,
        config=SimpleNamespace(save_interval=10, num_train_steps=20),
        is_main=False,
        data_config=SimpleNamespace(),
    )

    assert consolidated_to == [0] * len(optimizer.optimizers_by_dtype)


def test_camera_preview_covers_every_image_slot_of_the_training_batch(monkeypatch):
    """The preview replaces a second DataLoader, so it must work from a real batch."""
    logged = {}
    monkeypatch.setattr(train_pytorch.wandb, "Image", lambda array: array)
    monkeypatch.setattr(train_pytorch.wandb, "log", lambda payload, step: logged.update(payload=payload, step=step))
    observation = SimpleNamespace(
        images={
            "base_0_rgb": torch.zeros(3, 3, 2, 4),
            "left_wrist_0_rgb": torch.ones(3, 3, 2, 4),
        }
    )

    train_pytorch.log_camera_views(observation, step=0, max_samples=2)

    assert logged["step"] == 0
    previews = logged["payload"]["camera_views"]
    assert len(previews) == 2
    # NCHW -> NHWC with both slots concatenated along width: 4 + 4.
    assert previews[0].shape == (2, 8, 3)


def test_camera_preview_never_exceeds_the_batch_size(monkeypatch):
    logged = {}
    monkeypatch.setattr(train_pytorch.wandb, "Image", lambda array: array)
    monkeypatch.setattr(train_pytorch.wandb, "log", lambda payload, step: logged.update(payload=payload, step=step))
    observation = SimpleNamespace(images={"base_0_rgb": torch.zeros(1, 3, 2, 4)})

    train_pytorch.log_camera_views(observation, step=7)

    assert len(logged["payload"]["camera_views"]) == 1
    assert logged["step"] == 7


def test_gradient_accumulation_suppresses_sync_until_last_microbatch():
    class FakeDistributedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.no_sync_calls = 0

        def no_sync(self):
            self.no_sync_calls += 1
            return train_pytorch.nullcontext("no_sync")

    model = FakeDistributedModel()
    markers = []
    for micro_step in range(4):
        with train_pytorch.gradient_accumulation_context(model, micro_step, 4) as marker:
            markers.append(marker)

    assert markers == ["no_sync", "no_sync", "no_sync", None]
    assert model.no_sync_calls == 3


def test_data_loader_fetches_effective_batch_once_per_optimizer_step():
    assert train_pytorch.data_loader_global_batch_size(96, 4) == 384


def test_streaming_data_loader_fetches_one_physical_batch_at_a_time():
    assert train_pytorch.data_loader_global_batch_size(96, 4, streaming_accumulation=True) == 96


def test_split_effective_batch_preserves_order_and_microbatch_size():
    observation = {
        "state": torch.arange(48 * 2).reshape(48, 2),
        "nested": (torch.arange(48),),
    }
    actions = torch.arange(48 * 3).reshape(48, 3)

    microbatches = list(train_pytorch.split_effective_batch(observation, actions, 12, 4))

    assert len(microbatches) == 4
    for index, (micro_observation, micro_actions) in enumerate(microbatches):
        start = index * 12
        stop = start + 12
        torch.testing.assert_close(micro_observation["state"], observation["state"][start:stop])
        torch.testing.assert_close(micro_observation["nested"][0], observation["nested"][0][start:stop])
        torch.testing.assert_close(micro_actions, actions[start:stop])


def test_split_effective_batch_rejects_wrong_size():
    with pytest.raises(ValueError, match="expected 48"):
        list(train_pytorch.split_effective_batch({"state": torch.zeros(47, 2)}, torch.zeros(47, 3), 12, 4))


def test_validate_streaming_microbatch_rejects_effective_batch():
    train_pytorch.validate_streaming_microbatch(torch.zeros(12, 3), 12)
    with pytest.raises(ValueError, match="expected physical batch 12"):
        train_pytorch.validate_streaming_microbatch(torch.zeros(48, 3), 12)


def test_async_data_iterator_prefetches_into_bounded_queue():
    producer_reached_third_item = threading.Event()

    def source():
        for value in range(4):
            if value == 2:
                producer_reached_third_item.set()
            yield value

    iterator = train_pytorch.AsyncDataIterator(source(), max_prefetch=2)
    try:
        assert producer_reached_third_item.wait(timeout=1.0)
        assert iterator.ready_batches == 2
        assert next(iterator) == 0
        assert next(iterator) == 1
    finally:
        iterator.close()

    assert iterator.closed


def test_async_data_iterator_propagates_producer_exception():
    def source():
        yield 1
        raise RuntimeError("data failed")

    iterator = train_pytorch.AsyncDataIterator(source(), max_prefetch=2)
    assert next(iterator) == 1
    with pytest.raises(RuntimeError, match="data failed"):
        next(iterator)
