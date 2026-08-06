from collections.abc import Iterator, Sequence
import dataclasses
import logging
import multiprocessing
import os
import time
import typing
from typing import Any, Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
from openpi.training import profiling as _profiling
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
from openpi.training.petrel_dataset import PetrelDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset  # PetrelDataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        started = time.perf_counter()
        item = self._dataset[index]
        _profiling.record("data.access_and_preload", time.perf_counter() - started)
        started = time.perf_counter()
        result = self._transform(item)
        _profiling.record("data.preprocessing", time.perf_counter() - started)
        return result

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitems__(self, indices: list[int]) -> list[T_co]:
        getitems = getattr(self._dataset, "__getitems__", None)
        started = time.perf_counter()
        items = getitems(indices) if callable(getitems) else [self._dataset[index] for index in indices]
        _profiling.record("data.access_and_preload", time.perf_counter() - started)
        started = time.perf_counter()
        result = [self._transform(item) for item in items]
        _profiling.record("data.preprocessing", time.perf_counter() - started)
        return result

    def set_epoch(self, epoch: int) -> None:
        set_epoch = getattr(self._dataset, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)

    @property
    def prefer_sequential_indices(self) -> bool:
        return bool(getattr(self._dataset, "prefer_sequential_indices", False))

    @property
    def requires_worker_restart_each_epoch(self) -> bool:
        return bool(getattr(self._dataset, "requires_worker_restart_each_epoch", False))

    @property
    def training_block_size(self) -> int:
        return int(getattr(self._dataset, "training_block_size", 1))

    @property
    def frame_filter_identity(self) -> str | None:
        return getattr(self._dataset, "frame_filter_identity", None)

    @property
    def frame_filter_summary(self) -> dict[str, Any]:
        return dict(getattr(self._dataset, "frame_filter_summary", {"enabled": False}))


class ContiguousDistributedSampler(torch.utils.data.Sampler[int]):
    """Give each DDP rank one contiguous shard so episode-local index blocks stay intact."""

    def __init__(
        self,
        dataset,
        *,
        num_replicas: int,
        rank: int,
        block_size: int = 1,
        drop_last: bool = True,
    ):
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank {rank} is outside [0, {num_replicas})")
        self._dataset = dataset
        self._num_replicas = int(num_replicas)
        self._rank = int(rank)
        if not drop_last:
            raise ValueError("ContiguousDistributedSampler currently requires drop_last=True")
        block_size = max(1, int(block_size))
        blocks_per_rank = (len(dataset) // block_size) // self._num_replicas
        self._num_samples = blocks_per_rank * block_size

    def __iter__(self):
        start = self._rank * self._num_samples
        return iter(range(start, start + self._num_samples))

    def __len__(self) -> int:
        return self._num_samples

    def set_epoch(self, epoch: int) -> None:
        del epoch


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    if data_config.petrel is not None:  # TrainConfig.data.dataset
        return PetrelDataset(
            data_config.petrel,
            action_horizon=action_horizon,
            model_action_dim=model_config.action_dim,
        )

    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.NormalizeByGroup(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.NormalizeByGroup(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    # DataConfig()
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        worker_batch_size=config.dataloader_worker_batch_size,
        prefetch_factor=config.dataloader_prefetch_factor,
        in_order=config.dataloader_in_order,
        pin_memory=config.dataloader_pin_memory,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    worker_batch_size: int | None = None,
    prefetch_factor: int | None = None,
    in_order: bool = True,
    pin_memory: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    if getattr(dataset, "prefer_sequential_indices", False) and shuffle:
        logging.info("Disabling outer shuffle to preserve Petrel block sampling locality.")
        shuffle = False

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            if getattr(dataset, "prefer_sequential_indices", False) and not shuffle:
                sampler = ContiguousDistributedSampler(
                    dataset,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=torch.distributed.get_rank(),
                    block_size=getattr(dataset, "training_block_size", 1),
                    drop_last=True,
                )
                logging.info("Using contiguous DDP shards to preserve Petrel episode blocks.")
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=torch.distributed.get_rank(),
                    shuffle=shuffle,
                    drop_last=True,
                )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
        worker_batch_size=worker_batch_size,
        prefetch_factor=prefetch_factor,
        in_order=in_order,
        pin_memory=pin_memory,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
        worker_batch_size: int | None = None,
        prefetch_factor: int | None = None,
        in_order: bool = True,
        pin_memory: bool = False,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if framework not in {"jax", "pytorch"}:
            raise ValueError(f"Unsupported data loader framework: {framework!r}")
        if framework == "jax" and jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple JAX processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        worker_batch_size = local_batch_size if worker_batch_size is None else int(worker_batch_size)
        if worker_batch_size <= 0:
            raise ValueError(f"Worker batch size must be positive, got {worker_batch_size}.")
        if local_batch_size % worker_batch_size != 0:
            raise ValueError(
                f"Local batch size ({local_batch_size}) must be divisible by worker batch size ({worker_batch_size})."
            )
        if prefetch_factor is not None and prefetch_factor <= 0:
            raise ValueError(f"prefetch_factor must be positive when provided, got {prefetch_factor}.")
        self._local_batch_size = local_batch_size
        self._worker_batch_size = worker_batch_size

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        self._dataset = dataset  #  TransformedDataset
        self._sampler = sampler

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        persistent_workers = num_workers > 0 and not bool(
            getattr(dataset, "requires_worker_restart_each_epoch", False)
        )
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=worker_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=persistent_workers,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            in_order=in_order,
            pin_memory=pin_memory,
        )
        logging.info(
            "DataLoader worker_batch_size=%d local_batch_size=%d micro_batches_per_step=%d "
            "workers=%d persistent_workers=%s prefetch_factor=%s in_order=%s pin_memory=%s",
            worker_batch_size,
            local_batch_size,
            local_batch_size // worker_batch_size,
            num_workers,
            persistent_workers,
            prefetch_factor,
            in_order,
            pin_memory,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    @property
    def frame_filter_identity(self) -> str | None:
        return getattr(self._dataset, "frame_filter_identity", None)

    @property
    def frame_filter_summary(self) -> dict[str, Any]:
        return dict(getattr(self._dataset, "frame_filter_summary", {"enabled": False}))

    def __iter__(self):
        num_items = 0
        epoch = 0
        while True:
            set_epoch = getattr(self._dataset, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch)
            sampler_set_epoch = getattr(self._sampler, "set_epoch", None)
            if callable(sampler_set_epoch):
                sampler_set_epoch(epoch)
            data_iter = iter(self._data_loader)
            pending_batches = []
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                if isinstance(batch, _ProfiledBatch):
                    _profiling.merge_snapshot(batch.metrics)
                    batch = batch.batch
                pending_batches.append(batch)
                if len(pending_batches) * self._worker_batch_size < self._local_batch_size:
                    continue
                if len(pending_batches) > 1:
                    started = time.perf_counter()
                    batch = jax.tree.map(lambda *xs: np.concatenate(xs, axis=0), *pending_batches)
                    _profiling.record("dataloader.microbatch_concat", time.perf_counter() - started)
                pending_batches.clear()
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)
            epoch += 1


@dataclasses.dataclass
class _ProfiledBatch:
    batch: typing.Any
    metrics: dict[str, dict[str, float | int]]


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    started = time.perf_counter()
    batch = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)
    if not _profiling.enabled():
        return batch
    _profiling.record("dataloader.collate", time.perf_counter() - started)
    return _ProfiledBatch(batch=batch, metrics=_profiling.snapshot(reset=True))


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader  # TorchDataLoader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    @property
    def frame_filter_identity(self) -> str | None:
        return getattr(self._data_loader, "frame_filter_identity", None)

    @property
    def frame_filter_summary(self) -> dict[str, Any]:
        return dict(getattr(self._data_loader, "frame_filter_summary", {"enabled": False}))

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
