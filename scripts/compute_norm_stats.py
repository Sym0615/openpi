"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses
import pathlib

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.petrel_norm_stats as petrel_norm_stats
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = not bool(getattr(dataset, "prefer_sequential_indices", False))
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
        framework="pytorch",
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_data_config(config: _config.TrainConfig) -> _config.DataConfig:
    """Create the pre-model data config needed by normalization statistics."""
    data_factory = config.data
    if isinstance(data_factory, _config.PetrelDataConfig):
        # Norm stats only consume the pre-model state/actions. Building the
        # normal PI0.5 model transforms would instantiate the PaliGemma
        # tokenizer and unnecessarily download its SentencePiece model.
        data_factory = dataclasses.replace(
            data_factory,
            model_transforms=_config.IdentityTransformFactory(),
        )
    return data_factory.create(config.assets_dirs, config.model)


def main(
    config_name: str,
    max_frames: int | None = None,
    petrel_num_workers: int = 32,
    petrel_episode_batch_size: int = 4096,
    petrel_reservoir_rows: int = 200000,
    petrel_local_root: str | None = None,
    output_dir: str | None = None,
):
    config = _config.get_config(config_name)
    data_config = create_data_config(config)
    if data_config.petrel is not None:
        if data_config.repo_id is None:
            raise ValueError("Petrel data config must have a repo_id.")
        norm_stats = petrel_norm_stats.compute_fast_petrel_norm_stats(
            data_config.petrel.dataset_config,
            action_horizon=config.model.action_horizon,
            output_dim=config.model.action_dim,
            max_frames=max_frames,
            num_workers=petrel_num_workers,
            episode_batch_size=petrel_episode_batch_size,
            reservoir_rows=petrel_reservoir_rows,
            local_root=petrel_local_root or data_config.petrel.local_root,
        )
        assets_dir = config.assets_dirs if output_dir is None else pathlib.Path(output_dir)
        output_path = assets_dir / data_config.repo_id
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats)
        return

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(feature_shape=(config.model.action_horizon, config.model.action_dim)),
    }

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    assets_dir = config.assets_dirs if output_dir is None else pathlib.Path(output_dir)
    output_path = assets_dir / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
