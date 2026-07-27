"""Native openpi dataset for remote Petrel LeRobot v2/v3 data."""

# ruff: noqa: SLF001

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import dataclasses
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from openpi.training import petrel_lerobot_reader as reader

_OPENPI_CAMERA_NAMES = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_COMPACT_VALID_PROJECTION = "compact_valid"
_PI05_ALOHA_32_PROJECTION = "pi05_aloha_32"


def read_petrel_dataset_metadata(config_path: str) -> tuple[str, Mapping[str, Any]]:
    """Read the dataset id and openpi contract without accessing remote storage."""
    return reader.read_openpi_petrel_metadata(config_path)


def _dimension_padding(item: Mapping[str, Any], key: str, dimension: int) -> torch.Tensor:
    value = item.get(key)
    if value is None:
        return torch.zeros(dimension, dtype=torch.bool)
    padding = torch.as_tensor(value, dtype=torch.bool).flatten()
    if padding.numel() != dimension:
        raise ValueError(f"Expected {key} to have {dimension} entries, got {padding.numel()}.")
    return padding


def _pad_feature(value: torch.Tensor, output_dim: int) -> torch.Tensor:
    if value.shape[-1] > output_dim:
        raise ValueError(f"Projected feature dim {value.shape[-1]} exceeds openpi output_dim={output_dim}.")
    if value.shape[-1] == output_dim:
        return value
    padding = torch.zeros((*value.shape[:-1], output_dim - value.shape[-1]), dtype=value.dtype)
    return torch.cat((value, padding), dim=-1)


def _pi05_aloha_32_projection(
    state: torch.Tensor, actions: torch.Tensor, output_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map canonical controls to ALOHA 14D before the standard model transform pads to 32D."""
    if output_dim != 32:
        raise ValueError(f"pi05_aloha_32 requires output_dim=32, got {output_dim}.")
    if state.shape[-1] != 80:
        raise ValueError(f"pi05_aloha_32 requires the canonical 80D robot layout, got {state.shape[-1]}D.")

    projected_state = torch.zeros((*state.shape[:-1], 14), dtype=state.dtype, device=state.device)
    projected_actions = torch.zeros((*actions.shape[:-1], 14), dtype=actions.dtype, device=actions.device)

    # Official ALOHA order: left joint[6], left gripper, right joint[6], right gripper.
    component_slices = (
        (slice(0, 6), slice(0, 6)),
        (slice(16, 17), slice(6, 7)),
        (slice(40, 46), slice(7, 13)),
        (slice(56, 57), slice(13, 14)),
    )
    for source_slice, target_slice in component_slices:
        projected_state[..., target_slice] = state[..., source_slice]
        projected_actions[..., target_slice] = actions[..., source_slice]

    return projected_state, projected_actions


def projected_feature_dim(output_dim: int, projection: str) -> int:
    if projection == _PI05_ALOHA_32_PROJECTION:
        if output_dim != 32:
            raise ValueError(f"pi05_aloha_32 requires output_dim=32, got {output_dim}.")
        return 14
    return output_dim


def project_petrel_features(
    item: Mapping[str, Any], output_dim: int, *, projection: str = _COMPACT_VALID_PROJECTION
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project canonical Petrel tensors into OpenPI's fixed-width control space."""
    state = torch.as_tensor(item["observation.state"], dtype=torch.float32)
    actions = torch.as_tensor(item["action"], dtype=torch.float32)
    if state.shape[-1] != actions.shape[-1]:
        raise ValueError(
            "Petrel state/action dimensions must match before openpi projection: "
            f"state={state.shape[-1]}, action={actions.shape[-1]}."
        )
    dimension = int(state.shape[-1])
    state_padding = _dimension_padding(item, "observation.state_dim_is_pad", dimension)
    action_padding = _dimension_padding(item, "action_dim_is_pad", dimension)
    state = state.masked_fill(state_padding, 0.0)
    actions = actions.masked_fill(action_padding, 0.0)

    if projection == _PI05_ALOHA_32_PROJECTION:
        return _pi05_aloha_32_projection(state, actions, output_dim)
    if projection != _COMPACT_VALID_PROJECTION:
        raise ValueError(f"Unsupported Petrel control projection {projection!r}.")

    action_indices = torch.nonzero(~action_padding, as_tuple=False).flatten()
    if action_indices.numel() > output_dim:
        dataset_name = str(item.get("dataset_name", "unknown"))
        raise ValueError(
            f"Dataset {dataset_name!r} has {action_indices.numel()} active action dimensions, "
            f"which exceeds openpi action_dim={output_dim}."
        )
    state_only = torch.nonzero(~state_padding & action_padding, as_tuple=False).flatten()
    remaining = output_dim - int(action_indices.numel())
    indices = torch.cat((action_indices, state_only[:remaining]))
    if indices.numel() > output_dim:
        raise ValueError(
            f"Selected {indices.numel()} feature dimensions, which exceeds openpi output_dim={output_dim}."
        )
    return (
        _pad_feature(state.index_select(-1, indices), output_dim),
        _pad_feature(actions.index_select(-1, indices), output_dim),
    )


@dataclasses.dataclass(frozen=True)
class PetrelDatasetConfig:
    """Configuration for OpenPI SDK/local LeRobot data."""

    dataset_config: str
    # None uses storage.petrel_conf from the YAML. Set this only as a runtime override.
    conf_path: str | None = None
    # None uses storage.backend/local_root from the YAML. CLI overrides are useful for switching hosts.
    storage_backend: Literal["sdk", "local"] | None = None
    local_root: str | None = None
    prompt_template: str = "{task}"
    training: bool = True
    epoch_length: int | None = None
    load_images: bool = True
    skip_bad_videos: bool = True
    max_decode_retries: int = 8

    def __post_init__(self) -> None:
        if not self.dataset_config:
            raise ValueError("dataset_config must be set for Petrel data.")
        if self.storage_backend not in {None, "sdk", "local"}:
            raise ValueError("storage_backend must be one of: sdk, local.")
        if self.storage_backend == "sdk" and self.local_root is not None:
            raise ValueError("local_root can only be used with storage_backend='local'.")
        if "{task}" not in self.prompt_template:
            raise ValueError("prompt_template must contain '{task}'.")
        if self.epoch_length is not None and self.epoch_length <= 0:
            raise ValueError("epoch_length must be positive when provided.")
        if self.max_decode_retries < 0:
            raise ValueError("max_decode_retries must be non-negative.")


class PetrelDataset(torch.utils.data.Dataset):
    """Expose remote LeRobot samples using openpi's pre-transform schema."""

    def __init__(
        self,
        config: PetrelDatasetConfig,
        *,
        action_horizon: int,
        model_action_dim: int,
    ) -> None:
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")
        if model_action_dim <= 0:
            raise ValueError("model_action_dim must be positive.")

        self.config = config
        self.action_horizon = int(action_horizon)
        self.model_action_dim = int(model_action_dim)
        self.output_dim = self.model_action_dim
        self.control_projection = _COMPACT_VALID_PROJECTION
        self._camera_key_map: dict[str, str] = {}
        self._datasets: list[Any] | None = None
        self._mixed_dataset: Any | None = None
        self._length: int | None = None
        self._epoch = 0
        self.prefer_sequential_indices = False
        self.requires_worker_restart_each_epoch = True
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        if self._mixed_dataset is not None:
            return

        loader = reader
        config_path = Path(self.config.dataset_config).expanduser()  # "configs/petrel/pi05_xxx.yaml"
        raw_config, specs = loader.load_dataset_specs(
            config_path,
            self.config.conf_path,
            storage_backend=self.config.storage_backend,
            local_root=self.config.local_root,
        )
        self._validate_openpi_contract(raw_config)
        if not specs:
            raise ValueError(f"Petrel dataset config expanded to no datasets: {config_path}")

        storage_cfg = raw_config.get("storage", {}) if isinstance(raw_config.get("storage"), dict) else {}
        sampling_cfg = raw_config.get("sampling", {}) if isinstance(raw_config.get("sampling"), dict) else {}
        parquet_cache_size = int(storage_cfg.get("parquet_cache_size", 8))
        datasets = [
            loader.PetrelLeRobotDataset(
                spec,
                num_frames=int(sampling_cfg["num_frames"]),
                action_size=self.action_horizon,
                global_sample_stride=int(sampling_cfg["global_sample_stride"]),
                parquet_cache_size=parquet_cache_size,
                use_path_index_cache=True,
            )
            for spec in specs
        ]

        mixture_cfg = raw_config.get("mixture", {}) if isinstance(raw_config.get("mixture"), dict) else {}
        allow_padding = bool(mixture_cfg.get("allow_padding_at_end", False))
        # List Weights a1_articulation_franka.yaml-datasets  petrel.yaml weights * singel a1_articulation_franka.yaml-datasets / all a1_articulation_franka.yaml-datasets
        dataset_weights = loader.distributed_dataset_weights(
            specs,
            datasets,
            allow_padding_at_end=allow_padding,
        )
        epoch_length = self.config.epoch_length
        if epoch_length is None:
            configured_epoch_length = mixture_cfg.get("epoch_length")
            epoch_length = None if configured_epoch_length in (None, "") else int(configured_epoch_length)

        mixed_dataset = loader.MixedPetrelLeRobotDataset(
            datasets,
            dataset_weights=dataset_weights,
            training=self.config.training,
            balance_dataset_weights=bool(mixture_cfg.get("balance_dataset_weights", False)),
            balance_trajectory_weights=bool(mixture_cfg.get("balance_trajectory_weights", True)),
            seed=int(mixture_cfg.get("seed", 42)),
            allow_padding_at_end=bool(allow_padding),
            epoch_length=epoch_length,
            training_block_size=int(mixture_cfg.get("training_block_size", mixture_cfg.get("block_size", 1))),
            training_frame_stride=int(mixture_cfg.get("training_frame_stride", 1)),
        )
        mixed_dataset.set_epoch(self._epoch)

        self._datasets = datasets
        self._mixed_dataset = mixed_dataset
        self._length = len(mixed_dataset)
        self.prefer_sequential_indices = bool(getattr(mixed_dataset, "prefer_sequential_indices", False))
        self._validate_feature_dimensions()

    def _validate_openpi_contract(self, raw_config: Mapping[str, Any]) -> None:
        contract = raw_config.get("openpi")
        if not isinstance(contract, Mapping):
            raise ValueError("Validated Petrel config is missing the openpi contract.")
        state_dim = int(contract["state_dim"])
        action_dim = int(contract["action_dim"])
        action_horizon = int(contract["action_horizon"])
        if state_dim != self.model_action_dim or action_dim != self.model_action_dim:
            raise ValueError(
                "Petrel YAML/model dimension mismatch: "
                f"yaml state={state_dim}, action={action_dim}; model action_dim={self.model_action_dim}."
            )
        if action_horizon != self.action_horizon:
            raise ValueError(f"Petrel YAML/model horizon mismatch: yaml={action_horizon}, model={self.action_horizon}.")
        if contract.get("normalization") != "quantile":
            raise ValueError("pi05_petrel requires openpi.normalization: quantile.")
        self.output_dim = action_dim
        self.control_projection = str(contract.get("control_projection", _COMPACT_VALID_PROJECTION))

    def _effective_feature_dim(self, dimension: int) -> int:
        del dimension
        return self.output_dim

    def _validate_feature_dimensions(self) -> None:
        assert self._datasets is not None
        oversized = [
            (
                dataset.name,
                self._effective_feature_dim(int(dataset.state_target_dim)),
                self._effective_feature_dim(int(dataset.action_target_dim)),
            )
            for dataset in self._datasets
            if self._effective_feature_dim(int(dataset.state_target_dim)) > self.model_action_dim
            or self._effective_feature_dim(int(dataset.action_target_dim)) > self.model_action_dim
        ]
        if oversized:
            details = ", ".join(
                f"{name}(state={state_dim}, action={action_dim})" for name, state_dim, action_dim in oversized[:8]
            )
            raise ValueError(
                f"Petrel feature dimensions exceed openpi model action_dim={self.model_action_dim}: {details}. "
                "Use a lower-dimensional Petrel projection or increase model action_dim with a compatible checkpoint."
            )

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_datasets"] = None
        state["_mixed_dataset"] = None
        return state

    def __len__(self) -> int:
        if self._length is None:
            self._initialize_backend()
        assert self._length is not None
        return self._length

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        if self._mixed_dataset is not None:
            self._mixed_dataset.set_epoch(self._epoch)

    def _ensure_backend(self) -> tuple[Any, list[Any], Any]:
        self._initialize_backend()
        assert self._datasets is not None
        assert self._mixed_dataset is not None
        return reader, self._datasets, self._mixed_dataset

    @staticmethod
    def _dataset_for_item(item: Mapping[str, Any], datasets: Sequence[Any]) -> tuple[int, Any]:
        dataset_index = int(torch.as_tensor(item["dataset_index"]).item())
        return dataset_index, datasets[dataset_index]

    def _camera_assignments(self, dataset: Any) -> list[tuple[str, str]]:
        assignments: list[tuple[str, str]] = []
        used_targets: set[str] = set()
        for source_key in dataset.video_keys:
            target = self._camera_key_map.get(source_key)
            if target is not None and target not in used_targets:
                assignments.append((source_key, target))
                used_targets.add(target)

        assigned_sources = {source for source, _ in assignments}
        remaining_sources = [key for key in dataset.video_keys if key not in assigned_sources]
        remaining_targets = [target for target in _OPENPI_CAMERA_NAMES if target not in used_targets]
        assignments.extend(zip(remaining_sources, remaining_targets, strict=False))
        return assignments

    @staticmethod
    def _frame_to_openpi_image(frame: torch.Tensor) -> np.ndarray:
        value = frame.detach().cpu()
        if value.ndim != 3:
            raise ValueError(f"Expected a video frame with shape [C,H,W], got {tuple(value.shape)}")
        if value.shape[0] not in {1, 3, 4}:
            raise ValueError(f"Expected channels-first video frame, got {tuple(value.shape)}")
        if torch.is_floating_point(value):
            value = value.mul(255.0).round().clamp(0, 255).to(torch.uint8)
        elif value.dtype != torch.uint8:
            value = value.clamp(0, 255).to(torch.uint8)
        if value.shape[0] == 4:
            value = value[:3]
        if value.shape[0] == 1:
            value = value.expand(3, -1, -1)
        return value.permute(1, 2, 0).contiguous().numpy()

    def _decode_item_images(self, item: Mapping[str, Any], dataset: Any) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for source_key, target_key in self._camera_assignments(dataset):
            clip = dataset.get_video_for_item(dict(item), video_key=source_key, sample_indices=(0,))
            if clip.ndim != 4 or clip.shape[0] == 0:
                raise ValueError(f"Decoded camera {source_key!r} has invalid shape {tuple(clip.shape)}")
            images[target_key] = self._frame_to_openpi_image(clip[0])
        return images

    def _project_features(self, item: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        return project_petrel_features(item, self.output_dim, projection=self.control_projection)

    def _pack_item(self, item: Mapping[str, Any], images: Mapping[str, np.ndarray]) -> dict[str, Any]:
        state, actions = self._project_features(item)
        if state.ndim != 2 or state.shape[0] == 0:
            raise ValueError(f"Expected Petrel state shape [T,D], got {tuple(state.shape)}")
        if actions.ndim != 2 or actions.shape[0] != self.action_horizon:
            raise ValueError(f"Expected Petrel action shape [{self.action_horizon},D], got {tuple(actions.shape)}")
        task = str(item.get("task", item.get("prompt", "")))
        output = {
            "state": state[0].numpy(),
            "actions": actions.numpy(),
            "prompt": self.config.prompt_template.format(task=task),
        }
        if not self.config.load_images:
            return output
        if not images:
            raise ValueError("Petrel sample contains no decoded camera images.")

        base_image = images.get("base_0_rgb")
        if base_image is None:
            base_image = next(iter(images.values()))
        output_images: dict[str, np.ndarray] = {}
        image_masks: dict[str, np.bool_] = {}
        for camera_name in _OPENPI_CAMERA_NAMES:
            image = images.get(camera_name)
            output_images[camera_name] = np.zeros_like(base_image) if image is None else image
            image_masks[camera_name] = np.bool_(image is not None)

        return {
            **output,
            "image": output_images,
            "image_mask": image_masks,
        }

    def _replacement_index(self, index: int, attempt: int) -> int:
        return int((index + attempt * 104729 + 17) % max(len(self), 1))

    def __getitem__(self, index: int) -> dict[str, Any]:
        _, datasets, mixed_dataset = self._ensure_backend()
        if not self.config.load_images:
            return self._pack_item(mixed_dataset[int(index)], {})
        attempts = self.config.max_decode_retries + 1 if self.config.skip_bad_videos else 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            sample_index = int(index) if attempt == 0 else self._replacement_index(int(index), attempt)
            item = mixed_dataset[sample_index]
            _, dataset = self._dataset_for_item(item, datasets)
            try:
                images = self._decode_item_images(item, dataset)
                return self._pack_item(item, images)
            except Exception as exc:
                last_error = exc
                if not self.config.skip_bad_videos:
                    raise
                logging.warning(
                    "Failed Petrel sample index=%d replacement=%d attempt=%d/%d: %s",
                    index,
                    sample_index,
                    attempt + 1,
                    attempts,
                    exc,
                )
        raise RuntimeError(f"Unable to decode Petrel sample index={index} after {attempts} attempts") from last_error

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        if not indices:
            return []
        loader, datasets, mixed_dataset = self._ensure_backend()
        items = [mixed_dataset[int(index)] for index in indices]
        if not self.config.load_images:
            return [self._pack_item(item, {}) for item in items]
        images_by_position: list[dict[str, np.ndarray]] = [{} for _ in items]
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for position, item in enumerate(items):
            dataset_index, _ = self._dataset_for_item(item, datasets)
            episode_index = int(torch.as_tensor(item["episode_index"]).item())
            groups[(dataset_index, episode_index)].append(position)

        try:
            for (dataset_index, episode_index), positions in groups.items():
                dataset = datasets[dataset_index]
                for source_key, target_key in self._camera_assignments(dataset):
                    timestamps: list[float] = []
                    for position in positions:
                        item = items[position]
                        if "timestamp" in item:
                            timestamp = float(torch.as_tensor(item["timestamp"])[0].item())
                        else:
                            frame_index = int(torch.as_tensor(item["frame_index"])[0].item())
                            timestamp = frame_index / float(dataset.fps)
                        timestamp += float(dataset._video_timestamp_offset(episode_index, source_key))
                        timestamps.append(timestamp)

                    unique_timestamps = sorted(set(timestamps))
                    raw_video = dataset._load_episode_video_bytes(episode_index, source_key)
                    decoded = loader.decode_mp4_bytes(raw_video, unique_timestamps)
                    timestamp_positions = {timestamp: pos for pos, timestamp in enumerate(unique_timestamps)}
                    for position, timestamp in zip(positions, timestamps, strict=True):
                        frame = decoded[timestamp_positions[timestamp]]
                        images_by_position[position][target_key] = self._frame_to_openpi_image(frame)
            return [self._pack_item(item, images) for item, images in zip(items, images_by_position, strict=True)]
        except Exception as exc:
            if not self.config.skip_bad_videos:
                raise
            logging.warning("Batched Petrel decode failed; retrying individually: %s", exc)
            return [self[int(index)] for index in indices]
