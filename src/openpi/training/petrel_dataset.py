"""Native OpenPI dataset for locally mounted LeRobot v2/v3 data."""

# ruff: noqa: SLF001

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import dataclasses
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from openpi.training import petrel_lerobot_reader as reader
from openpi.training import profiling as _profiling

_OPENPI_CAMERA_NAMES = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def read_petrel_dataset_metadata(config_path: str) -> tuple[str, Mapping[str, Any]]:
    """Read the dataset id and OpenPI contract without scanning dataset files."""
    return reader.read_openpi_petrel_metadata(config_path)


def _dimension_padding(item: Mapping[str, Any], key: str, dimension: int) -> torch.Tensor:
    value = item.get(key)
    if value is None:
        return torch.zeros(dimension, dtype=torch.bool)
    padding = torch.as_tensor(value, dtype=torch.bool).flatten()
    if padding.numel() != dimension:
        raise ValueError(f"Expected {key} to have {dimension} entries, got {padding.numel()}.")
    return padding


def project_petrel_features(item: Mapping[str, Any], output_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
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

    delta_indices = [int(index) for index in item.get("action_delta_from_state_indices", ())]
    if delta_indices:
        if min(delta_indices) < 0 or max(delta_indices) >= dimension:
            raise ValueError(f"action_delta_from_state_indices exceed mapped dimension {dimension}: {delta_indices}")
        index_tensor = torch.as_tensor(delta_indices, dtype=torch.long, device=actions.device)
        reference_state = state.index_select(-1, index_tensor)[..., 0, :].unsqueeze(-2)
        absolute_actions = actions.index_select(-1, index_tensor)
        actions = actions.clone()
        actions.index_copy_(-1, index_tensor, absolute_actions - reference_state)

    if dimension != output_dim:
        raise ValueError(
            f"Mapped Petrel features require canonical dimension={output_dim}, got {dimension}. "
            "Align source_control_space.dim with openpi.action_dim."
        )
    return state, actions


@dataclasses.dataclass(frozen=True)
class PetrelDatasetConfig:
    """Configuration for locally mounted LeRobot data."""

    dataset_config: str
    # None uses storage.local_root from the YAML; set this to override the mount point at runtime.
    local_root: str | None = None
    prompt_template: str = "{task}"
    training: bool = True
    epoch_length: int | None = None
    load_images: bool = True
    skip_bad_videos: bool = True
    max_decode_retries: int = 8
    # Local files are passed directly to PyAV so it can seek instead of reading and copying the whole MP4.
    local_video_path_decode: bool = True

    def __post_init__(self) -> None:
        if not self.dataset_config:
            raise ValueError("dataset_config must be set for Petrel data.")
        if "{task}" not in self.prompt_template:
            raise ValueError("prompt_template must contain '{task}'.")
        if self.epoch_length is not None and self.epoch_length <= 0:
            raise ValueError("epoch_length must be positive when provided.")
        if self.max_decode_retries < 0:
            raise ValueError("max_decode_retries must be non-negative.")


class PetrelDataset(torch.utils.data.Dataset):
    """Expose locally mounted LeRobot samples using OpenPI's pre-transform schema."""

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
        self._camera_key_map: dict[str, str] = {}
        self._datasets: list[Any] | None = None
        self._mixed_dataset: Any | None = None
        self._length: int | None = None
        self._epoch = 0
        self.prefer_sequential_indices = False
        self.training_block_size = 1
        self.requires_worker_restart_each_epoch = True
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        if self._mixed_dataset is not None:
            return

        loader = reader
        config_path = Path(self.config.dataset_config).expanduser()  # "configs/petrel/pi05_xxx.yaml"
        raw_config, specs = loader.load_dataset_specs(config_path, local_root=self.config.local_root)
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
                local_video_path_decode=self.config.local_video_path_decode,
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
        self.training_block_size = int(getattr(mixed_dataset, "training_block_size", 1))
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
        image_config = contract.get("images", {})
        if not isinstance(image_config, Mapping):
            raise TypeError("Petrel openpi.images must be a mapping.")
        key_map = image_config.get("key_map", {})
        if not isinstance(key_map, Mapping):
            raise TypeError("Petrel openpi.images.key_map must be a mapping.")
        self._camera_key_map = {str(source): str(target) for source, target in key_map.items()}
        self.output_dim = action_dim

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
        # A control profile can restrict the video keys and provide its own mapping.
        # This is needed for schemas such as ABC-130K stereo-top recordings, where
        # only one of the two head cameras should be decoded for a three-image model.
        key_map = dict(self._camera_key_map)
        modalities = getattr(dataset, "modalities", {})
        if isinstance(modalities, Mapping):
            video_config = modalities.get("video", {})
            if isinstance(video_config, Mapping):
                profile_key_map = video_config.get("key_map", {})
                if not isinstance(profile_key_map, Mapping):
                    raise TypeError("Dataset video.key_map must be a mapping.")
                key_map.update({str(source): str(target) for source, target in profile_key_map.items()})

        assignments: list[tuple[str, str]] = []
        used_targets: set[str] = set()
        for source_key in dataset.video_keys:
            target = key_map.get(source_key)
            if target is not None and target not in used_targets:
                assignments.append((source_key, target))
                used_targets.add(target)

        if key_map:
            return assignments

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
        return project_petrel_features(item, self.output_dim)

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
            "stats_group": str(item.get("stats_group", "")),
            "stats_group_fallbacks": tuple(str(group) for group in item.get("stats_group_fallbacks", ())),
        }
        state_dim_is_pad = np.asarray(item.get("observation.state_dim_is_pad", ()), dtype=bool)
        action_dim_is_pad = np.asarray(item.get("action_dim_is_pad", ()), dtype=bool)
        if state_dim_is_pad.shape == (state.shape[-1],):
            output["state_dim_is_pad"] = state_dim_is_pad
        if action_dim_is_pad.shape == (actions.shape[-1],):
            output["actions_dim_is_pad"] = action_dim_is_pad
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
        batch_started = time.perf_counter()
        loader, datasets, mixed_dataset = self._ensure_backend()
        phase_started = time.perf_counter()
        item_groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for position, index in enumerate(indices):
            dataset_index, episode_index, frame_index = mixed_dataset.sample_step(int(index))
            item_groups[(dataset_index, episode_index)].append((position, frame_index))
        _profiling.record("petrel.batch.group_indices", time.perf_counter() - phase_started)

        phase_started = time.perf_counter()
        items: list[dict[str, Any] | None] = [None] * len(indices)
        for (dataset_index, episode_index), position_frames in item_groups.items():
            dataset = datasets[dataset_index]
            episode_items = dataset.get_step_items(
                episode_index,
                [frame_index for _, frame_index in position_frames],
            )
            if len(episode_items) != len(position_frames):
                raise RuntimeError(
                    f"Dataset {dataset.name!r} returned {len(episode_items)} items for "
                    f"{len(position_frames)} requested frames in episode {episode_index}."
                )
            for (position, _), item in zip(position_frames, episode_items, strict=True):
                item["dataset_index"] = torch.tensor(dataset_index, dtype=torch.long)
                items[position] = item
        _profiling.record("petrel.batch.parquet_and_samples", time.perf_counter() - phase_started)

        if any(item is None for item in items):
            raise RuntimeError("Batched Petrel loading did not populate every requested position.")
        loaded_items = [item for item in items if item is not None]
        if not self.config.load_images:
            phase_started = time.perf_counter()
            packed_items = [self._pack_item(item, {}) for item in loaded_items]
            _profiling.record("petrel.batch.pack", time.perf_counter() - phase_started)
            _profiling.record("petrel.batch.total", time.perf_counter() - batch_started)
            return packed_items
        images_by_position: list[dict[str, np.ndarray]] = [{} for _ in loaded_items]
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for position, item in enumerate(loaded_items):
            dataset_index, _ = self._dataset_for_item(item, datasets)
            episode_index = int(torch.as_tensor(item["episode_index"]).item())
            groups[(dataset_index, episode_index)].append(position)

        try:
            for (dataset_index, episode_index), positions in groups.items():
                dataset = datasets[dataset_index]
                for source_key, target_key in self._camera_assignments(dataset):
                    timestamps: list[float] = []
                    for position in positions:
                        item = loaded_items[position]
                        if "timestamp" in item:
                            timestamp = float(torch.as_tensor(item["timestamp"])[0].item())
                        else:
                            frame_index = int(torch.as_tensor(item["frame_index"])[0].item())
                            timestamp = frame_index / float(dataset.fps)
                        timestamp += float(dataset._video_timestamp_offset(episode_index, source_key))
                        timestamps.append(timestamp)

                    unique_timestamps = sorted(set(timestamps))
                    decode_episode_video = getattr(dataset, "decode_episode_video", None)
                    if callable(decode_episode_video):
                        decoded = decode_episode_video(episode_index, source_key, unique_timestamps)
                    else:
                        # Compatibility for lightweight test doubles and older reader implementations.
                        raw_video = dataset._load_episode_video_bytes(episode_index, source_key)
                        decoded = loader.decode_mp4_bytes(raw_video, unique_timestamps)
                    timestamp_positions = {timestamp: pos for pos, timestamp in enumerate(unique_timestamps)}
                    phase_started = time.perf_counter()
                    for position, timestamp in zip(positions, timestamps, strict=True):
                        frame = decoded[timestamp_positions[timestamp]]
                        images_by_position[position][target_key] = self._frame_to_openpi_image(frame)
                    _profiling.record("petrel.batch.frame_conversion", time.perf_counter() - phase_started)
            phase_started = time.perf_counter()
            packed_items = [
                self._pack_item(item, images)
                for item, images in zip(loaded_items, images_by_position, strict=True)
            ]
            _profiling.record("petrel.batch.pack", time.perf_counter() - phase_started)
            _profiling.record("petrel.batch.total", time.perf_counter() - batch_started)
            return packed_items
        except Exception as exc:
            if not self.config.skip_bad_videos:
                raise
            logging.warning("Batched Petrel decode failed; retrying individually: %s", exc)
            return [self[int(index)] for index in indices]
