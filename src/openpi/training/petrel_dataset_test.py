from collections import OrderedDict
import dataclasses
import pathlib
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from openpi import transforms
from openpi.shared import normalize
from openpi.training import petrel_dataset
from openpi.training import petrel_norm_stats


class _FakePetrelDataset:
    def __init__(self, spec, **kwargs):
        del kwargs
        self.name = spec.name
        self.state_target_dim = 8
        self.action_target_dim = 8
        self.video_keys = ["head", "left_wrist"]
        self.fps = 10.0
        self.step_item_batches = []

    @staticmethod
    def _step_item(episode_index, frame_index):
        return {
            "dataset_name": "fake",
            "episode_index": torch.tensor(episode_index),
            "frame_index": torch.tensor([frame_index]),
            "observation.state": torch.tensor([[0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            "action": torch.arange(24, dtype=torch.float32).reshape(3, 8),
            "observation.state_dim_is_pad": torch.tensor([False, True, False, True, True, True, True, True]),
            "action_dim_is_pad": torch.tensor([False, True, False, True, True, True, True, True]),
            "task": "pick up the cup",
        }

    def get_step_item(self, episode_index, frame_index):
        return self._step_item(episode_index, frame_index)

    def get_step_items(self, episode_index, frame_indices):
        self.step_item_batches.append((episode_index, list(frame_indices)))
        return [self._step_item(episode_index, frame_index) for frame_index in frame_indices]

    def get_video_for_item(self, item, *, video_key, sample_indices):
        del item, sample_indices
        value = 0.25 if video_key == "head" else 0.75
        return torch.full((1, 3, 6, 8), value, dtype=torch.float32)

    def _video_timestamp_offset(self, episode_index, video_key):
        del episode_index, video_key
        return 0.0

    def _load_episode_video_bytes(self, episode_index, video_key):
        del episode_index
        return video_key.encode()


class _FakeMixedDataset:
    prefer_sequential_indices = True

    def __init__(self, datasets, **kwargs):
        del kwargs
        self.datasets = datasets
        self.epoch = 0

    def __len__(self):
        return 16

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample_step(self, index):
        return 0, 2, index

    def __getitem__(self, index):
        dataset_index, episode_index, frame_index = self.sample_step(index)
        item = self.datasets[dataset_index].get_step_item(episode_index, frame_index)
        item["dataset_index"] = torch.tensor(dataset_index)
        return item


def _fake_reader(action_dim=8):
    return SimpleNamespace(
        load_dataset_specs=lambda path, **kwargs: (
            {
                "openpi": {
                    "state_dim": action_dim,
                    "action_dim": action_dim,
                    "action_horizon": 3,
                    "normalization": "quantile",
                    "control_projection": "mapped",
                },
                "storage": {"parquet_cache_size": 8},
                "sampling": {"num_frames": 1, "global_sample_stride": 1},
                "mixture": {"training_block_size": 4, "training_frame_stride": 2},
            },
            [SimpleNamespace(name="fake")],
        ),
        distributed_dataset_weights=lambda specs, datasets, allow_padding_at_end: [1.0],
        PetrelLeRobotDataset=_FakePetrelDataset,
        MixedPetrelLeRobotDataset=_FakeMixedDataset,
        decode_mp4_bytes=lambda raw, timestamps: torch.full(
            (len(timestamps), 3, 6, 8),
            0.25 if raw == b"head" else 0.75,
            dtype=torch.float32,
        ),
    )


def test_petrel_dataset_outputs_openpi_schema(monkeypatch, tmp_path):
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text("mixture: {}", encoding="utf-8")
    fake = _fake_reader()
    for name, value in vars(fake).items():
        monkeypatch.setattr(petrel_dataset.reader, name, value)

    dataset = petrel_dataset.PetrelDataset(
        petrel_dataset.PetrelDatasetConfig(
            dataset_config=str(config_path),
            prompt_template="Robot instruction: {task}",
            skip_bad_videos=False,
        ),
        action_horizon=3,
        model_action_dim=8,
    )

    sample = dataset[0]

    assert sample["state"].shape == (8,)
    assert sample["actions"].shape == (3, 8)
    np.testing.assert_array_equal(sample["state"], np.asarray([0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert sample["prompt"] == "Robot instruction: pick up the cup"
    assert sample["image"]["base_0_rgb"].shape == (6, 8, 3)
    assert sample["image"]["base_0_rgb"].dtype == np.uint8
    assert sample["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.False_,
    }
    assert dataset.prefer_sequential_indices


def test_petrel_dataset_stats_mode_skips_video_decode(monkeypatch, tmp_path):
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text("mixture: {}", encoding="utf-8")
    fake = _fake_reader()
    for name, value in vars(fake).items():
        monkeypatch.setattr(petrel_dataset.reader, name, value)
    monkeypatch.setattr(
        _FakePetrelDataset,
        "get_video_for_item",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("video must not be read")),
    )
    dataset = petrel_dataset.PetrelDataset(
        petrel_dataset.PetrelDatasetConfig(dataset_config=str(config_path), load_images=False),
        action_horizon=3,
        model_action_dim=8,
    )

    sample = dataset[0]
    samples = dataset.__getitems__([0, 1])

    expected_keys = {
        "state",
        "actions",
        "prompt",
        "stats_group",
        "stats_group_fallbacks",
        "state_dim_is_pad",
        "actions_dim_is_pad",
    }
    assert set(sample) == expected_keys
    assert len(samples) == 2
    assert all(set(item) == expected_keys for item in samples)


def test_petrel_dataset_batch_decode(monkeypatch, tmp_path):
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text("mixture: {}", encoding="utf-8")
    fake = _fake_reader()
    for name, value in vars(fake).items():
        monkeypatch.setattr(petrel_dataset.reader, name, value)
    dataset = petrel_dataset.PetrelDataset(
        petrel_dataset.PetrelDatasetConfig(dataset_config=str(config_path), skip_bad_videos=False),
        action_horizon=3,
        model_action_dim=8,
    )

    samples = dataset.__getitems__([0, 1, 2])

    assert len(samples) == 3
    assert all(sample["actions"].shape == (3, 8) for sample in samples)
    assert dataset._datasets[0].step_item_batches == [(2, [0, 1, 2])]  # noqa: SLF001


def test_petrel_dataset_worker_reuses_resolved_specs(monkeypatch, tmp_path):
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text("mixture: {}", encoding="utf-8")
    fake = _fake_reader()
    load_calls = 0
    original_load = fake.load_dataset_specs

    def counted_load(*args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return original_load(*args, **kwargs)

    fake.load_dataset_specs = counted_load
    for name, value in vars(fake).items():
        monkeypatch.setattr(petrel_dataset.reader, name, value)

    dataset = petrel_dataset.PetrelDataset(
        petrel_dataset.PetrelDatasetConfig(dataset_config=str(config_path)),
        action_horizon=3,
        model_action_dim=8,
    )
    assert load_calls == 1

    worker_state = dataset.__getstate__()
    assert worker_state["_datasets"] is None
    assert worker_state["_mixed_dataset"] is None
    assert worker_state["_resolved_backend_config"] is not None
    assert worker_state["_resolved_specs"] is not None

    monkeypatch.setattr(
        petrel_dataset.reader,
        "load_dataset_specs",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("worker must reuse resolved specs")),
    )
    worker_dataset = pickle.loads(pickle.dumps(dataset))
    worker_dataset._initialize_backend()  # noqa: SLF001

    assert worker_dataset._mixed_dataset is not None  # noqa: SLF001
    assert worker_dataset._datasets is not None  # noqa: SLF001


def test_petrel_dataset_persistent_worker_observes_shared_epoch(monkeypatch, tmp_path):
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text("mixture: {}", encoding="utf-8")
    fake = _fake_reader()
    for name, value in vars(fake).items():
        monkeypatch.setattr(petrel_dataset.reader, name, value)

    dataset = petrel_dataset.PetrelDataset(
        petrel_dataset.PetrelDatasetConfig(dataset_config=str(config_path), load_images=False),
        action_horizon=3,
        model_action_dim=8,
    )
    # A worker has its own Python-side epoch/backend but retains the shared tensor.
    worker_dataset = object.__new__(petrel_dataset.PetrelDataset)
    worker_dataset.__dict__ = dict(dataset.__dict__)
    worker_dataset._epoch = 0  # noqa: SLF001
    worker_dataset._mixed_dataset = _FakeMixedDataset(worker_dataset._datasets)  # noqa: SLF001

    dataset.set_epoch(7)
    worker_dataset._ensure_backend()  # noqa: SLF001

    assert not dataset.requires_worker_restart_each_epoch
    assert worker_dataset._epoch == 7  # noqa: SLF001
    assert worker_dataset._mixed_dataset.epoch == 7  # noqa: SLF001


def _mock_distributed(monkeypatch, *, rank, broadcast_value=None, captured=None):
    distributed = petrel_dataset.reader.torch.distributed
    monkeypatch.setattr(distributed, "is_available", lambda: True)
    monkeypatch.setattr(distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(distributed, "get_rank", lambda: rank)

    def broadcast(payload, src):
        assert src == 0
        if captured is not None:
            captured.append(payload[0])
        if broadcast_value is not None:
            payload[0] = broadcast_value

    monkeypatch.setattr(distributed, "broadcast_object_list", broadcast)


def test_nonzero_rank_reuses_rank0_dataset_root_validation(monkeypatch, tmp_path):
    resolved_root = tmp_path / "mounted"
    config = {"storage": {"local_root": "/must/not/be/touched"}}
    spec = petrel_dataset.reader.DatasetSpec(name="dataset", remote_root="parent/dataset")
    monkeypatch.setattr(petrel_dataset.reader, "read_openpi_petrel_config", lambda path: config)
    monkeypatch.setattr(petrel_dataset.reader, "_flatten_source_specs", lambda value: [spec])
    monkeypatch.setattr(
        petrel_dataset.reader.os,
        "scandir",
        lambda path: (_ for _ in ()).throw(AssertionError("non-zero rank must not scan AOSS")),
    )
    _mock_distributed(
        monkeypatch,
        rank=1,
        broadcast_value={"resolved_root": str(resolved_root), "error": None},
    )

    loaded_config, specs = petrel_dataset.reader.load_dataset_specs("unused.yaml")

    assert loaded_config["storage"]["local_root"] == str(resolved_root)
    assert specs[0].remote_root == str(resolved_root / "parent/dataset")
    assert specs[0].local_root == str(resolved_root)


def test_rank0_validates_dataset_roots_once_and_broadcasts(monkeypatch, tmp_path, capsys):
    resolved_root = tmp_path / "mounted"
    (resolved_root / "parent/dataset").mkdir(parents=True)
    config = {"storage": {"local_root": str(resolved_root)}}
    spec = petrel_dataset.reader.DatasetSpec(name="dataset", remote_root="parent/dataset")
    monkeypatch.setattr(petrel_dataset.reader, "read_openpi_petrel_config", lambda path: config)
    monkeypatch.setattr(petrel_dataset.reader, "_flatten_source_specs", lambda value: [spec])
    original_scandir = petrel_dataset.reader.os.scandir
    scans = []

    def counted_scandir(path):
        scans.append(path)
        return original_scandir(path)

    monkeypatch.setattr(petrel_dataset.reader.os, "scandir", counted_scandir)
    broadcasts = []
    _mock_distributed(monkeypatch, rank=0, captured=broadcasts)

    _, specs = petrel_dataset.reader.load_dataset_specs("unused.yaml")

    assert scans == [resolved_root / "parent"]
    assert broadcasts == [{"resolved_root": str(resolved_root), "error": None}]
    assert specs[0].remote_root == str(resolved_root / "parent/dataset")
    output = capsys.readouterr().out
    assert "[petrel-init][rank=0] AOSS scan started specs=1 parents=1" in output
    assert "[petrel-init][rank=0] AOSS scan completed elapsed_s=" in output
    assert "[petrel-init][rank=0] broadcast completed elapsed_s=" in output


def test_rank0_broadcasts_dataset_root_validation_failure(monkeypatch, tmp_path):
    resolved_root = tmp_path / "mounted"
    resolved_root.mkdir()
    config = {"storage": {"local_root": str(resolved_root)}}
    spec = petrel_dataset.reader.DatasetSpec(name="missing", remote_root="parent/missing")
    monkeypatch.setattr(petrel_dataset.reader, "read_openpi_petrel_config", lambda path: config)
    monkeypatch.setattr(petrel_dataset.reader, "_flatten_source_specs", lambda value: [spec])
    broadcasts = []
    _mock_distributed(monkeypatch, rank=0, captured=broadcasts)

    with pytest.raises(RuntimeError, match="Rank 0 dataset root validation failed: FileNotFoundError"):
        petrel_dataset.reader.load_dataset_specs("unused.yaml")

    assert len(broadcasts) == 1
    assert broadcasts[0]["resolved_root"] is None
    assert broadcasts[0]["error"].startswith("FileNotFoundError:")


def test_nonzero_rank_raises_rank0_dataset_root_validation_failure(monkeypatch, tmp_path):
    config = {"storage": {"local_root": "/must/not/be/touched"}}
    spec = petrel_dataset.reader.DatasetSpec(name="dataset", remote_root="parent/dataset")
    monkeypatch.setattr(petrel_dataset.reader, "read_openpi_petrel_config", lambda path: config)
    monkeypatch.setattr(petrel_dataset.reader, "_flatten_source_specs", lambda value: [spec])
    monkeypatch.setattr(
        petrel_dataset.reader.os,
        "scandir",
        lambda path: (_ for _ in ()).throw(AssertionError("non-zero rank must not scan AOSS")),
    )
    _mock_distributed(
        monkeypatch,
        rank=1,
        broadcast_value={"resolved_root": None, "error": "FileNotFoundError: missing dataset root"},
    )

    with pytest.raises(RuntimeError, match="Rank 0 dataset root validation failed: FileNotFoundError"):
        petrel_dataset.reader.load_dataset_specs("unused.yaml")


def test_petrel_episode_parquet_lru_cache_hits_and_evicts():
    dataset = object.__new__(petrel_dataset.reader.PetrelLeRobotDataset)
    dataset.name = "fake"
    dataset._client = object()  # noqa: SLF001
    dataset.parquet_cache_size = 2
    dataset._parquet_cache = OrderedDict()  # noqa: SLF001
    loads = []

    def load_uncached(episode_index):
        loads.append(episode_index)
        return {"episode": torch.tensor([episode_index])}

    dataset._load_episode_parquet_uncached = load_uncached  # noqa: SLF001

    first = dataset._load_episode_parquet(1)  # noqa: SLF001
    assert dataset._load_episode_parquet(1) is first  # noqa: SLF001
    dataset._load_episode_parquet(2)  # noqa: SLF001
    assert dataset._load_episode_parquet(1) is first  # noqa: SLF001
    dataset._load_episode_parquet(3)  # noqa: SLF001
    dataset._load_episode_parquet(2)  # noqa: SLF001

    assert loads == [1, 2, 3, 2]
    assert list(dataset._parquet_cache) == [3, 2]  # noqa: SLF001

    worker_state = dataset.__getstate__()
    assert worker_state["_client"] is None
    assert worker_state["_parquet_cache"] == OrderedDict()


def test_fast_petrel_accumulator_matches_exact_small_statistics():
    from openpi.training import petrel_norm_stats

    accumulator = petrel_norm_stats._Accumulator(dim=2, reservoir_rows=32, seed=7)  # noqa: SLF001
    accumulator.update(np.asarray([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32))
    accumulator.update(np.asarray([[4.0, 6.0], [6.0, 8.0]], dtype=np.float32))

    stats = accumulator.finalize()

    np.testing.assert_allclose(stats.mean, np.asarray([3.0, 5.0]))
    np.testing.assert_allclose(stats.std, np.sqrt(np.asarray([5.0, 5.0])))
    np.testing.assert_allclose(stats.q01, np.asarray([0.06, 2.06]))
    np.testing.assert_allclose(stats.q99, np.asarray([5.94, 7.94]))


def test_fast_petrel_action_accumulator_preserves_horizon_statistics():
    from openpi.training import petrel_norm_stats

    accumulator = petrel_norm_stats._HorizonAccumulator(  # noqa: SLF001
        horizon=2, dim=2, reservoir_rows=64, seed=7
    )
    actions = np.asarray(
        [
            [[0.0, 10.0], [100.0, 1000.0]],
            [[2.0, 14.0], [300.0, 1400.0]],
            [[4.0, 18.0], [500.0, 1800.0]],
        ],
        dtype=np.float32,
    )
    accumulator.update(actions[:2])
    accumulator.update(actions[2:])

    stats = accumulator.finalize()

    assert stats.mean.shape == (2, 2)
    assert stats.std.shape == (2, 2)
    assert stats.q01.shape == (2, 2)
    assert stats.q99.shape == (2, 2)
    np.testing.assert_allclose(stats.mean, actions.mean(axis=0))
    np.testing.assert_allclose(stats.std, actions.std(axis=0), rtol=1e-6)
    np.testing.assert_allclose(stats.q01, np.quantile(actions, 0.01, axis=0))
    np.testing.assert_allclose(stats.q99, np.quantile(actions, 0.99, axis=0))


def test_fast_petrel_episode_blocks_preserve_global_cap():
    from openpi.training import petrel_norm_stats

    units = [
        {"windows": 10, "spec_index": 0, "trajectory_pos": 0, "episode_index": 0},
        {"windows": 20, "spec_index": 1, "trajectory_pos": 0, "episode_index": 1},
        {"windows": 30, "spec_index": 2, "trajectory_pos": 0, "episode_index": 2},
        {"windows": 40, "spec_index": 3, "trajectory_pos": 0, "episode_index": 3},
    ]
    selected = petrel_norm_stats._select_episode_blocks(  # noqa: SLF001
        units,
        target=37,
        block_size=16,
        seed=7,
    )

    assert sum(int(unit["quota"]) for unit in selected) == 37
    assert all(int(unit["quota"]) <= int(unit["windows"]) for unit in selected)


def test_fast_petrel_episode_blocks_cover_every_norm_group():
    from openpi.training import petrel_norm_stats

    units = [
        {
            "windows": 1000,
            "spec_index": 0,
            "trajectory_pos": 0,
            "episode_index": 0,
            "stats_groups": ("profile/3cam", "dataset/abc"),
        },
        {
            "windows": 10,
            "spec_index": 1,
            "trajectory_pos": 0,
            "episode_index": 1,
            "stats_groups": ("profile/4cam", "dataset/abc"),
        },
    ]
    selected = petrel_norm_stats._select_episode_blocks(  # noqa: SLF001
        units,
        target=64,
        block_size=64,
        seed=7,
    )

    covered = {group for unit in selected if int(unit["quota"]) > 0 for group in unit["stats_groups"]}
    assert covered == {"profile/3cam", "profile/4cam", "dataset/abc"}
    assert sum(int(unit["quota"]) for unit in selected) == 64


def test_batched_petrel_projection_matches_openpi_shape():
    item = {
        "dataset_name": "batched",
        "observation.state": torch.arange(24, dtype=torch.float32).reshape(3, 1, 8),
        "action": torch.arange(72, dtype=torch.float32).reshape(3, 3, 8),
        "observation.state_dim_is_pad": torch.tensor([False, True, False, True, True, True, True, True]),
        "action_dim_is_pad": torch.tensor([False, True, False, True, True, True, True, True]),
    }

    state, actions = petrel_dataset.project_petrel_features(item, output_dim=8)

    assert state.shape == (3, 1, 8)
    assert actions.shape == (3, 3, 8)
    torch.testing.assert_close(state[..., [1, 3, 4, 5, 6, 7]], torch.zeros(3, 1, 6))


def test_mapped_projection_converts_only_declared_absolute_joint_actions_to_delta():
    state = torch.arange(32, dtype=torch.float32).reshape(1, 32)
    actions = torch.stack([state[0] + 10.0, state[0] + 20.0])
    item = {
        "dataset_name": "mapped",
        "observation.state": state,
        "action": actions,
        "observation.state_dim_is_pad": torch.zeros(32, dtype=torch.bool),
        "action_dim_is_pad": torch.zeros(32, dtype=torch.bool),
        "action_delta_from_state_indices": (0, 5, 7, 12, 26, 27),
    }

    mapped_state, mapped_actions = petrel_dataset.project_petrel_features(item, 32)

    torch.testing.assert_close(mapped_state, state)
    torch.testing.assert_close(mapped_actions[:, [0, 5, 7, 12, 26, 27]], torch.tensor([[10.0] * 6, [20.0] * 6]))
    # Gripper and EE slots are already expressed in their target semantics and are not subtracted again.
    torch.testing.assert_close(mapped_actions[:, 14:26], actions[:, 14:26])
    torch.testing.assert_close(mapped_actions[:, [6, 13]], actions[:, [6, 13]])


def test_source_control_space_requires_explicit_indices():
    contract = {"state_dim": 32, "action_dim": 32, "control_projection": "mapped"}
    control_space = {
        "name": "openpi",
        "dim": 32,
        "slots": {"left": {"joint": [0, 7]}},
    }

    with pytest.raises(TypeError, match="containing only indices"):
        petrel_dataset.reader._validate_source_control_space(control_space, contract)  # noqa: SLF001


def test_pose_rpy_round_trip_and_delta_output_format():
    rpy = torch.tensor([[0.2, -0.3, 0.4], [-0.5, 0.1, -0.2]], dtype=torch.float32)
    rotvec = petrel_dataset.reader._rpy_to_rotvec(rpy)  # noqa: SLF001
    recovered = petrel_dataset.reader._rotvec_to_rpy(rotvec)  # noqa: SLF001
    torch.testing.assert_close(recovered, rpy, atol=1e-5, rtol=1e-5)

    xyz = torch.tensor([[1.0, 2.0, 3.0]])
    rotation = petrel_dataset.reader._rotvec_to_matrix(rotvec[:1])  # noqa: SLF001
    formatted = petrel_dataset.reader._format_xyz_matrix_pose(xyz, rotation, "delta_xyz_rpy")  # noqa: SLF001
    torch.testing.assert_close(formatted, torch.tensor([[1.0, 2.0, 3.0, 0.2, -0.3, 0.4]]), atol=1e-5, rtol=1e-5)


def test_absolute_ee_target_becomes_world_translation_and_relative_rpy_delta():
    current_source = torch.zeros(1, 6)
    target_source = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.1]])
    frame_transform = {
        "world_from_source": "robot_to_opencv",
        "source_ee_from_canonical_ee": "identity",
    }
    current = petrel_dataset.reader._transform_xyz_matrix_pose(  # noqa: SLF001
        *petrel_dataset.reader._pose_to_xyz_matrix(current_source, "xyz_rpy"),  # noqa: SLF001
        frame_transform,
        is_delta=False,
    )
    target = petrel_dataset.reader._transform_xyz_matrix_pose(  # noqa: SLF001
        *petrel_dataset.reader._pose_to_xyz_matrix(target_source, "xyz_rpy"),  # noqa: SLF001
        frame_transform,
        is_delta=False,
    )
    delta = petrel_dataset.reader._pose_delta_world_from_transformed_xyz_matrix(current, target)  # noqa: SLF001
    delta_rpy = petrel_dataset.reader._format_xyz_matrix_pose(*delta, "delta_xyz_rpy")  # noqa: SLF001

    torch.testing.assert_close(
        delta_rpy,
        torch.tensor([[-2.0, -3.0, 1.0, 0.0, 0.0, 0.1]]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_absolute_ee_action_chunk_uses_initial_observation_as_delta_reference():
    dataset = object.__new__(petrel_dataset.reader.PetrelLeRobotDataset)
    dataset.name = "pose_reference"
    dataset.canonical_adapter = {
        "pose_input": "xyz_rpy",
        "action_pose_semantics": "absolute_target",
        "action_pose_output": "delta_xyz_rpy",
    }
    component = {
        "state_key": "state_pose",
        "action_key": "action_pose",
        "raw_slice": [0, 6],
    }
    parquet = {
        "state_pose": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.5, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "action_pose": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    }

    actions = dataset._canonical_pose_from_component(  # noqa: SLF001
        parquet,
        component,
        [0, 1],
        "action",
        reference_indices=[0, 0],
    )

    assert actions is not None
    torch.testing.assert_close(actions[:, 0], torch.tensor([1.0, 2.0]))


def test_absolute_ee_action_target_stays_absolute():
    dataset = object.__new__(petrel_dataset.reader.PetrelLeRobotDataset)
    dataset.name = "absolute_pose_target"
    dataset.canonical_adapter = {
        "pose_input": "xyz_rpy",
        "action_pose_semantics": "absolute_target",
        "action_pose_output": "xyz_rpy",
    }
    component = {
        "state_key": "state_pose",
        "action_key": "action_pose",
        "raw_slice": [0, 6],
        "target_action_semantics": "absolute",
    }
    parquet = {
        "state_pose": torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "action_pose": torch.tensor([[1.0, 2.0, 3.0, 0.1, -0.2, 0.3]]),
    }

    actions = dataset._canonical_pose_from_component(  # noqa: SLF001
        parquet,
        component,
        [0],
        "action",
        reference_indices=[0],
    )

    assert actions is not None
    torch.testing.assert_close(
        actions,
        torch.tensor([[1.0, 2.0, 3.0, 0.1, -0.2, 0.3]]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_grouped_norm_stats_select_sample_group_and_remove_selector():
    group_a = normalize.NormStats(mean=np.zeros(2), std=np.ones(2), q01=np.zeros(2), q99=np.ones(2))
    group_b = normalize.NormStats(mean=np.zeros(2), std=np.ones(2), q01=np.full(2, 10.0), q99=np.full(2, 20.0))
    norm_stats = {
        transforms.grouped_norm_key("state", "source/robot/task_a"): group_a,
        transforms.grouped_norm_key("actions", "source/robot/task_a"): group_a,
        transforms.grouped_norm_key("state", "source/robot/task_b"): group_b,
        transforms.grouped_norm_key("actions", "source/robot/task_b"): group_b,
    }
    data = {
        "state": np.asarray([15.0, 20.0]),
        "actions": np.asarray([[10.0, 15.0]]),
        "stats_group": "source/robot/task_b",
    }

    result = transforms.NormalizeByGroup(norm_stats, use_quantiles=True)(data)

    np.testing.assert_allclose(result["state"], np.asarray([0.0, 1.0]), atol=1e-6)
    np.testing.assert_allclose(result["actions"], np.asarray([[-1.0, 0.0]]), atol=1e-6)
    assert "stats_group" not in result

    fallback_data = {
        "state": np.asarray([15.0, 20.0]),
        "actions": np.asarray([[10.0, 15.0]]),
        "stats_group": "source/robot/missing_task",
        "stats_group_fallbacks": ("source/robot/task_b", "dataset/all"),
    }
    fallback_result = transforms.NormalizeByGroup(norm_stats, use_quantiles=True)(fallback_data)
    np.testing.assert_allclose(fallback_result["state"], np.asarray([0.0, 1.0]), atol=1e-6)
    assert "stats_group_fallbacks" not in fallback_result

    selected = transforms.select_norm_stats(norm_stats, "source/robot/task_b")
    assert selected == {"state": group_b, "actions": group_b}
    with np.testing.assert_raises(ValueError):
        transforms.select_norm_stats(norm_stats, None)


def test_grouped_norm_stats_broadcast_per_action_horizon():
    state_stats = normalize.NormStats(
        mean=np.zeros(2),
        std=np.ones(2),
        q01=np.zeros(2),
        q99=np.ones(2),
    )
    action_stats = normalize.NormStats(
        mean=np.zeros((2, 2)),
        std=np.ones((2, 2)),
        q01=np.asarray([[0.0, 10.0], [100.0, 1000.0]]),
        q99=np.asarray([[10.0, 20.0], [200.0, 1100.0]]),
    )
    group = "dataset/abc130k"
    data = {
        "state": np.asarray([0.5, 1.0]),
        "actions": np.asarray([[5.0, 15.0], [150.0, 1050.0]]),
        "stats_group": group,
    }

    result = transforms.NormalizeByGroup(
        {
            transforms.grouped_norm_key("state", group): state_stats,
            transforms.grouped_norm_key("actions", group): action_stats,
        },
        use_quantiles=True,
    )(data)

    np.testing.assert_allclose(result["actions"], np.zeros((2, 2)), atol=1e-6)


def test_grouped_norm_restores_missing_control_slots_to_zero():
    stats = normalize.NormStats(
        mean=np.zeros(4),
        std=np.ones(4),
        q01=np.zeros(4),
        q99=np.asarray([1.0, 0.0, 1.0, 0.0]),
    )
    data = {
        "state": np.asarray([0.5, 0.0, 0.5, 0.0]),
        "actions": np.asarray([[0.5, 0.0, 0.5, 0.0]]),
        "state_dim_is_pad": np.asarray([False, True, False, True]),
        "actions_dim_is_pad": np.asarray([False, True, False, True]),
        "stats_group": "profile/test",
    }
    norm_stats = {
        transforms.grouped_norm_key("state", "profile/test"): stats,
        transforms.grouped_norm_key("actions", "profile/test"): stats,
    }

    result = transforms.NormalizeByGroup(norm_stats, use_quantiles=True)(data)

    np.testing.assert_array_equal(result["state"][[1, 3]], np.zeros(2))
    np.testing.assert_array_equal(result["actions"][:, [1, 3]], np.zeros((1, 2)))
    assert "state_dim_is_pad" not in result
    np.testing.assert_array_equal(result["actions_dim_is_pad"], np.asarray([False, True, False, True]))


def test_dataset_config_owns_cache_templates_and_dataset_norm_group():
    config = petrel_dataset.reader.read_openpi_petrel_config("configs/petrel/pi05_interna1.yaml")
    profile = config["embodiments"]["a1_single_ee"]

    assert "local_meta_dir_template" not in profile
    assert "path_index_dir_template" not in profile
    assert profile["stats_group_template"] is None
    assert config["cache_templates"]["local_meta_dir"].endswith("data/petrel_cache/meta/{dataset_id}/{remote}")
    assert config["cache_templates"]["path_index_dir"].endswith("data/petrel_cache/path_index/{dataset_id}/{remote}")

    source = next(item for item in config["sources"] if item["name"] == "a1_articulation_franka")
    assert source["stats_group_template"] is None

    specs = petrel_dataset.reader._flatten_source_specs(config)  # noqa: SLF001
    spec = next(item for item in specs if item.source_name == "a1_articulation_franka")
    assert spec.local_meta_dir.endswith(f"data/petrel_cache/meta/intern_a1/{spec.remote_root}")
    assert spec.path_index_dir.endswith(f"data/petrel_cache/path_index/intern_a1/{spec.remote_root}")
    assert spec.stats_group == "dataset/intern_a1"
    assert spec.stats_group_fallbacks == ()

    source["stats_group_template"] = "task/{dataset_id}/{source}/{name}"
    task_specs = petrel_dataset.reader._flatten_source_specs(config)  # noqa: SLF001
    task_spec = next(item for item in task_specs if item.source_name == "a1_articulation_franka")
    assert task_spec.stats_group == f"task/intern_a1/a1_articulation_franka/{task_spec.name}"
    assert task_spec.stats_group_fallbacks == ("dataset/intern_a1",)


def test_reader_exposes_only_local_file_client():
    assert hasattr(petrel_dataset.reader, "LocalFileClient")
    assert not hasattr(petrel_dataset.reader, "PetrelClient")
    assert not hasattr(petrel_dataset.reader, "_parse_petrel_s3_url")


def test_openpi_petrel_v1_schema_translation_and_local_backend(tmp_path):
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    (dataset_dir / "task_a.yaml").write_text(
        """
schema: openpi_petrel_dataset_group_v1
name: one
category: single
profile: single_arm
control_profiles:
  single_arm:
    control_type: single_arm
    mapping:
      slots:
        right:
          joint:
            state_key: observation.state
            action_key: action
            raw_slice: [0, 7]
            target_slot: right.joint
    fields:
      state: {keys: [observation.state]}
      action: {keys: [action]}
      video: {key: auto}
format: v3
root: .
datasets:
  - name: task_a
    path: task_a
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "single.yaml"
    config_path.write_text(
        """
schema: openpi_petrel_v1
dataset_id: single_robot
openpi:
  state_dim: 32
  action_dim: 32
  action_horizon: 16
  control_projection: mapped
  normalization: quantile
  images:
    size: [224, 224]
    dtype: uint8
    channels: rgb
    slots: [base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb]
storage:
  parquet_cache_size: 2
  metadata_cache: cache/meta/{remote}
  path_index_cache: cache/{remote}
sampling:
  state_frames: 1
  action_horizon: 16
  stride: 1
  allow_end_padding: false
mixture:
  category_weights: {single: 1.0}
  distribute_category_weight_by_valid_windows: true
source_control_space:
  name: openpi
  dim: 32
  slots:
    right:
      joint:
        indices: [7, 8, 9, 10, 11, 12, 27]
  missing_policy: pad_and_mask
  pose:
    state: xyz_rpy
    action: delta_xyz_rpy
  action_semantics:
    joint: absolute
dataset_files:
  directory: datasets
  recursive: true
  pattern: '*.yaml'
""".strip(),
        encoding="utf-8",
    )

    translated = petrel_dataset.reader.read_openpi_petrel_config(config_path)

    assert translated["dataset_id"] == "single_robot"
    assert translated["openpi"]["action_dim"] == 32
    assert translated["sources"][0]["adapter"] == "single_arm"
    assert translated["sources"][0]["datasets"] == [{"dir": "task_a", "name": "task_a"}]
    assert translated["dataset_weights"]["source_files"] == {"single": 1.0}

    local_root = tmp_path / "local_data"
    local_dataset = local_root / "task_a"
    (local_dataset / "meta").mkdir(parents=True)
    (local_dataset / "meta/info.json").write_text("{}", encoding="utf-8")
    payload_path = local_dataset / "payload.bin"
    payload_path.write_bytes(b"local-only")
    local_config, local_specs = petrel_dataset.reader.load_dataset_specs(config_path, local_root=str(local_root))

    assert local_config["storage"]["local_root"] == str(local_root)
    assert local_specs[0].remote_root == str(local_dataset)
    assert local_specs[0].local_root == str(local_root)
    assert local_specs[0].format_version is None
    assert local_specs[0].local_meta_dir.endswith("cache/meta/task_a")
    assert local_specs[0].path_index_dir.endswith("cache/task_a")

    dataset = object.__new__(petrel_dataset.reader.PetrelLeRobotDataset)
    dataset.spec = dataclasses.replace(local_specs[0], local_meta_dir=str(tmp_path / "meta_cache"))
    dataset.name = local_specs[0].name
    dataset.remote_root = local_specs[0].remote_root
    dataset._client = None  # noqa: SLF001
    assert isinstance(dataset.client, petrel_dataset.reader.LocalFileClient)
    assert dataset.client.get_bytes(str(payload_path)) == b"local-only"
    outside_path = tmp_path / "outside.bin"
    outside_path.write_bytes(b"outside")
    with np.testing.assert_raises(ValueError):
        dataset.client.get_bytes(str(outside_path))

    first_read = dataset._read_bytes("meta/info.json")  # noqa: SLF001
    (local_dataset / "meta/info.json").write_text('{"changed": true}', encoding="utf-8")
    second_read = dataset._read_bytes("meta/info.json")  # noqa: SLF001
    assert first_read == second_read == b"{}"
    assert (tmp_path / "meta_cache/info.json").read_bytes() == b"{}"


def test_openpi_petrel_metadata_does_not_scan_dataset_groups(tmp_path):
    config_path = tmp_path / "metadata_only.yaml"
    config_path.write_text(
        """
schema: openpi_petrel_v1
dataset_id: metadata_only
openpi:
  state_dim: 32
  action_dim: 32
  action_horizon: 16
  control_projection: mapped
  normalization: quantile
  images:
    size: [224, 224]
    slots: [base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb]
dataset_files:
  directory: directory_that_does_not_exist
""".strip(),
        encoding="utf-8",
    )

    dataset_id, contract = petrel_dataset.reader.read_openpi_petrel_metadata(config_path)

    assert dataset_id == "metadata_only"
    assert contract["action_dim"] == 32
    assert contract["action_horizon"] == 16


def test_pi05_petrel_legacy_config_is_rejected_until_mapped_migration():
    with pytest.raises(ValueError, match="control_projection must be 'mapped'"):
        petrel_dataset.reader.read_openpi_petrel_config("configs/petrel/pi05_petrel.yaml")


def test_pi05_interna1_yaml_only_expands_intern_a1():
    config = petrel_dataset.reader.read_openpi_petrel_config("configs/petrel/pi05_interna1.yaml")
    specs = petrel_dataset.reader._flatten_source_specs(config)  # noqa: SLF001

    assert config["dataset_id"] == "intern_a1"
    assert config["openpi"]["control_projection"] == "mapped"
    assert config["dataset_weights"]["source_files"] == {"intern_a1": 1.0}
    assert set(config["embodiments"]) == {"a1_single_ee", "a1_dual_ee", "a1_dual7_ee"}
    assert specs
    assert {spec.stats_group for spec in specs} == {"dataset/intern_a1"}
    assert {spec.stats_group_fallbacks for spec in specs} == {()}
    assert {source["_source_file"] for source in config["sources"]} == {"intern_a1"}
    assert config["canonical_action_space"]["action_semantics"] == {
        "joint": "absolute",
        "gripper": "absolute",
        "ee": "delta",
    }

    single_profile = config["embodiments"]["a1_single_ee"]["canonical_adapter"]
    assert set(single_profile["slots"]["right"]) == {"joint", "gripper"}
    assert single_profile["slots"]["right"]["joint"]["state_target_indices"] == [7, 8, 9, 10, 11, 12, 27]

    dual7_profile = config["embodiments"]["a1_dual7_ee"]["canonical_adapter"]
    assert set(dual7_profile["slots"]["left"]) == {"joint", "gripper"}
    assert set(dual7_profile["slots"]["right"]) == {"joint", "gripper"}
    assert dual7_profile["slots"]["left"]["joint"]["state_target_indices"] == [0, 1, 2, 3, 4, 5, 26]
    assert dual7_profile["slots"]["right"]["joint"]["state_target_indices"] == [7, 8, 9, 10, 11, 12, 27]

    articulation = next(source for source in config["sources"] if source["name"] == "a1_articulation_franka")
    assert articulation["adapter"] == "a1_single_ee"
    assert articulation["canonical_adapter"]["frame_transform"]["world_from_source"] == "robot_to_opencv"
    assert articulation["canonical_adapter"]["frame_transform"]["source_ee_from_canonical_ee"] == [
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]

    articulation_spec = next(spec for spec in specs if spec.source_name == "a1_articulation_franka")
    assert articulation_spec.canonical_adapter["slots"]["right"]["joint"]["state_key"] == "states.joint.position"
    assert (
        articulation_spec.canonical_adapter["frame_transform"] == articulation["canonical_adapter"]["frame_transform"]
    )
    assert articulation_spec.modalities["video"]["key_map"] == {
        "images.rgb.head": "base_0_rgb",
        "images.rgb.hand": "right_wrist_0_rgb",
    }


def test_interna1_camera_mapping_does_not_depend_on_video_key_order():
    adapter = object.__new__(petrel_dataset.PetrelDataset)
    adapter._camera_key_map = {}  # noqa: SLF001

    franka = SimpleNamespace(
        video_keys=["images.rgb.unmapped", "images.rgb.hand", "images.rgb.head"],
        modalities={
            "video": {
                "key_map": {
                    "images.rgb.head": "base_0_rgb",
                    "images.rgb.hand": "right_wrist_0_rgb",
                }
            }
        },
    )
    dual_arm = SimpleNamespace(
        video_keys=["images.rgb.hand_right", "images.rgb.head", "images.rgb.hand_left"],
        modalities={
            "video": {
                "key_map": {
                    "images.rgb.head": "base_0_rgb",
                    "images.rgb.hand_left": "left_wrist_0_rgb",
                    "images.rgb.hand_right": "right_wrist_0_rgb",
                }
            }
        },
    )

    assert adapter._camera_assignments(franka) == [  # noqa: SLF001
        ("images.rgb.hand", "right_wrist_0_rgb"),
        ("images.rgb.head", "base_0_rgb"),
    ]
    assert adapter._camera_assignments(dual_arm) == [  # noqa: SLF001
        ("images.rgb.hand_right", "right_wrist_0_rgb"),
        ("images.rgb.head", "base_0_rgb"),
        ("images.rgb.hand_left", "left_wrist_0_rgb"),
    ]


def test_interna1_groups_use_registered_control_profiles():
    group_dir = pathlib.Path("configs/petrel/dataset_groups/intern_a1")
    group_payloads = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in group_dir.glob("*.yaml")]

    assert len(group_payloads) == 44
    assert all("control_profiles" not in payload for payload in group_payloads)
    assert {payload["profile"] for payload in group_payloads} == {
        "a1_single_ee",
        "a1_dual_ee",
        "a1_dual7_ee",
    }


def test_pi05_abc130k_config_and_manifest():
    dataset_id, contract = petrel_dataset.reader.read_openpi_petrel_metadata("configs/petrel/pi05_abc130k.yaml")
    translated = petrel_dataset.reader.read_openpi_petrel_config("configs/petrel/pi05_abc130k.yaml")

    assert dataset_id == "abc130k"
    assert contract["control_projection"] == "mapped"
    assert translated["source_control_space"]["dim"] == 32
    specs = petrel_dataset.reader._flatten_source_specs(translated)  # noqa: SLF001
    assert {spec.stats_group for spec in specs} == {"dataset/abc130k"}
    assert {spec.stats_group_fallbacks for spec in specs} == {()}
    assert all("data/petrel_cache/meta/abc130k/" in str(spec.local_meta_dir) for spec in specs)
    assert all("data/petrel_cache/path_index/abc130k/" in str(spec.path_index_dir) for spec in specs)
    assert all("data/petrel_cache/filter/abc130k/" in str(spec.frame_filter_dir) for spec in specs)
    assert all(spec.frame_filter_enabled for spec in specs)
    assert all(spec.frame_filter_required for spec in specs)
    assert all(spec.frame_filter_strict_episode_coverage for spec in specs)
    assert {spec.frame_filter_filename for spec in specs} == {"frame_filter.jsonl"}
    assert {spec.frame_filter_window_policy for spec in specs} == {"full_chunk"}
    adapter = translated["embodiments"]["abc130k_3cam"]["canonical_adapter"]
    assert adapter["slots"]["left"]["joint"]["state_target_indices"] == [0, 1, 2, 3, 4, 5]
    assert adapter["slots"]["right"]["joint"]["state_target_indices"] == [7, 8, 9, 10, 11, 12]
    assert adapter["slots"]["left"]["ee"]["state_target_indices"] == [14, 15, 16, 17, 18, 19]
    assert adapter["slots"]["right"]["ee"]["action_target_indices"] == [20, 21, 22, 23, 24, 25]
    assert adapter["delta_from_state_indices"] == [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    manifests = {
        path.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in pathlib.Path("configs/petrel/dataset_groups/abc130k").glob("*.yaml")
    }
    assert manifests["abc130k.yaml"]["profile"] == "abc130k_3cam"
    assert manifests["abc130k_4cam.yaml"]["profile"] == "abc130k_4cam_top_left"
    assert len(manifests["abc130k.yaml"]["datasets"]) == 398
    assert len(manifests["abc130k_4cam.yaml"]["datasets"]) == 87

    datasets = [dataset for manifest in manifests.values() for dataset in manifest["datasets"]]
    assert len(datasets) == 485
    assert len({dataset["name"] for dataset in datasets}) == 485
    assert len({dataset["path"] for dataset in datasets}) == 485

    profile_registry = yaml.safe_load(
        pathlib.Path("configs/petrel/control_profiles/abc130k.yaml").read_text(encoding="utf-8")
    )
    profiles = profile_registry["control_profiles"]
    assert profiles["abc130k_3cam"]["fields"]["video"]["key_map"]["observation.images.top_camera"] == ("base_0_rgb")
    assert profiles["abc130k_4cam_top_left"]["fields"]["video"]["key_map"] == {
        "observation.images.top_left_camera": "base_0_rgb",
        "observation.images.left_wrist_camera": "left_wrist_0_rgb",
        "observation.images.right_wrist_camera": "right_wrist_0_rgb",
    }
    assert profiles["abc130k_4cam_top_right"]["fields"]["video"]["key_map"] == {
        "observation.images.top_right_camera": "base_0_rgb",
        "observation.images.left_wrist_camera": "left_wrist_0_rgb",
        "observation.images.right_wrist_camera": "right_wrist_0_rgb",
    }


def test_pi05_abc130k_debug_uses_local_mount_and_separate_assets():
    from openpi.training import config as training_config

    train_config = training_config.get_config("pi05_abc130k_debug")
    translated = petrel_dataset.reader.read_openpi_petrel_config("configs/petrel/pi05_abc130k.yaml")

    assert train_config.data.repo_id == "abc130k"
    assert train_config.model.action_horizon == 30
    # The debug variant reuses the shared YAML and only overrides the mount point.
    assert train_config.data.dataset.dataset_config == "configs/petrel/pi05_abc130k.yaml"
    assert train_config.data.dataset.local_root == "/oss/datahub/public/real/ABC-130K-v3/train"
    assert train_config.pytorch_weight_path == "/oss/datahub/public/model/pi05_base_pytorch_bf16"
    assert translated["mixture"]["training_block_size"] == 1
    assert train_config.assets_dirs / train_config.data.repo_id == (
        pathlib.Path.cwd() / "assets/pi05_abc130k_debug/abc130k"
    )


def test_frame_filter_limits_windows_without_subtask_segments():
    reader = petrel_dataset.reader
    dataset = reader.PetrelLeRobotDataset.__new__(reader.PetrelLeRobotDataset)
    dataset.spec = SimpleNamespace(frame_filter_enabled=True, frame_filter_window_policy="full_chunk")
    dataset.episodes_dict = {0: {"length": 100, "start_frame": 0}}
    dataset.subtask_action_segments = {}
    dataset.frame_filter_clean_segments = {0: ((10, 50), (60, 100))}
    dataset.max_delta_index = 29
    dataset.trajectory_ids = np.asarray([0], dtype=np.int64)
    dataset.trajectory_lengths = np.asarray([100], dtype=np.int64)

    assert dataset._valid_start_intervals_for_episode(0, 100) == ((10, 20), (60, 70))  # noqa: SLF001
    assert dataset.valid_start_counts().tolist() == [22]
    assert dataset.is_valid_start_for_trajectory_pos(0, 10)
    assert dataset.is_valid_start_for_trajectory_pos(0, 20)
    assert not dataset.is_valid_start_for_trajectory_pos(0, 21)
    assert not dataset.is_valid_start_for_trajectory_pos(0, 55)
    assert dataset.is_valid_start_for_trajectory_pos(0, 60)
    norm_starts = petrel_norm_stats._valid_start_indices(  # noqa: SLF001
        dataset,
        {"trajectory_pos": 0, "episode_index": 0, "windows": 22, "blocks": None},
        allow_padding=False,
    )
    assert norm_starts.tolist() == list(range(10, 21)) + list(range(60, 71))

    dataset.frame_filter_clean_segments = {0: ()}
    assert dataset._valid_start_intervals_for_episode(0, 100) == ()  # noqa: SLF001
    assert dataset.valid_start_counts().tolist() == [0]


def test_abc130k_camera_mapping_uses_profile_selected_stereo_top_view():
    dataset = SimpleNamespace(
        video_keys=[
            "observation.images.top_right_camera",
            "observation.images.left_wrist_camera",
            "observation.images.right_wrist_camera",
        ],
        modalities={
            "video": {
                "key_map": {
                    "observation.images.top_right_camera": "base_0_rgb",
                    "observation.images.left_wrist_camera": "left_wrist_0_rgb",
                    "observation.images.right_wrist_camera": "right_wrist_0_rgb",
                }
            }
        },
    )
    adapter = object.__new__(petrel_dataset.PetrelDataset)
    adapter._camera_key_map = {}  # noqa: SLF001

    assert adapter._camera_assignments(dataset) == [  # noqa: SLF001
        ("observation.images.top_right_camera", "base_0_rgb"),
        ("observation.images.left_wrist_camera", "left_wrist_0_rgb"),
        ("observation.images.right_wrist_camera", "right_wrist_0_rgb"),
    ]


def test_pi05_petrel_data_config_create_uses_yaml_contract(tmp_path):
    from openpi.training import config as training_config

    train_config = training_config.get_config("pi05_petrel")
    data_factory = dataclasses.replace(
        train_config.data,
        model_transforms=training_config.IdentityTransformFactory(),
    )
    with pytest.raises(ValueError, match="control_projection must be 'mapped'"):
        data_factory.create(tmp_path, train_config.model)


def test_compute_norm_stats_petrel_config_does_not_build_tokenizer(monkeypatch):
    from openpi.models import tokenizer
    from openpi.training import config as training_config
    from scripts import compute_norm_stats

    monkeypatch.setattr(
        tokenizer,
        "PaligemmaTokenizer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("tokenizer must not be initialized")),
    )
    train_config = training_config.get_config("pi05_interna1")
    data_config = compute_norm_stats.create_data_config(train_config)

    assert data_config.repo_id == "intern_a1"
    assert data_config.petrel is not None
    assert data_config.model_transforms == training_config.IdentityTransformFactory()(train_config.model)


def test_pi05_interna1_data_config_uses_separate_norm_asset(tmp_path):
    from openpi.training import config as training_config

    train_config = training_config.get_config("pi05_interna1")
    data_factory = dataclasses.replace(
        train_config.data,
        model_transforms=training_config.IdentityTransformFactory(),
    )
    data_config = data_factory.create(tmp_path, train_config.model)

    assert data_config.repo_id == "intern_a1"
    assert data_config.asset_id == "intern_a1"
    assert data_config.petrel is not None
    assert data_config.petrel.dataset_config == "configs/petrel/pi05_interna1.yaml"
    assert train_config.pytorch_weight_path == "/datahub/public/model/pi05_base_pytorch_bf16"
    assert train_config.pytorch_training_precision == "bfloat16"


def test_norm_stats_merge_multiple_dataset_batches_into_standard_global_format():
    from openpi.shared import normalize

    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats(feature_shape=(2, 2))
    # These two updates represent two different datasets selected by the mixture.
    state_stats.update(np.asarray([[0.0, 2.0], [2.0, 4.0]]))
    state_stats.update(np.asarray([[4.0, 6.0], [6.0, 8.0]]))
    action_stats.update(np.asarray([[[0.0, 1.0], [2.0, 3.0]]]))
    action_stats.update(np.asarray([[[4.0, 5.0], [6.0, 7.0]]]))

    payload = normalize.serialize_json(
        {"state": state_stats.get_statistics(), "actions": action_stats.get_statistics()}
    )
    restored = normalize.deserialize_json(payload)

    assert set(restored) == {"state", "actions"}
    np.testing.assert_allclose(restored["state"].mean, np.asarray([3.0, 5.0]))
    np.testing.assert_allclose(restored["actions"].mean, np.asarray([[2.0, 3.0], [4.0, 5.0]]))
    assert restored["state"].q01.shape == (2,)
    assert restored["actions"].q99.shape == (2, 2)


def test_pytorch_torch_dataloader_does_not_initialize_jax_backend(monkeypatch):
    from openpi.training import data_loader

    class CpuDataset:
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {"state": np.asarray([index], dtype=np.float32)}

    monkeypatch.setattr(
        data_loader.jax,
        "process_count",
        lambda: (_ for _ in ()).throw(AssertionError("JAX backend must not be initialized")),
    )
    loader = data_loader.TorchDataLoader(
        CpuDataset(),
        local_batch_size=2,
        num_batches=1,
        framework="pytorch",
    )

    batch = next(iter(loader))

    assert tuple(batch["state"].shape) == (2, 1)


def test_transformed_dataset_forwards_batch_and_epoch():
    from openpi.training import data_loader

    class BatchDataset:
        prefer_sequential_indices = True
        requires_worker_restart_each_epoch = True

        def __init__(self):
            self.epoch = None

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {"value": np.asarray(index)}

        def __getitems__(self, indices):
            return [{"value": np.asarray(index)} for index in indices]

        def set_epoch(self, epoch):
            self.epoch = epoch

    raw_dataset = BatchDataset()
    raw_dataset.frame_filter_identity = "filter-identity"
    raw_dataset.frame_filter_summary = {"enabled": True, "leaves": 2}
    dataset = data_loader.TransformedDataset(raw_dataset, [lambda sample: {"value": sample["value"] + 1}])

    assert [int(sample["value"]) for sample in dataset.__getitems__([0, 2])] == [1, 3]
    assert dataset.prefer_sequential_indices
    assert dataset.requires_worker_restart_each_epoch
    assert dataset.frame_filter_identity == "filter-identity"
    assert dataset.frame_filter_summary == {"enabled": True, "leaves": 2}
    dataset.set_epoch(7)
    assert raw_dataset.epoch == 7
