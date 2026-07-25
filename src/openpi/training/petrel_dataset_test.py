import dataclasses
from types import SimpleNamespace

import numpy as np
import torch

from openpi.training import petrel_dataset


class _FakePetrelDataset:
    def __init__(self, spec, **kwargs):
        del kwargs
        self.name = spec.name
        self.state_target_dim = 4
        self.action_target_dim = 4
        self.video_keys = ["head", "left_wrist"]
        self.fps = 10.0

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

    def __getitem__(self, index):
        return {
            "dataset_name": "fake",
            "dataset_index": torch.tensor(0),
            "episode_index": torch.tensor(2),
            "frame_index": torch.tensor([index]),
            "observation.state": torch.arange(4, dtype=torch.float32)[None],
            "action": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "observation.state_dim_is_pad": torch.tensor([False, True, False, True]),
            "action_dim_is_pad": torch.tensor([False, True, False, True]),
            "task": "pick up the cup",
        }


def _fake_reader(action_dim=8):
    return SimpleNamespace(
        load_dataset_specs=lambda path, conf: (
            {
                "openpi": {
                    "state_dim": action_dim,
                    "action_dim": action_dim,
                    "action_horizon": 3,
                    "normalization": "quantile",
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
    np.testing.assert_array_equal(sample["state"][:2], np.asarray([0.0, 2.0]))
    np.testing.assert_array_equal(sample["state"][2:], np.zeros(6))
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

    assert set(sample) == {"state", "actions", "prompt"}
    assert len(samples) == 2
    assert all(set(item) == {"state", "actions", "prompt"} for item in samples)


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


def test_batched_petrel_projection_matches_openpi_shape():
    item = {
        "dataset_name": "batched",
        "observation.state": torch.arange(12, dtype=torch.float32).reshape(3, 1, 4),
        "action": torch.arange(36, dtype=torch.float32).reshape(3, 3, 4),
        "observation.state_dim_is_pad": torch.tensor([False, True, False, True]),
        "action_dim_is_pad": torch.tensor([False, True, False, True]),
    }

    state, actions = petrel_dataset.project_petrel_features(item, output_dim=8)

    assert state.shape == (3, 1, 8)
    assert actions.shape == (3, 3, 8)
    torch.testing.assert_close(state[..., 2:], torch.zeros(3, 1, 6))


def test_compact_projection_preserves_all_actions_before_state_only_dims(monkeypatch, tmp_path):
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text("mixture: {}", encoding="utf-8")
    fake = _fake_reader(action_dim=32)
    for name, value in vars(fake).items():
        monkeypatch.setattr(petrel_dataset.reader, name, value)
    dataset = petrel_dataset.PetrelDataset(
        petrel_dataset.PetrelDatasetConfig(dataset_config=str(config_path), skip_bad_videos=False),
        action_horizon=3,
        model_action_dim=32,
    )
    item = {
        "dataset_name": "dual_arm",
        "observation.state": torch.arange(34, dtype=torch.float32)[None],
        "action": torch.arange(3 * 34, dtype=torch.float32).reshape(3, 34),
        "observation.state_dim_is_pad": torch.zeros(34, dtype=torch.bool),
        "action_dim_is_pad": torch.tensor([False] * 28 + [True] * 6),
    }

    state, actions = dataset._project_features(item)  # noqa: SLF001

    assert state.shape == (1, 32)
    assert actions.shape == (3, 32)
    torch.testing.assert_close(state[0], torch.arange(32, dtype=torch.float32))
    torch.testing.assert_close(actions[0, :28], torch.arange(28, dtype=torch.float32))
    torch.testing.assert_close(actions[0, 28:], torch.zeros(4))


def test_petrel_s3_url_parser():
    target = petrel_dataset.reader._parse_petrel_s3_url(  # noqa: SLF001
        "manip:bucket/path/to/meta/info.json"
    )

    assert target.section == "manip"
    assert target.bucket == "bucket"
    assert target.key == "path/to/meta/info.json"


def test_openpi_petrel_v1_schema_translation(tmp_path):
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
      layout: robot
      slots:
        right:
          joint: {state_key: observation.state, action_key: action, target_slice: [40, 47]}
    fields:
      state: {keys: [observation.state]}
      action: {keys: [action]}
      video: {key: auto}
format: v3
root: example:s3://bucket/root
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
  control_projection: compact_valid
  normalization: quantile
  images:
    size: [224, 224]
    dtype: uint8
    channels: rgb
    slots: [base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb]
storage:
  petrel_conf: ~/petreloss.conf
  parquet_cache_size: 2
sampling:
  state_frames: 1
  action_horizon: 16
  stride: 1
  allow_end_padding: false
mixture:
  category_weights: {single: 1.0}
  distribute_category_weight_by_valid_windows: true
source_control_space: {name: robot, dim: 80}
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
  control_projection: compact_valid
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


def test_pi05_petrel_yaml_expands_all_repository_datasets():
    config, specs = petrel_dataset.reader.load_dataset_specs("configs/petrel/pi05_petrel.yaml")

    assert config["dataset_id"] == "petrel_mixed"
    assert len(specs) == 1785
    assert all(spec.remote_root for spec in specs)
    assert all(spec.modalities for spec in specs)
    assert all(spec.canonical_adapter for spec in specs)
    assert all(spec.dataset_weight > 0 for spec in specs)
    assert {spec.source_name for spec in specs} >= {
        "a1_articulation_franka",
        "intern_m1",
        "bridgev2",
        "agibotworld",
        "galaxea_no_mobile_torso",
        "oxe_quat",
    }


def test_pi05_interna1_yaml_only_expands_intern_a1():
    config, specs = petrel_dataset.reader.load_dataset_specs("configs/petrel/pi05_interna1.yaml")

    assert config["dataset_id"] == "intern_a1"
    assert config["dataset_weights"]["source_files"] == {"intern_a1": 1.0}
    assert specs
    assert {source["_source_file"] for source in config["sources"]} == {"intern_a1"}


def test_pi05_petrel_data_config_create_uses_yaml_contract(tmp_path):
    from openpi.training import config as training_config

    train_config = training_config.get_config("pi05_petrel")
    data_factory = dataclasses.replace(
        train_config.data,
        model_transforms=training_config.IdentityTransformFactory(),
    )
    data_config = data_factory.create(tmp_path, train_config.model)

    assert data_config.repo_id == "petrel_mixed"
    assert data_config.petrel is not None
    assert train_config.model.action_dim == 32
    assert train_config.model.action_horizon == 16
    assert tmp_path / data_config.repo_id == tmp_path / "petrel_mixed"


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


def test_norm_stats_merge_multiple_dataset_batches_into_standard_global_format():
    from openpi.shared import normalize

    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
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
    np.testing.assert_allclose(restored["actions"].mean, np.asarray([3.0, 4.0]))
    assert restored["state"].q01.shape == (2,)
    assert restored["actions"].q99.shape == (2,)


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
    dataset = data_loader.TransformedDataset(raw_dataset, [lambda sample: {"value": sample["value"] + 1}])

    assert [int(sample["value"]) for sample in dataset.__getitems__([0, 2])] == [1, 3]
    assert dataset.prefer_sequential_indices
    assert dataset.requires_worker_restart_each_epoch
    dataset.set_epoch(7)
    assert raw_dataset.epoch == 7
