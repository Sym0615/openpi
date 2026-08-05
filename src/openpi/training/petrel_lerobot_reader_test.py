import math
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from openpi.training import petrel_lerobot_reader as reader


def test_parquet_row_groups_for_episode_uses_episode_statistics(tmp_path):
    path = tmp_path / "episodes.parquet"
    schema = pa.schema(
        [
            ("episode_index", pa.int64()),
            ("index", pa.int64()),
            ("action", pa.float32()),
        ]
    )
    with pq.ParquetWriter(path, schema) as writer:
        writer.write_table(pa.table({"episode_index": [3, 3], "index": [10, 11], "action": [1.0, 2.0]}, schema=schema))
        writer.write_table(
            pa.table({"episode_index": [4, 4, 4], "index": [12, 13, 14], "action": [3.0, 4.0, 5.0]}, schema=schema)
        )

    parquet_file = pq.ParquetFile(path)

    assert reader._parquet_row_groups_for_episode(  # noqa: SLF001
        parquet_file,
        source_episode_index=4,
        index_bounds=(12, 15),
    ) == [1]
    table = parquet_file.read_row_groups([1])
    assert table["episode_index"].to_pylist() == [4, 4, 4]


def test_parquet_row_groups_for_episode_returns_empty_for_missing_episode(tmp_path):
    path = tmp_path / "episodes.parquet"
    pq.write_table(pa.table({"episode_index": [0, 0], "index": [0, 1]}), path)

    assert (
        reader._parquet_row_groups_for_episode(  # noqa: SLF001
            pq.ParquetFile(path),
            source_episode_index=9,
            index_bounds=(100, 101),
        )
        == []
    )


def test_parquet_row_groups_for_episode_falls_back_to_index_statistics(tmp_path):
    path = tmp_path / "index_only_statistics.parquet"
    table = pa.table(
        {
            "episode_index": [3, 3, 4, 4],
            "index": [10, 11, 12, 13],
        }
    )
    pq.write_table(table, path, row_group_size=2, write_statistics=["index"])

    assert reader._parquet_row_groups_for_episode(  # noqa: SLF001
        pq.ParquetFile(path),
        source_episode_index=4,
        index_bounds=(12, 14),
    ) == [1]


def test_parquet_row_groups_index_statistics_by_parquet_leaf_column(tmp_path):
    """Multi-leaf columns shift every later leaf ordinal away from the Arrow field ordinal."""
    path = tmp_path / "multi_leaf.parquet"
    schema = pa.schema(
        [
            ("meta", pa.struct([("a", pa.int64()), ("b", pa.int64())])),
            ("episode_index", pa.int64()),
            ("index", pa.int64()),
        ]
    )
    with pq.ParquetWriter(path, schema) as writer:
        writer.write_table(pa.table({"meta": [{"a": 0, "b": 1}], "episode_index": [3], "index": [10]}, schema=schema))
        writer.write_table(pa.table({"meta": [{"a": 2, "b": 3}], "episode_index": [4], "index": [11]}, schema=schema))

    parquet_file = pq.ParquetFile(path)
    # The Arrow field ordinal and the parquet leaf ordinal disagree here.
    assert parquet_file.schema_arrow.names.index("episode_index") == 1
    assert parquet_file.metadata.row_group(0).column(1).path_in_schema == "meta.b"

    assert reader._parquet_row_groups_for_episode(  # noqa: SLF001
        parquet_file,
        source_episode_index=4,
        index_bounds=(11, 12),
    ) == [1]


def test_parquet_row_groups_keep_groups_without_statistics(tmp_path):
    """A writer may drop statistics per chunk; those groups must not be skipped."""

    class _Statistics:
        def __init__(self, minimum, maximum):
            self.min = minimum
            self.max = maximum
            self.has_min_max = True

    class _RowGroup:
        def __init__(self, statistics):
            self._statistics = statistics

        def column(self, index):
            assert index == 0
            return SimpleNamespace(statistics=self._statistics)

    class _Schema:
        def __len__(self):
            return 1

        def column(self, index):
            return SimpleNamespace(path="episode_index")

    row_groups = [_RowGroup(_Statistics(3, 3)), _RowGroup(None), _RowGroup(_Statistics(4, 4))]
    metadata = SimpleNamespace(
        schema=_Schema(),
        num_row_groups=len(row_groups),
        row_group=row_groups.__getitem__,
    )

    selected = reader._parquet_row_groups_for_episode(  # noqa: SLF001
        SimpleNamespace(metadata=metadata),
        source_episode_index=4,
        index_bounds=None,
    )

    assert selected == [1, 2]


def test_local_file_client_prefers_staged_parquet(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "episode.parquet"
    source_path.write_bytes(b"source")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    client = reader.LocalFileClient(source_root, parquet_cache_dir=cache_dir)

    assert client.resolve_parquet_path(str(source_path)) == source_path

    cached_path = reader.local_parquet_cache_path(cache_dir, source_path)
    cached_path.write_bytes(b"cache!")

    assert client.resolve_parquet_path(str(source_path)) == cached_path


def test_local_file_client_ignores_a_staged_parquet_that_no_longer_matches(tmp_path):
    """A source parquet regenerated without re-staging must not keep serving stale bytes."""
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "episode.parquet"
    source_path.write_bytes(b"source")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached_path = reader.local_parquet_cache_path(cache_dir, source_path)
    cached_path.write_bytes(b"cache!")
    client = reader.LocalFileClient(source_root, parquet_cache_dir=cache_dir)
    assert client.resolve_parquet_path(str(source_path)) == cached_path

    source_path.write_bytes(b"regenerated source")

    assert client.resolve_parquet_path(str(source_path)) == source_path


def test_staged_parquet_cache_path_survives_a_symlinked_mount(tmp_path):
    """The stager hashes the path index entry; the reader hashes it after resolve()."""
    source_root = tmp_path / "datahub"
    source_root.mkdir()
    source_path = source_root / "episode.parquet"
    source_path.write_bytes(b"source")
    alias_root = tmp_path / "oss_datahub"
    alias_root.symlink_to(source_root, target_is_directory=True)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # The staging tool only sees the alias recorded in path_index.jsonl.
    staged_path = reader.local_parquet_cache_path(cache_dir, alias_root / "episode.parquet")
    staged_path.write_bytes(b"source")

    client = reader.LocalFileClient(source_root, parquet_cache_dir=cache_dir)
    assert client.resolve_parquet_path(str(alias_root / "episode.parquet")) == staged_path


def _control_space(action_semantics):
    return {
        "name": "test_layout",
        "dim": 32,
        "missing_policy": "zero",
        "camera_frame_delta": False,
        "pose": {"state": "xyz_rotvec", "action": "xyz_rotvec"},
        "slot_indices": {"left": {"ee": list(range(14, 20))}},
        "action_semantics": action_semantics,
    }


def _ee_mapping():
    return {"slots": {"left": {"ee": {"target_slot": "left.ee", "state_key": "state.ee", "action_key": "action.ee"}}}}


def test_ee_actions_default_to_delta_from_state():
    """Omitting ee from action_semantics must not flip EE targets to absolute world poses."""
    adapter = reader._resolve_control_adapter(  # noqa: SLF001
        _ee_mapping(), _control_space({}), profile_name="test"
    )

    assert adapter["slots"]["left"]["ee"]["target_action_semantics"] == "delta"


def test_ee_actions_honour_an_explicit_absolute_declaration():
    adapter = reader._resolve_control_adapter(  # noqa: SLF001
        _ee_mapping(), _control_space({"ee": "absolute"}), profile_name="test"
    )

    assert adapter["slots"]["left"]["ee"]["target_action_semantics"] == "absolute"


def test_table_to_tensors_handles_fixed_lists_without_changing_values():
    table = pa.table(
        {
            "vector": pa.array([[1.0, 2.0], [3.0, 4.0]], type=pa.list_(pa.float64(), 2)),
            "index": pa.array([5, 6], type=pa.int64()),
            "label": pa.array(["a", "b"]),
        }
    )

    converted = reader._table_to_tensors(table)  # noqa: SLF001

    torch.testing.assert_close(converted["vector"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    torch.testing.assert_close(converted["index"], torch.tensor([5, 6]))
    assert converted["label"] == ["a", "b"]


def test_matrix_to_rotvec_stays_accurate_across_all_shepperd_branches():
    """Each of the four quaternion derivations must be selected and stay accurate.

    Near theta = pi the trace derivation degenerates, so the largest-diagonal
    branch has to take over; an acos-based rotvec is off by ~2 rad there. This
    pins that behaviour for every branch, including the exactly-180-degree case.
    """
    axes = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # largest diagonal is m00
            [0.0, 1.0, 0.0],  # m11
            [0.0, 0.0, 1.0],  # m22
            [1.0, -2.0, 3.0],
            [-3.0, 1.0, 0.5],
        ]
    )
    axes = axes / axes.norm(dim=-1, keepdim=True)
    for angle in (1e-6, 0.5, math.pi / 2, math.pi - 1e-3, math.pi - 1e-6, math.pi):
        rotations = reader._rotvec_to_matrix(axes * angle)  # noqa: SLF001

        recovered = reader._matrix_to_rotvec(rotations)  # noqa: SLF001

        assert bool(torch.isfinite(recovered).all()), angle
        torch.testing.assert_close(
            reader._rotvec_to_matrix(recovered),  # noqa: SLF001
            rotations,
            rtol=1e-5,
            atol=1e-5,
            msg=lambda text, angle=angle: f"angle={angle}: {text}",
        )


def test_matrix_to_rotvec_selects_the_diagonal_branch_away_from_the_trace_branch():
    """Guards the branchless torch.where selection against a swapped predicate."""
    identity = reader._matrix_to_rotvec(torch.eye(3)[None])  # noqa: SLF001
    torch.testing.assert_close(identity, torch.zeros(1, 3))

    # trace < 0 for every 180-degree rotation, so the trace branch must not win.
    for index in range(3):
        axis = torch.zeros(1, 3)
        axis[0, index] = math.pi
        rotation = reader._rotvec_to_matrix(axis)  # noqa: SLF001
        assert float(rotation[0].diagonal().sum()) < 0.0

        recovered = reader._matrix_to_rotvec(rotation)  # noqa: SLF001

        torch.testing.assert_close(recovered.abs(), axis.abs(), rtol=1e-5, atol=1e-5)


def test_matrix_to_rotvec_round_trips_near_pi():
    axes = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, -2.0, 3.0],
        ]
    )
    axes = axes / axes.norm(dim=-1, keepdim=True)
    rotations = reader._rotvec_to_matrix(axes * (math.pi - 1e-5))  # noqa: SLF001

    recovered = reader._matrix_to_rotvec(rotations)  # noqa: SLF001

    torch.testing.assert_close(
        reader._rotvec_to_matrix(recovered),  # noqa: SLF001
        rotations,
        rtol=2e-5,
        atol=2e-5,
    )
    assert bool((recovered.norm(dim=-1) <= math.pi + 1e-5).all())


def test_absolute_pose_delta_stays_small_near_pi_after_frame_transform():
    # Regression values from store_the_eggs_part2, episode 936, frames 526-527.
    current = torch.tensor([[0.0, 0.0, 0.0, 0.72425735, -0.68856251, -0.00129203, -0.03648711]])
    target = torch.tensor([[0.0, 0.0, 0.0, 0.72425741, -0.68856239, -0.00129200, -0.03648736]])
    frame_transform = {
        "world_from_source": "robot_to_opencv",
        "source_ee_from_canonical_ee": [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
    }
    current_xyz, current_rotation = reader._pose_to_xyz_matrix(current, "xyz_quat_wxyz")  # noqa: SLF001
    target_xyz, target_rotation = reader._pose_to_xyz_matrix(target, "xyz_quat_wxyz")  # noqa: SLF001
    current_xyz, current_rotation = reader._transform_xyz_matrix_pose(  # noqa: SLF001
        current_xyz, current_rotation, frame_transform, is_delta=False
    )
    target_xyz, target_rotation = reader._transform_xyz_matrix_pose(  # noqa: SLF001
        target_xyz, target_rotation, frame_transform, is_delta=False
    )

    delta_xyz, delta_rotation = reader._pose_delta_world_from_transformed_xyz_matrix(  # noqa: SLF001
        (current_xyz, current_rotation), (target_xyz, target_rotation)
    )
    delta = reader._format_xyz_matrix_pose(delta_xyz, delta_rotation, "xyz_rotvec")  # noqa: SLF001

    assert float(delta[..., 3:].norm(dim=-1).max()) < 2e-5


def test_rot6d_output_is_formatted_directly_from_rotation_matrix():
    rotvec = torch.tensor([[math.pi - 1e-5, 0.0, 0.0]])
    rotation = reader._rotvec_to_matrix(rotvec)  # noqa: SLF001
    xyz = torch.tensor([[1.0, 2.0, 3.0]])

    output = reader._format_xyz_matrix_pose(xyz, rotation, "xyz_rot6d")  # noqa: SLF001

    torch.testing.assert_close(output[..., :3], xyz)
    torch.testing.assert_close(output[..., 3:], rotation[..., :, :2].transpose(-1, -2).reshape(1, 6))
