#!/usr/bin/env python3
"""Build compact ABC-130K training filter sidecars from filter audit outputs.

The script validates every filter episode against the corresponding local path
index before publishing any sidecars.  Output paths mirror the path-index cache:

    data/petrel_cache/filter/abc130k/<task>/schema_<schema>/
        frame_filter.jsonl
        frame_filter_manifest.json

If validation fails, no sidecars are published and validation_errors.jsonl is
written at the output root.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/petrel/pi05_abc130k.yaml"
DEFAULT_FILTER_ROOT = Path("/datahub/public/meta/filter/ABC-130K-v3/train")
DEFAULT_PATH_INDEX_ROOT = REPO_ROOT / "data/petrel_cache/path_index/abc130k"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/petrel_cache/filter/abc130k"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mode", choices=["audit", "task-selection"], default="audit")
    parser.add_argument("--dataset-root", type=Path, default=Path("/dataset/public/sim/abc-sim"))
    parser.add_argument(
        "--selection-root", type=Path, default=REPO_ROOT / "data/abc_bottles_petrel_probe"
    )
    parser.add_argument("--task-name", default="sim_put_the_plastic_bottles_in_the_bin")
    parser.add_argument("--action-horizon", type=int, default=30, help="task-selection mode only")
    parser.add_argument("--stride", type=int, default=1, help="task-selection mode only")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--filter-root", type=Path, default=DEFAULT_FILTER_ROOT)
    parser.add_argument("--path-index-root", type=Path, default=DEFAULT_PATH_INDEX_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--expected-leaves",
        type=int,
        default=485,
        help="Fail validation unless the config expands to this many leaves; use 0 to disable.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without writing files.")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    return payload


def _safe_path_part(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _cache_relative_path(dataset_path: str) -> Path:
    parts = [part for part in Path(dataset_path).parts if part not in {"", ".", ".."}]
    if len(parts) != 2:
        raise ValueError(f"ABC-130K leaf path must be <task>/schema=<schema>, got {dataset_path!r}")
    return Path(*(_safe_path_part(part) for part in parts))


def _expand_leaves(config_path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    config = _load_yaml(config_path)
    if config.get("dataset_id") != "abc130k":
        raise ValueError(f"Expected dataset_id=abc130k in {config_path}, got {config.get('dataset_id')!r}")
    dataset_files = config.get("dataset_files")
    if not isinstance(dataset_files, dict):
        raise TypeError(f"Missing dataset_files mapping in {config_path}")
    group_dir = Path(str(dataset_files.get("directory", "")))
    if not group_dir.is_absolute():
        group_dir = config_path.parent / group_dir
    pattern = str(dataset_files.get("pattern", "*.yaml"))
    recursive = bool(dataset_files.get("recursive", False))
    group_paths = sorted(group_dir.rglob(pattern) if recursive else group_dir.glob(pattern))
    if not group_paths:
        raise FileNotFoundError(f"No dataset group files matched {group_dir / pattern}")

    leaves: list[dict[str, str]] = []
    for group_path in group_paths:
        group = _load_yaml(group_path)
        datasets = group.get("datasets")
        if not isinstance(datasets, list):
            raise TypeError(f"datasets must be a list: {group_path}")
        for index, dataset in enumerate(datasets):
            if not isinstance(dataset, dict) or not dataset.get("path"):
                raise TypeError(f"Invalid datasets[{index}] in {group_path}")
            dataset_path = str(dataset["path"]).strip("/")
            leaves.append(
                {
                    "name": str(dataset.get("name") or dataset_path),
                    "path": dataset_path,
                    "group_file": str(group_path),
                }
            )

    paths = [leaf["path"] for leaf in leaves]
    if len(set(paths)) != len(paths):
        duplicates = sorted(path for path in set(paths) if paths.count(path) > 1)
        raise ValueError(f"Duplicate ABC-130K leaf paths: {duplicates[:10]}")
    return leaves, config


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected a JSON object at {path}:{line_number}")
            yield line_number, value


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        for row in rows:
            encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            handle.write(encoded)
            digest.update(encoded)
            count += 1
        temp_path = Path(handle.name)
    os.replace(temp_path, path)
    return count, digest.hexdigest()


def _path_matches(filter_path: Any, indexed_path: Any) -> bool:
    source = str(filter_path or "").replace("\\", "/").rstrip("/")
    indexed = str(indexed_path or "").replace("\\", "/").rstrip("/")
    return bool(source and indexed and (source == indexed or indexed.endswith("/" + source.lstrip("/"))))


def _validated_segments(raw: Any, frame_count: int) -> list[list[int]]:
    if not isinstance(raw, list):
        raise TypeError("keep_segments must be a list")
    result: list[list[int]] = []
    previous_end = -1
    for index, segment in enumerate(raw):
        if not isinstance(segment, list | tuple) or len(segment) != 2:
            raise TypeError(f"keep_segments[{index}] must contain exactly [start, end]")
        if isinstance(segment[0], bool) or isinstance(segment[1], bool):
            raise TypeError(f"keep_segments[{index}] bounds must be integers")
        start, end = int(segment[0]), int(segment[1])
        if start != segment[0] or end != segment[1]:
            raise TypeError(f"keep_segments[{index}] bounds must be integers")
        if not (0 <= start < end <= frame_count):
            raise ValueError(
                f"keep_segments[{index}]={[start, end]} is outside episode frame range [0, {frame_count})"
            )
        if start < previous_end:
            raise ValueError(f"keep_segments must be sorted and non-overlapping at index {index}")
        result.append([start, end])
        previous_end = end
    return result


def _add_error(
    errors: list[dict[str, Any]],
    *,
    leaf: str,
    code: str,
    message: str,
    episode_index: int | None = None,
    source_file: str | None = None,
    line_number: int | None = None,
) -> None:
    row: dict[str, Any] = {"leaf": leaf, "code": code, "message": message}
    if episode_index is not None:
        row["episode_index"] = episode_index
    if source_file is not None:
        row["source_file"] = source_file
    if line_number is not None:
        row["line_number"] = line_number
    errors.append(row)


def _load_path_index(path: Path, leaf: str, errors: list[dict[str, Any]]) -> dict[int, dict[str, Any]] | None:
    if not path.is_file():
        _add_error(errors, leaf=leaf, code="path_index_missing", message=f"Missing {path}")
        return None
    rows: dict[int, dict[str, Any]] = {}
    try:
        for line_number, row in _read_jsonl(path):
            if "episode_index" not in row:
                _add_error(
                    errors,
                    leaf=leaf,
                    code="path_index_episode_index_missing",
                    message="path-index row has no episode_index",
                    source_file=str(path),
                    line_number=line_number,
                )
                continue
            episode_index = int(row["episode_index"])
            if episode_index in rows:
                _add_error(
                    errors,
                    leaf=leaf,
                    code="path_index_duplicate_episode",
                    message=f"Duplicate path-index episode_index={episode_index}",
                    episode_index=episode_index,
                    source_file=str(path),
                    line_number=line_number,
                )
                continue
            rows[episode_index] = row
    except Exception as exc:
        _add_error(errors, leaf=leaf, code="path_index_parse_error", message=f"{type(exc).__name__}: {exc}")
        return None
    return rows


def _validate_leaf(
    leaf: dict[str, str],
    *,
    filter_root: Path,
    path_index_root: Path,
    action_horizon: int,
    stride: int,
    errors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    leaf_path = leaf["path"]
    leaf_error_count_before = len(errors)
    cache_relative = _cache_relative_path(leaf_path)
    path_index_path = path_index_root / cache_relative / "path_index.jsonl"
    filter_leaf_dir = filter_root / leaf_path
    latest_path = filter_leaf_dir / "latest.json"
    if not latest_path.is_file():
        _add_error(errors, leaf=leaf_path, code="latest_missing", message=f"Missing {latest_path}")
        return None

    try:
        latest = _read_json(latest_path)
        run_id = str(latest["run_id"])
        aggregate_id = str(latest["aggregate_id"])
    except Exception as exc:
        _add_error(errors, leaf=leaf_path, code="latest_invalid", message=f"{type(exc).__name__}: {exc}")
        return None

    aggregate_dir = filter_leaf_dir / "runs" / run_id / "aggregates" / aggregate_id
    episodes_path = aggregate_dir / "episodes.jsonl"
    aggregate_manifest_path = aggregate_dir / "aggregate_manifest.json"
    if not episodes_path.is_file():
        _add_error(errors, leaf=leaf_path, code="episodes_missing", message=f"Missing {episodes_path}")
        return None
    if not aggregate_manifest_path.is_file():
        _add_error(
            errors, leaf=leaf_path, code="aggregate_manifest_missing", message=f"Missing {aggregate_manifest_path}"
        )
        return None

    try:
        aggregate_manifest = _read_json(aggregate_manifest_path)
    except Exception as exc:
        _add_error(errors, leaf=leaf_path, code="aggregate_manifest_invalid", message=f"{type(exc).__name__}: {exc}")
        return None
    if str(aggregate_manifest.get("run_id")) != run_id or str(aggregate_manifest.get("aggregate_id")) != aggregate_id:
        _add_error(
            errors,
            leaf=leaf_path,
            code="aggregate_identity_mismatch",
            message="latest.json identity does not match aggregate_manifest.json",
        )

    path_rows = _load_path_index(path_index_path, leaf_path, errors)
    if path_rows is None:
        return None

    sidecar_rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    stats = {
        "episodes": 0,
        "kept_episodes": 0,
        "filtered_episodes": 0,
        "source_frames": 0,
        "kept_frames": 0,
        "valid_windows": 0,
    }
    try:
        for line_number, row in _read_jsonl(episodes_path):
            try:
                if "episode_index" not in row:
                    raise KeyError("episode_index is missing")
                episode_index = int(row["episode_index"])
                if episode_index in seen:
                    raise ValueError(f"duplicate filter episode_index={episode_index}")
                seen.add(episode_index)
                indexed = path_rows.get(episode_index)
                if indexed is None:
                    raise ValueError(f"episode_index={episode_index} does not exist in path-index")

                source = row.get("source")
                if not isinstance(source, dict):
                    raise TypeError("source must be an object")
                frame_count = int(source["frame_count"])
                indexed_frame_count = int(indexed.get("frame_count", indexed.get("length")))
                if frame_count != indexed_frame_count:
                    raise ValueError(f"frame_count mismatch: filter={frame_count}, path_index={indexed_frame_count}")
                source_from = int(source["dataset_from_index"])
                source_to = int(source["dataset_to_index"])
                if source_to - source_from != frame_count:
                    raise ValueError(
                        f"source index range [{source_from}, {source_to}) does not match frame_count={frame_count}"
                    )
                indexed_from = int(indexed["data_from_index"])
                indexed_to = int(indexed["data_to_index"])
                if (source_from, source_to) != (indexed_from, indexed_to):
                    raise ValueError(
                        "dataset index mismatch: "
                        f"filter=[{source_from},{source_to}), path_index=[{indexed_from},{indexed_to})"
                    )
                if not _path_matches(source.get("data_path"), indexed.get("data_path")):
                    raise ValueError(
                        f"data_path mismatch: filter={source.get('data_path')!r}, "
                        f"path_index={indexed.get('data_path')!r}"
                    )

                status = str(row.get("status", ""))
                if status not in {"kept", "filtered"}:
                    raise ValueError(f"unsupported status={status!r}")
                clean_segments = _validated_segments(row.get("keep_segments"), frame_count)
                if status == "filtered" and clean_segments:
                    raise ValueError("status=filtered requires empty keep_segments")
                if status == "kept" and not clean_segments:
                    raise ValueError("status=kept requires non-empty keep_segments")

                kept_frames = sum(end - start for start, end in clean_segments)
                required_span = max(0, action_horizon - 1) * stride
                valid_windows = sum(max(0, end - start - required_span) for start, end in clean_segments)
                sidecar_rows.append(
                    {"episode_index": episode_index, "status": status, "clean_segments": clean_segments}
                )
                stats["episodes"] += 1
                stats[f"{status}_episodes"] += 1
                stats["source_frames"] += frame_count
                stats["kept_frames"] += kept_frames
                stats["valid_windows"] += valid_windows
            except Exception as exc:
                episode_index_value = row.get("episode_index")
                _add_error(
                    errors,
                    leaf=leaf_path,
                    code="episode_validation_error",
                    message=f"{type(exc).__name__}: {exc}",
                    episode_index=int(episode_index_value) if episode_index_value is not None else None,
                    source_file=str(episodes_path),
                    line_number=line_number,
                )
    except Exception as exc:
        _add_error(errors, leaf=leaf_path, code="episodes_parse_error", message=f"{type(exc).__name__}: {exc}")

    missing_filter_episodes = sorted(set(path_rows) - seen)
    for episode_index in missing_filter_episodes:
        _add_error(
            errors,
            leaf=leaf_path,
            code="filter_episode_missing",
            message=f"path-index episode_index={episode_index} is missing from filter results",
            episode_index=episode_index,
        )

    if len(errors) != leaf_error_count_before:
        return None
    sidecar_rows.sort(key=lambda row: int(row["episode_index"]))
    if len(sidecar_rows) != len(path_rows):
        _add_error(
            errors,
            leaf=leaf_path,
            code="episode_coverage_mismatch",
            message=f"validated filter episodes={len(sidecar_rows)}, path-index episodes={len(path_rows)}",
        )
        return None

    return {
        "leaf": leaf,
        "cache_relative": str(cache_relative),
        "path_index_path": str(path_index_path),
        "episodes_path": str(episodes_path),
        "sidecar_rows": sidecar_rows,
        "stats": stats,
        "source": {
            "run_id": run_id,
            "aggregate_id": aggregate_id,
            "dataset_fingerprint": aggregate_manifest.get("dataset_fingerprint"),
            "identity_hash": aggregate_manifest.get("identity_hash"),
            "output_schema_version": aggregate_manifest.get("output_schema_version"),
        },
    }


def _sum_stats(results: list[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "episodes",
        "kept_episodes",
        "filtered_episodes",
        "source_frames",
        "kept_frames",
        "valid_windows",
    )
    return {key: sum(int(result["stats"][key]) for result in results) for key in keys}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_task_selection_leaf(
    leaf: dict[str, str],
    *,
    dataset_root: Path,
    selection_root: Path,
    task_name: str,
    action_horizon: int,
    stride: int,
    errors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    import pyarrow.parquet as pq

    split = leaf["path"]
    selection_path = selection_root / f"{split}_episode_selection.json"
    split_root = dataset_root / split
    try:
        selection = _read_json(selection_path)
        info = _read_json(split_root / "meta/info.json")
        _require(selection.get("split") == split, f"{split}: selection split mismatch")
        _require(selection.get("task_name") == task_name, f"{split}: selection task mismatch")
        _require(str(info["codebase_version"]).startswith("v3"), f"{split}: expected LeRobot v3")
        _require(float(info["fps"]) == 30, f"{split}: expected 30 FPS")

        selected_list = selection["episode_indices"]
        _require(isinstance(selected_list, list), f"{split}: episode_indices must be a list")
        _require(
            all(isinstance(value, int) and not isinstance(value, bool) for value in selected_list),
            f"{split}: episode_indices must contain integers",
        )
        selected = set(selected_list)
        _require(len(selected) == len(selected_list), f"{split}: duplicate selected IDs")

        index_path = dataset_root / "sidecar" / split / "index.parquet"
        index_rows = pq.read_table(
            index_path, columns=["episode_index", "task_name"]
        ).to_pylist()
        task_by_id: dict[int, str] = {}
        for row in index_rows:
            episode_id = int(row["episode_index"])
            _require(episode_id not in task_by_id, f"{split}: duplicate sidecar ID {episode_id}")
            task_by_id[episode_id] = str(row["task_name"])

        expected_ids = {
            episode_id for episode_id, task in task_by_id.items() if task == task_name
        }
        _require(bool(expected_ids), f"{split}: task has no episodes")
        _require(selected == expected_ids, f"{split}: selection IDs do not match task sidecar")

        metadata_paths = sorted((split_root / "meta/episodes").rglob("*.parquet"))
        _require(bool(metadata_paths), f"{split}: missing episode metadata")
        lengths: dict[int, int] = {}
        for path in metadata_paths:
            for row in pq.read_table(path, columns=["episode_index", "length"]).to_pylist():
                episode_id, length = int(row["episode_index"]), int(row["length"])
                _require(episode_id not in lengths, f"{split}: duplicate metadata ID {episode_id}")
                _require(length >= 0, f"{split}: negative episode length")
                _require(episode_id not in selected or length > 0, f"{split}: empty target episode")
                lengths[episode_id] = length

        _require(set(lengths) == set(task_by_id), f"{split}: metadata/sidecar ID mismatch")
        _require(
            len(lengths) == int(info["total_episodes"]) == int(selection["source_total_episodes"]),
            f"{split}: source episode count mismatch",
        )
        _require(sum(lengths.values()) == int(info["total_frames"]), f"{split}: source frame count mismatch")
        _require(len(selected) == int(selection["selected_episodes"]), f"{split}: selected count mismatch")

        rows = []
        stats = {
            "episodes": len(lengths),
            "kept_episodes": len(selected),
            "filtered_episodes": len(lengths) - len(selected),
            "source_frames": sum(lengths.values()),
            "kept_frames": 0,
            "valid_windows": 0,
        }
        required_span = (action_horizon - 1) * stride
        for episode_id, length in sorted(lengths.items()):
            kept = episode_id in selected
            segments = _validated_segments([[0, length]] if kept else [], length)
            rows.append({
                "episode_index": episode_id,
                "status": "kept" if kept else "filtered",
                "clean_segments": segments,
            })
            stats["kept_frames"] += sum(end - start for start, end in segments)
            stats["valid_windows"] += sum(
                max(0, end - start - required_span) for start, end in segments
            )

        _require(
            stats["kept_frames"] == int(selection["selected_frames"]),
            f"{split}: selected frame count mismatch",
        )
        _require(stats["valid_windows"] > 0, f"{split}: no valid training windows")

        selection_sha256 = hashlib.sha256(selection_path.read_bytes()).hexdigest()
        fingerprint = hashlib.sha256(json.dumps(
            {"info": info, "lengths": sorted(lengths.items()), "tasks": sorted(task_by_id.items())},
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

        print(f"[{split}] {json.dumps(stats, sort_keys=True)}", flush=True)
        return {
            "leaf": leaf,
            "cache_relative": split,
            "path_index_path": None,
            "episodes_path": str(selection_path),
            "sidecar_rows": rows,
            "stats": stats,
            "source": {
                "type": "task_selection",
                "dataset_root": str(dataset_root),
                "split": split,
                "task_name": task_name,
                "selection_sha256": selection_sha256,
                "dataset_fingerprint": fingerprint,
                "identity_hash": hashlib.sha256((fingerprint + selection_sha256).encode()).hexdigest(),
            },
        }
    except Exception as exc:
        _add_error(
            errors, leaf=split, code="task_selection_validation_error",
            message=f"{type(exc).__name__}: {exc}", source_file=str(selection_path),
        )
        return None


def _print_summary(*, leaves: int, validated: int, errors: int, stats: dict[str, int]) -> None:
    print(f"leaves_total={leaves}")
    print(f"leaves_validated={validated}")
    print(f"validation_errors={errors}")
    for key, value in stats.items():
        print(f"{key}={value}")


def main() -> int:
    args = _parse_args()
    if args.input_mode == "task-selection":
        dataset_root = args.dataset_root.expanduser().resolve()
        filter_root = args.selection_root.expanduser().resolve()
        output_root = (
            args.output_root or REPO_ROOT / "data/petrel_cache/filter/abc_sim_bottles"
        ).expanduser().resolve()

        for destination in (output_root, output_root / "train", output_root / "val"):
            _require(
                not destination.resolve().is_relative_to(dataset_root),
                "Output must be outside the source dataset",
            )

        config_path = None
        path_index_root = None
        leaves = [
            {"name": f"abc_sim_{split}", "path": split, "group_file": str(filter_root)}
            for split in ("train", "val")
        ]
        config = {"sampling": {"action_horizon": args.action_horizon, "stride": args.stride}}
        leaf_schema = "openpi_task_selection_frame_filter_leaf_v1"
        collection_schema = "openpi_task_selection_frame_filter_collection_v1"
    else:
        config_path = args.config.expanduser().resolve()
        filter_root = args.filter_root.expanduser().resolve()
        path_index_root = args.path_index_root.expanduser().resolve()
        output_root = (args.output_root or DEFAULT_OUTPUT_ROOT).expanduser().resolve()
        leaves, config = _expand_leaves(config_path)
        leaf_schema = "openpi_abc130k_frame_filter_leaf_v1"
        collection_schema = "openpi_abc130k_frame_filter_collection_v1"
    
    errors: list[dict[str, Any]] = []
    if args.input_mode == "audit" and args.expected_leaves > 0 and len(leaves) != args.expected_leaves:
        _add_error(
            errors,
            leaf="<config>",
            code="leaf_count_mismatch",
            message=f"Expected {args.expected_leaves} leaves, config expanded to {len(leaves)}",
        )

    sampling = config.get("sampling") if isinstance(config.get("sampling"), dict) else {}
    action_horizon = int(sampling.get("action_horizon", config.get("openpi", {}).get("action_horizon", 1)))
    stride = int(sampling.get("stride", 1))
    if action_horizon <= 0 or stride <= 0:
        raise ValueError(f"Invalid sampling settings: action_horizon={action_horizon}, stride={stride}")

    results: list[dict[str, Any]] = []
    for index, leaf in enumerate(leaves, start=1):
        if args.input_mode == "task-selection":
            result = _validate_task_selection_leaf(
                leaf, dataset_root=dataset_root, selection_root=filter_root,
                task_name=args.task_name, action_horizon=action_horizon,
                stride=stride, errors=errors,
            )
        else:
            result = _validate_leaf(
                leaf,
                filter_root=filter_root,
                path_index_root=path_index_root,
                action_horizon=action_horizon,
                stride=stride,
                errors=errors,
            )
        
        if result is not None:
            results.append(result)
        if index % 50 == 0 or index == len(leaves):
            print(f"validated_progress={index}/{len(leaves)} errors={len(errors)}", flush=True)

    stats = _sum_stats(results)
    _print_summary(leaves=len(leaves), validated=len(results), errors=len(errors), stats=stats)
    generated_at = dt.datetime.now(dt.UTC).isoformat()

    if args.dry_run:
        if errors:
            for error in errors[:20]:
                print("ERROR " + json.dumps(error, ensure_ascii=False))
            if len(errors) > 20:
                print(f"ERROR ... {len(errors) - 20} additional validation errors")
        return 1 if errors else 0

    output_root.mkdir(parents=True, exist_ok=True)
    error_path = output_root / "validation_errors.jsonl"
    if errors:
        _atomic_write_jsonl(error_path, errors)
        _atomic_write_json(
            output_root / "frame_filter_manifest.json",
            {
                "schema": collection_schema,
                "status": "validation_failed",
                "published": False,
                "generated_at": generated_at,
                "config": str(config_path) if config_path is not None else None,
                "filter_root": str(filter_root),
                "path_index_root": str(path_index_root) if path_index_root is not None else None,
                "output_root": str(output_root),
                "leaves_total": len(leaves),
                "leaves_validated": len(results),
                "validation_error_count": len(errors),
                "validation_errors": str(error_path),
            },
        )
        print(f"Validation failed; wrote {error_path}. No sidecars were published.")
        return 1

    leaf_manifests: list[dict[str, Any]] = []
    for result in results:
        destination = output_root / result["cache_relative"]
        sidecar_path = destination / "frame_filter.jsonl"
        row_count, sidecar_sha256 = _atomic_write_jsonl(sidecar_path, result["sidecar_rows"])
        leaf_manifest = {
            "schema": leaf_schema,
            "generated_at": generated_at,
            "leaf_name": result["leaf"]["name"],
            "leaf_path": result["leaf"]["path"],
            "path_index_path": result["path_index_path"],
            "filter_episodes_path": result["episodes_path"],
            "sidecar_path": str(sidecar_path),
            "sidecar_rows": row_count,
            "sidecar_sha256": sidecar_sha256,
            "source": result["source"],
            "sampling": {"action_horizon": action_horizon, "stride": stride, "allow_end_padding": False},
            "stats": result["stats"],
        }
        _atomic_write_json(destination / "frame_filter_manifest.json", leaf_manifest)
        leaf_manifests.append(
            {
                "leaf_path": result["leaf"]["path"],
                "cache_relative": result["cache_relative"],
                "sidecar_sha256": sidecar_sha256,
                "source": result["source"],
                "stats": result["stats"],
            }
        )

    collection_identity = hashlib.sha256(
        json.dumps(leaf_manifests, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _atomic_write_jsonl(error_path, [])
    _atomic_write_json(
        output_root / "frame_filter_manifest.json",
        {
            "schema": collection_schema,
            "status": "ready",
            "published": True,
            "generated_at": generated_at,
            "config": str(config_path) if config_path is not None else None,
            "filter_root": str(filter_root),
            "path_index_root": str(path_index_root) if path_index_root is not None else None,
            "output_root": str(output_root),
            "collection_identity_sha256": collection_identity,
            "sampling": {"action_horizon": action_horizon, "stride": stride, "allow_end_padding": False},
            "leaves_total": len(leaves),
            "validation_error_count": 0,
            "stats": stats,
            "leaves": leaf_manifests,
        },
    )
    print(f"Published {len(results)} sidecars under {output_root}")
    print(f"collection_identity_sha256={collection_identity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
