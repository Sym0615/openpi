#!/usr/bin/env python3
"""Run ABC130K trim/quality filters on the full ABC-Sim train split.

Reuse the original data_filter_pipeline implementation without importing it in
training workers. Only compact, validated sidecars are required on GPU nodes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import datetime as dt
import hashlib
import itertools
import json
import multiprocessing
import os
from pathlib import Path
import sys

from build_abc130k_frame_filter_sidecars import _atomic_write_json
from build_abc130k_frame_filter_sidecars import _atomic_write_jsonl
from build_abc130k_frame_filter_sidecars import _validated_segments
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime(config: dict, source: str):
    sys.path.insert(0, source)
    from dfp.filters.motionless import MotionlessTrimFilter
    from dfp.filters.sudden_change import SuddenChangeFilter
    from dfp.filters.trend_alignment import TrendAlignmentFilter
    from dfp.lerobot.signal_config import build_signal_specs
    from dfp.lerobot.signals import derive_layout
    from dfp.runner.pipeline import RuntimeGroup

    classes = {cls.__name__: cls for cls in (MotionlessTrimFilter, SuddenChangeFilter, TrendAlignmentFilter)}
    specs = build_signal_specs(config["signals"])
    layouts = {name: derive_layout(spec, spec.dim_names) for name, spec in specs.items()}
    groups = []
    for group in config["pipeline"]:
        filters = {}
        for name, stage in group["stages"].items():
            if stage["type"] not in classes:
                raise ValueError(f"Unapproved stage: {stage['type']}")
            filters[name] = classes[stage["type"]](name, stage["config"])
        groups.append(RuntimeGroup(group["group"], filters))
    return specs, layouts, groups


def _file_job(data_path: str, rows: list[dict], config: dict, source: str, shard: str) -> dict:
    specs, layouts, groups = _runtime(config, source)
    from dfp.lerobot.reader import _build_episode
    from dfp.runner.pipeline import evaluate_episode

    pa.set_cpu_count(1)
    columns = sorted({spec.source_key for spec in specs.values()} | {"episode_index", "frame_index", "index"})
    table = pq.read_table(data_path, columns=columns, use_threads=False)
    values = {}
    for name, spec in specs.items():
        col = table[spec.source_key].combine_chunks()
        if pa.types.is_list(col.type):
            array = np.asarray(col.values).reshape(len(col), len(spec.dim_names))
        else:
            array = np.asarray(col).reshape(len(col), 1)
        values[name] = array
    episode_ids = np.asarray(table["episode_index"])
    frame_ids = np.asarray(table["frame_index"])
    global_ids = np.asarray(table["index"])
    boundaries = np.r_[0, np.flatnonzero(np.diff(episode_ids)) + 1, len(episode_ids)]
    spans = {int(episode_ids[start]): (int(start), int(end)) for start, end in itertools.pairwise(boundaries)}
    if len(spans) != len(boundaries) - 1 or set(spans) != {int(row["episode_index"]) for row in rows}:
        raise ValueError(f"Parquet episode coverage mismatch: {data_path}")
    output = []
    counts: Counter = Counter()
    for row in sorted(rows, key=lambda r: int(r["episode_index"])):
        index = int(row["episode_index"])
        start, end = spans[index]
        length = int(row["frame_count"])
        if end - start != length or not np.array_equal(frame_ids[start:end], np.arange(length)):
            raise ValueError(f"Frame identity mismatch for episode {index}")
        if not np.array_equal(global_ids[start:end], np.arange(row["data_from_index"], row["data_to_index"])):
            raise ValueError(f"Global index mismatch for episode {index}")
        episode = _build_episode(
            index=index,
            raw_by_signal={name: data[start:end] for name, data in values.items()},
            specs=specs,
            layouts=layouts,
            fps=30.0,
        )
        result = evaluate_episode(episode, groups, specs, layouts, config["min_slice_frames"])
        if result.status == "failed":
            raise RuntimeError(f"Filter failed for episode {index}: {result.error}")
        segments = _validated_segments(result.keep_segments, length)
        counts[f"status_{result.status}"] += 1
        for name, stage in result.stages.items():
            counts[f"{name}_{stage['status']}"] += 1
        output.append(
            {"episode_index": index, "status": "kept" if segments else "filtered", "clean_segments": segments}
        )
    _atomic_write_jsonl(Path(shard), output)
    return {"shard": shard, "counts": dict(counts), "episodes": len(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/petrel/frame_filters/abc_sim.yaml")
    parser.add_argument(
        "--dfp-source", type=Path, default=Path(os.environ.get("OPENPI_DFP_SOURCE", "/sunyiming/data_filter_pipeline"))
    )
    parser.add_argument(
        "--path-index", type=Path, default=ROOT / "data/petrel_cache/path_index/abc_sim/train/path_index.jsonl"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/petrel_cache/filter/abc_sim/train")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    config = yaml.safe_load(args.config.read_text())
    if config["dataset_id"] != "abc_sim" or config["leaf_path"] != "train":
        raise ValueError("Only the full ABC-Sim train split is supported")
    _runtime(config, str(args.dfp_source))
    source_hashes = {
        str(path.relative_to(args.dfp_source)): _sha(path) for path in sorted((args.dfp_source / "dfp").rglob("*.py"))
    }
    by_file: dict[str, list[dict]] = {}
    expected = {}
    for line in args.path_index.read_text().splitlines():
        row = json.loads(line)
        index = int(row["episode_index"])
        if index in expected:
            raise ValueError(f"Duplicate indexed episode: {index}")
        expected[index] = row
        by_file.setdefault(row["data_path"], []).append(row)
    if not expected:
        raise ValueError("Empty path index")
    index_sha256 = _sha(args.path_index)
    started = dt.datetime.now(dt.UTC).isoformat()
    staging = args.output_root / "builds" / dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    staging.mkdir(parents=True, exist_ok=False)
    print(
        f"Filtering episodes={len(expected)} files={len(by_file)} workers={args.workers} stages=motionless,sudden_change,trend_alignment",
        flush=True,
    )
    shards = []
    counts: Counter = Counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = [
            executor.submit(_file_job, path, rows, config, str(args.dfp_source), str(staging / f"file-{i:03d}.jsonl"))
            for i, (path, rows) in enumerate(sorted(by_file.items()))
        ]
        for future in as_completed(futures):
            result = future.result()
            shards.append(result["shard"])
            counts.update(result["counts"])
            print(
                f"Completed files={len(shards)}/{len(futures)} episodes={sum(counts[k] for k in ('status_kept', 'status_filtered'))}",
                flush=True,
            )
    rows = [json.loads(line) for shard in shards for line in Path(shard).read_text().splitlines()]
    rows.sort(key=lambda row: row["episode_index"])
    if [row["episode_index"] for row in rows] != sorted(expected):
        raise ValueError("Final episode coverage mismatch")
    stats = {
        "episodes": len(rows),
        "kept_episodes": sum(row["status"] == "kept" for row in rows),
        "filtered_episodes": sum(row["status"] == "filtered" for row in rows),
        "source_frames": sum(int(row["frame_count"]) for row in expected.values()),
        "kept_frames": sum(end - start for row in rows for start, end in row["clean_segments"]),
        "valid_windows": sum(max(0, end - start - 29) for row in rows for start, end in row["clean_segments"]),
        "valid_episodes": sum(any(end - start >= 30 for start, end in row["clean_segments"]) for row in rows),
        "per_stage": dict(counts),
    }
    # Confirm code/config/index stayed fixed throughout the scan before publishing.
    if yaml.safe_load(args.config.read_text()) != config:
        raise ValueError("Filter config changed during build")
    if _sha(args.path_index) != index_sha256:
        raise ValueError("Path index changed during build")
    if any(_sha(args.dfp_source / path) != digest for path, digest in source_hashes.items()):
        raise ValueError("DFP implementation changed during build")
    n, digest = _atomic_write_jsonl(args.output_root / "frame_filter.jsonl", rows)
    manifest = {
        "schema": "openpi_abcsim_frame_filter_leaf_v1",
        "status": "ready",
        "published": True,
        "started_at": started,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "dataset_id": "abc_sim",
        "leaf_path": "train",
        "sidecar_rows": n,
        "sidecar_sha256": digest,
        "sampling": {"action_horizon": 30, "stride": 1, "allow_end_padding": False},
        "filter_config": config,
        "filter_config_sha256": _sha(args.config),
        "path_index_sha256": index_sha256,
        "dfp_source_sha256": source_hashes,
        "stats": stats,
    }
    _atomic_write_json(args.output_root / "frame_filter_manifest.json", manifest)
    print(json.dumps(stats, indent=2), flush=True)
    print(f"Published {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
