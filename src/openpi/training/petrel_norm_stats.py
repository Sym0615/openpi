"""Fast episode-vectorized normalization statistics for OpenPI Petrel datasets."""

from __future__ import annotations

from collections import defaultdict
import dataclasses
import heapq
import math
import multiprocessing as mp
import time
from typing import Any

import numpy as np
import torch

from openpi.shared import normalize
from openpi.training import petrel_dataset
from openpi.training import petrel_lerobot_reader as reader


@dataclasses.dataclass
class _Accumulator:
    dim: int
    reservoir_rows: int
    seed: int

    def __post_init__(self) -> None:
        self.count = 0
        self.sum = np.zeros(self.dim, dtype=np.float64)
        self.sumsq = np.zeros(self.dim, dtype=np.float64)
        self.reservoir = np.empty((max(0, self.reservoir_rows), self.dim), dtype=np.float32)
        self.reservoir_count = 0
        self.rows_seen = 0
        self.rng = np.random.default_rng(self.seed)

    def update(self, value: torch.Tensor | np.ndarray) -> None:
        array = np.asarray(value, dtype=np.float32).reshape(-1, self.dim)
        if not bool(np.isfinite(array).all()):
            raise ValueError("Petrel norm stats encountered non-finite projected values.")
        rows = int(array.shape[0])
        if rows == 0:
            return
        array64 = array.astype(np.float64, copy=False)
        self.count += rows
        self.sum += array64.sum(axis=0)
        self.sumsq += np.square(array64).sum(axis=0)
        self._update_reservoir(array)

    def _update_reservoir(self, rows: np.ndarray) -> None:
        capacity = int(self.reservoir.shape[0])
        if capacity <= 0:
            self.rows_seen += int(rows.shape[0])
            return
        for row in rows:
            self.rows_seen += 1
            if self.reservoir_count < capacity:
                self.reservoir[self.reservoir_count] = row
                self.reservoir_count += 1
                continue
            index = int(self.rng.integers(0, self.rows_seen))
            if index < capacity:
                self.reservoir[index] = row

    def merge(self, other: _Accumulator) -> None:
        self.count += int(other.count)
        self.sum += other.sum
        self.sumsq += other.sumsq
        self._update_reservoir(other.reservoir[: other.reservoir_count])

    def finalize(self) -> normalize.NormStats:
        if self.count < 2:
            raise ValueError(f"Cannot compute Petrel norm stats from only {self.count} rows.")
        mean = self.sum / float(self.count)
        variance = self.sumsq / float(self.count) - np.square(mean)
        std = np.sqrt(np.maximum(variance, 0.0))
        sample = self.reservoir[: self.reservoir_count].astype(np.float64, copy=False)
        if sample.shape[0] < 2:
            raise ValueError("Petrel norm reservoir contains fewer than two rows.")
        q01, q99 = np.quantile(sample, [0.01, 0.99], axis=0)
        return normalize.NormStats(mean=mean, std=std, q01=q01, q99=q99)


@dataclasses.dataclass
class _WorkerStats:
    state: _Accumulator
    actions: _Accumulator
    windows: int = 0
    skipped_windows: int = 0


def _make_dataset(spec: Any, raw_config: dict[str, Any], action_horizon: int) -> Any:
    storage = raw_config["storage"]
    sampling = raw_config["sampling"]
    return reader.PetrelLeRobotDataset(
        spec,
        num_frames=int(sampling["num_frames"]),
        action_size=int(action_horizon),
        global_sample_stride=int(sampling["global_sample_stride"]),
        parquet_cache_size=int(storage.get("parquet_cache_size", 8)),
        use_path_index_cache=True,
    )


def _select_episode_blocks(
    units: list[dict[str, Any]],
    *,
    target: int,
    block_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    blocks: list[tuple[int, int, int]] = []
    for unit_index, unit in enumerate(units):
        windows = int(unit["windows"])
        blocks.extend(
            (unit_index, offset, min(block_size, windows - offset)) for offset in range(0, windows, block_size)
        )

    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(blocks))
    selected_by_unit: dict[int, dict[str, Any]] = {}
    remaining = int(target)
    for block_index in order:
        unit_index, offset, length = blocks[int(block_index)]
        take = min(int(length), remaining)
        if take < length:
            offset += int(rng.integers(0, length - take + 1))
        selected = selected_by_unit.setdefault(
            unit_index,
            {**units[unit_index], "quota": 0, "blocks": []},
        )
        selected["blocks"].append((int(offset), int(take)))
        selected["quota"] += int(take)
        remaining -= int(take)
        if remaining <= 0:
            break
    if remaining != 0:
        raise RuntimeError(f"Episode block selection left {remaining} windows unallocated.")
    return list(selected_by_unit.values())


def build_episode_plan(
    raw_config: dict[str, Any],
    specs: list[Any],
    *,
    action_horizon: int,
    max_frames: int | None,
    sample_block_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int, int]:
    allow_padding = bool(raw_config["mixture"].get("allow_padding_at_end", False))
    units: list[dict[str, Any]] = []
    started = time.perf_counter()
    for spec_index, spec in enumerate(specs):
        dataset = _make_dataset(spec, raw_config, action_horizon)
        counts = dataset.valid_start_counts(allow_padding)
        for trajectory_pos, count in enumerate(counts):
            windows = int(count)
            if windows <= 0:
                continue
            units.append(
                {
                    "spec_index": int(spec_index),
                    "trajectory_pos": int(trajectory_pos),
                    "episode_index": int(dataset.trajectory_ids[int(trajectory_pos)]),
                    "windows": windows,
                    "quota": windows,
                }
            )
        if (spec_index + 1) % 100 == 0 or spec_index + 1 == len(specs):
            print(
                f"[petrel-fast-plan] datasets={spec_index + 1}/{len(specs)} "
                f"episodes={len(units)} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    total = int(sum(int(unit["windows"]) for unit in units))
    if total <= 0:
        raise ValueError("Petrel dataset has no valid normalization windows.")
    target = total if max_frames is None else min(int(max_frames), total)
    if target <= 0:
        raise ValueError("max_frames must select at least one Petrel window.")
    if target == total:
        selected = [{**unit, "quota": int(unit["windows"]), "blocks": None} for unit in units]
    else:
        selected = _select_episode_blocks(
            units,
            target=target,
            block_size=max(1, int(sample_block_size)),
            seed=int(seed),
        )
    print(
        f"[petrel-fast-plan] total_windows={total} selected_windows={target} selected_episodes={len(selected)}",
        flush=True,
    )
    return selected, total, target


def _split_plan(units: list[dict[str, Any]], num_workers: int) -> list[list[dict[str, Any]]]:
    heaps: list[tuple[int, int, list[dict[str, Any]]]] = [(0, index, []) for index in range(num_workers)]
    heapq.heapify(heaps)
    for unit in sorted(units, key=lambda item: int(item["quota"]), reverse=True):
        load, worker_index, worker_units = heapq.heappop(heaps)
        worker_units.append(unit)
        heapq.heappush(heaps, (load + int(unit["quota"]), worker_index, worker_units))
    return [units for _, _, units in sorted(heaps, key=lambda item: item[1]) if units]


def _valid_start_indices(dataset: Any, unit: dict[str, Any], *, allow_padding: bool) -> np.ndarray:
    trajectory_pos = int(unit["trajectory_pos"])
    episode_index = int(unit["episode_index"])
    windows = int(unit["windows"])
    if dataset.subtask_action_segments:
        episode_len = int(dataset.trajectory_lengths[trajectory_pos])
        intervals = dataset._valid_start_intervals_for_episode(  # noqa: SLF001
            episode_index, episode_len, allow_padding
        )
        starts = np.fromiter(
            (start for begin, end in intervals for start in range(int(begin), int(end) + 1)),
            dtype=np.int64,
            count=windows,
        )
    else:
        starts = np.arange(windows, dtype=np.int64)
    if int(starts.size) != windows:
        raise RuntimeError(
            f"Dataset {dataset.name} episode {episode_index} exposes {starts.size} starts, expected {windows}."
        )
    blocks = unit.get("blocks")
    if blocks is None:
        return starts
    selected = [starts[int(offset) : int(offset) + int(length)] for offset, length in blocks]
    return np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)


def _worker(payload: dict[str, Any]) -> _WorkerStats:
    torch.set_num_threads(1)
    worker_id = int(payload["worker_id"])
    output_dim = int(payload["output_dim"])
    reservoir_rows = int(payload["reservoir_rows"])
    stats = _WorkerStats(
        state=_Accumulator(output_dim, reservoir_rows, int(payload["seed"]) + worker_id * 1009 + 11),
        actions=_Accumulator(output_dim, reservoir_rows, int(payload["seed"]) + worker_id * 1009 + 23),
    )
    raw_config = payload["raw_config"]
    specs = payload["specs"]
    action_horizon = int(payload["action_horizon"])
    episode_batch_size = int(payload["episode_batch_size"])
    allow_padding = bool(raw_config["mixture"].get("allow_padding_at_end", False))
    by_spec: dict[int, list[dict[str, int]]] = defaultdict(list)
    for unit in payload["units"]:
        by_spec[int(unit["spec_index"])].append(unit)

    started = time.perf_counter()
    for spec_index, units in sorted(by_spec.items()):
        dataset = _make_dataset(specs[spec_index], raw_config, action_horizon)
        for unit in units:
            try:
                starts = _valid_start_indices(dataset, unit, allow_padding=allow_padding)
                max_start = dataset.max_start_for_trajectory_pos(int(unit["trajectory_pos"]), allow_padding)
                for item in dataset.iter_episode_stat_batches(
                    int(unit["episode_index"]),
                    allow_padding_at_end=allow_padding,
                    max_windows_per_batch=episode_batch_size,
                    max_start=max_start,
                    start_indices=starts,
                ):
                    state, actions = petrel_dataset.project_petrel_features(item, output_dim)
                    stats.state.update(state[:, 0, :])
                    stats.actions.update(actions)
                    stats.windows += int(item["_stats_num_samples"])
            except Exception as exc:
                stats.skipped_windows += int(unit["quota"])
                print(
                    f"[petrel-fast-worker {worker_id}] skip dataset={dataset.name} "
                    f"episode={unit['episode_index']}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        elapsed = max(time.perf_counter() - started, 1e-9)
        print(
            f"[petrel-fast-worker {worker_id}] datasets_done={len(by_spec)} "
            f"windows={stats.windows} speed={stats.windows / elapsed:.1f}/s",
            flush=True,
        )
    return stats


def compute_fast_petrel_norm_stats(
    dataset_config: str,
    *,
    action_horizon: int,
    output_dim: int,
    max_frames: int | None,
    num_workers: int,
    episode_batch_size: int,
    reservoir_rows: int,
    seed: int = 42,
    conf_path: str | None = None,
) -> dict[str, normalize.NormStats]:
    raw_config, specs = reader.load_dataset_specs(dataset_config, conf_path)
    units, _, selected_windows = build_episode_plan(
        raw_config,
        specs,
        action_horizon=action_horizon,
        max_frames=max_frames,
        sample_block_size=episode_batch_size,
        seed=seed,
    )
    worker_count = min(max(1, int(num_workers)), len(units))
    shards = _split_plan(units, worker_count)
    per_worker_reservoir = max(1024, math.ceil(int(reservoir_rows) * 2 / len(shards)))
    payloads = [
        {
            "worker_id": worker_id,
            "raw_config": raw_config,
            "specs": specs,
            "units": shard,
            "action_horizon": int(action_horizon),
            "output_dim": int(output_dim),
            "episode_batch_size": int(episode_batch_size),
            "reservoir_rows": per_worker_reservoir,
            "seed": int(seed),
        }
        for worker_id, shard in enumerate(shards)
    ]
    merged = _WorkerStats(
        state=_Accumulator(output_dim, int(reservoir_rows), int(seed) + 101),
        actions=_Accumulator(output_dim, int(reservoir_rows), int(seed) + 211),
    )
    started = time.perf_counter()
    if len(payloads) == 1:
        results = [_worker(payloads[0])]
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=len(payloads)) as pool:
            results = list(pool.imap_unordered(_worker, payloads))
    for partial in results:
        merged.state.merge(partial.state)
        merged.actions.merge(partial.actions)
        merged.windows += int(partial.windows)
        merged.skipped_windows += int(partial.skipped_windows)
    print(
        f"[petrel-fast] processed={merged.windows}/{selected_windows} "
        f"skipped={merged.skipped_windows} elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    if merged.windows <= 0:
        raise RuntimeError("Fast Petrel norm stats processed no windows.")
    return {
        "state": merged.state.finalize(),
        "actions": merged.actions.finalize(),
    }
