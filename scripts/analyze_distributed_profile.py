"""Summarize per-rank OpenPI DDP profiling output and flag GPU/rank imbalance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_last_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last_line = ""
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                last_line = line
    return json.loads(last_line) if last_line else None


def format_number(value: Any, digits: int = 1) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_dir", type=Path, help="Path to <checkpoint_dir>/profiling")
    parser.add_argument("--min-gpu-util", type=float, default=70.0)
    parser.add_argument("--max-step-imbalance", type=float, default=1.20)
    args = parser.parse_args()

    global_metrics = read_last_json(args.profile_dir / "step_metrics.jsonl")
    rank_rows = []
    for path in sorted((args.profile_dir / "ranks").glob("node_*_rank_*/step_metrics.jsonl")):
        payload = read_last_json(path)
        if payload is not None:
            rank_rows.append(payload)

    if not rank_rows:
        raise SystemExit(f"No per-rank step metrics found under {args.profile_dir / 'ranks'}")

    print("rank node local gpu_util% step_ms data_wait_ms samples/s")
    for row in sorted(rank_rows, key=lambda item: int(item["rank"])):
        print(
            f"{int(row['rank']):>4} {int(row['node_rank']):>4} {int(row['local_rank']):>5} "
            f"{format_number(row.get('gpu_util_avg')):>9} {format_number(row.get('step_ms')):>7} "
            f"{format_number(row.get('data_wait_ms')):>12} {format_number(row.get('samples_per_second'), 2):>9}"
        )

    gpu_utils = [float(row["gpu_util_avg"]) for row in rank_rows if "gpu_util_avg" in row]
    step_times = [float(row["step_ms"]) for row in rank_rows if float(row.get("step_ms", 0.0)) > 0]
    throughputs = [float(row["samples_per_second"]) for row in rank_rows if "samples_per_second" in row]

    print()
    if global_metrics:
        print(
            "global: "
            f"step={global_metrics.get('step')} "
            f"world_size={global_metrics.get('world_size')} "
            f"samples/s={format_number(global_metrics.get('samples_per_second_effective'), 2)} "
            f"slowest_step_ms={format_number(global_metrics.get('step_ms_max'))} "
            f"gpu_util_avg={format_number(global_metrics.get('gpu_util_avg'))}%"
        )

    warnings = []
    if gpu_utils and min(gpu_utils) < args.min_gpu_util:
        warnings.append(
            f"low GPU utilization: minimum rank average is {min(gpu_utils):.1f}% "
            f"(< {args.min_gpu_util:.1f}%)"
        )
    if step_times:
        imbalance = max(step_times) / min(step_times)
        if imbalance > args.max_step_imbalance:
            warnings.append(
                f"step-time imbalance: slowest/fastest={imbalance:.2f} "
                f"(> {args.max_step_imbalance:.2f})"
            )
    if throughputs and min(throughputs) > 0:
        throughput_imbalance = max(throughputs) / min(throughputs)
        if throughput_imbalance > args.max_step_imbalance:
            warnings.append(f"per-rank throughput imbalance: max/min={throughput_imbalance:.2f}")

    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("No utilization or rank-imbalance warning was triggered by the configured thresholds.")


if __name__ == "__main__":
    main()
