#!/usr/bin/env python3
"""Summarize worker-side DataLoader profiling metrics from a training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _metric_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("data_metrics.jsonl"))


def _merge_event(target: dict[str, float | int], source: dict[str, Any]) -> None:
    target["count"] = int(target["count"]) + int(source.get("count", 0))
    target["total_ms"] = float(target["total_ms"]) + float(source.get("total_ms", 0.0))
    target["max_ms"] = max(float(target["max_ms"]), float(source.get("max_ms", 0.0)))
    target["total_bytes"] = int(target["total_bytes"]) + int(source.get("total_bytes", 0))


def summarize(root: Path) -> tuple[dict[str, dict[str, float | int]], int, int]:
    files = _metric_files(root)
    if not files:
        raise FileNotFoundError(f"No data_metrics.jsonl files found under {root}")

    events: dict[str, dict[str, float | int]] = {}
    records = 0
    for path in files:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
                records += 1
                for name, value in payload.get("events", {}).items():
                    aggregate = events.setdefault(
                        name,
                        {"count": 0, "total_ms": 0.0, "max_ms": 0.0, "total_bytes": 0},
                    )
                    _merge_event(aggregate, value)
    return events, len(files), records


def _print_summary(events: dict[str, dict[str, float | int]], *, files: int, records: int, limit: int) -> None:
    print(f"DataLoader profiling: files={files} metric_windows={records} events={len(events)}")
    print(
        f"{'event':42} {'total_s':>11} {'count':>10} {'avg_ms':>11} "
        f"{'max_ms':>11} {'GiB':>9} {'MB/s':>10}"
    )
    print("-" * 112)
    ordered = sorted(events.items(), key=lambda item: float(item[1]["total_ms"]), reverse=True)
    for name, value in ordered[:limit]:
        count = int(value["count"])
        total_ms = float(value["total_ms"])
        total_bytes = int(value["total_bytes"])
        average_ms = total_ms / max(count, 1)
        throughput = total_bytes / 1e6 / max(total_ms / 1000.0, 1e-12) if total_bytes else 0.0
        print(
            f"{name[:42]:42} {total_ms / 1000.0:11.3f} {count:10d} {average_ms:11.3f} "
            f"{float(value['max_ms']):11.3f} {total_bytes / 2**30:9.3f} {throughput:10.1f}"
        )

    hits = int(events.get("parquet.cache_hit", {}).get("count", 0))
    misses = int(events.get("parquet.cache_miss", {}).get("count", 0))
    if hits + misses:
        print(f"\nParquet LRU cache: hits={hits} misses={misses} hit_rate={hits / (hits + misses):.2%}")
    print(
        "\nNote: total_s is summed worker time. Workers overlap, and parent stages such as "
        "petrel.batch.total include child stages, so these rows must not be added together."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Run directory, profiling directory, or one data_metrics.jsonl")
    parser.add_argument("--limit", type=int, default=30, help="Maximum number of events to display")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    events, files, records = summarize(args.path)
    _print_summary(events, files=files, records=records, limit=args.limit)


if __name__ == "__main__":
    main()
