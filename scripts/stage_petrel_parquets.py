#!/usr/bin/env python3
"""Stage unique Petrel parquet files from path indices into node-local storage."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import json
import os
from pathlib import Path
import shutil
import threading
import time

from openpi.training.petrel_lerobot_reader import local_parquet_cache_path


def collect_parquet_paths(path_index_root: Path) -> list[Path]:
    paths: set[Path] = set()
    for index_path in path_index_root.rglob("path_index.jsonl"):
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    paths.add(Path(json.loads(line)["data_path"]).expanduser())
    return sorted(paths, key=str)


def stage_one(source_path: Path, cache_dir: Path) -> tuple[int, bool]:
    destination = local_parquet_cache_path(cache_dir, source_path)
    source_size = source_path.stat().st_size
    try:
        if destination.is_file() and destination.stat().st_size == source_size:
            return source_size, False
    except OSError:
        pass

    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        with source_path.open("rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        if temporary.stat().st_size != source_size:
            raise OSError(
                f"Staged parquet size mismatch for {source_path}: "
                f"expected={source_size}, actual={temporary.stat().st_size}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return source_size, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-index-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if not args.path_index_root.is_dir():
        raise FileNotFoundError(f"Path-index root does not exist: {args.path_index_root}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    paths = collect_parquet_paths(args.path_index_root)
    if not paths:
        raise RuntimeError(f"No parquet paths found below {args.path_index_root}")
    disk = shutil.disk_usage(args.cache_dir)
    print(
        f"[parquet-stage] start files={len(paths)} workers={args.workers} "
        f"cache_dir={args.cache_dir} free_gib={disk.free / 2**30:.1f}",
        flush=True,
    )
    started = time.perf_counter()
    total_bytes = 0
    copied_bytes = 0
    copied_files = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(stage_one, path, args.cache_dir): path for path in paths}
        for future in as_completed(futures):
            size, copied = future.result()
            total_bytes += size
            copied_bytes += size if copied else 0
            copied_files += int(copied)
            completed += 1
            if completed % 100 == 0 or completed == len(paths):
                elapsed = time.perf_counter() - started
                print(
                    f"[parquet-stage] progress={completed}/{len(paths)} "
                    f"copied_files={copied_files} total_gib={total_bytes / 2**30:.1f} "
                    f"copied_gib={copied_bytes / 2**30:.1f} "
                    f"elapsed_s={elapsed:.1f}",
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    print(
        f"[parquet-stage] complete files={len(paths)} copied_files={copied_files} "
        f"total_gib={total_bytes / 2**30:.1f} "
        f"copied_gib={copied_bytes / 2**30:.1f} elapsed_s={elapsed:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
