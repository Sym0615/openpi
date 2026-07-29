"""Low-overhead profiling helpers shared by data loading and PyTorch training."""

from __future__ import annotations

import csv
import dataclasses
import logging
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any


@dataclasses.dataclass
class _Counter:
    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0
    total_bytes: int = 0


_LOCK = threading.Lock()
_COUNTERS: dict[str, _Counter] = {}
_ENABLED = os.environ.get("OPENPI_PROFILE", "0") == "1"


def set_enabled(*, enabled: bool) -> None:
    global _ENABLED  # noqa: PLW0603
    _ENABLED = enabled
    os.environ["OPENPI_PROFILE"] = "1" if enabled else "0"


def enabled() -> bool:
    return _ENABLED


def record(name: str, elapsed_s: float, *, num_bytes: int | None = None, **_: Any) -> None:
    """Aggregate an event without retaining per-sample details."""
    if not _ENABLED:
        return
    with _LOCK:
        counter = _COUNTERS.setdefault(name, _Counter())
        counter.count += 1
        counter.total_s += float(elapsed_s)
        counter.max_s = max(counter.max_s, float(elapsed_s))
        if num_bytes is not None:
            counter.total_bytes += int(num_bytes)


def snapshot(*, reset: bool = False) -> dict[str, dict[str, float | int]]:
    with _LOCK:
        result = {
            name: {
                "count": value.count,
                "total_ms": value.total_s * 1000.0,
                "average_ms": value.total_s * 1000.0 / max(1, value.count),
                "max_ms": value.max_s * 1000.0,
                "total_bytes": value.total_bytes,
                "throughput_mb_s": value.total_bytes / 1e6 / value.total_s if value.total_s > 0 else 0.0,
            }
            for name, value in _COUNTERS.items()
        }
        if reset:
            _COUNTERS.clear()
    return result


class GpuMonitor:
    """Continuously sample one GPU with nvidia-smi and write samples to CSV."""

    _FIELDS = (
        "timestamp",
        "gpu_index",
        "gpu_util_percent",
        "memory_util_percent",
        "memory_used_mib",
        "memory_total_mib",
        "power_w",
        "temperature_c",
    )

    def __init__(self, output_path: Path, device_index: str | int, interval_s: float):
        self._output_path = output_path
        self._device_index = device_index
        self._interval_s = max(0.2, interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, float]] = []

    def start(self) -> bool:
        if shutil.which("nvidia-smi") is None:
            logging.warning("GPU utilization sampling disabled: nvidia-smi was not found")
            return False
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="openpi-gpu-monitor", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self._interval_s * 2))

    def summary(self, *, reset: bool = False) -> dict[str, float]:
        with _LOCK:
            samples = list(self._samples)
            if reset:
                self._samples.clear()
        if not samples:
            return {}
        return {
            "gpu_util_avg": sum(x["gpu_util_percent"] for x in samples) / len(samples),
            "gpu_util_max": max(x["gpu_util_percent"] for x in samples),
            "memory_util_avg": sum(x["memory_util_percent"] for x in samples) / len(samples),
            "memory_used_mib_max": max(x["memory_used_mib"] for x in samples),
            "power_w_avg": sum(x["power_w"] for x in samples) / len(samples),
            "temperature_c_max": max(x["temperature_c"] for x in samples),
        }

    def _run(self) -> None:
        query = "index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu"
        write_header = not self._output_path.exists() or self._output_path.stat().st_size == 0
        with self._output_path.open("a", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=self._FIELDS)
            if write_header:
                writer.writeheader()
            while not self._stop.is_set():
                started = time.monotonic()
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            f"--query-gpu={query}",
                            "--format=csv,noheader,nounits",
                            "-i",
                            str(self._device_index),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=max(2.0, self._interval_s),
                    )
                    values = [value.strip() for value in result.stdout.strip().split(",")]
                    if len(values) == 7:
                        numeric = [float(value) for value in values]
                        sample = dict(zip(self._FIELDS[1:], numeric, strict=True))
                        row = {"timestamp": time.time(), **sample}
                        writer.writerow(row)
                        output.flush()
                        with _LOCK:
                            self._samples.append(sample)
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    logging.warning("Failed to sample GPU metrics: %s", exc)
                self._stop.wait(max(0.0, self._interval_s - (time.monotonic() - started)))
