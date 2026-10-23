"""Progress management for distributed evaluation with filesystem-based synchronization."""

from __future__ import annotations

import atexit
import json
from filelock import SoftFileLock, Timeout
import os
import socket
import time
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Lock timeout in seconds (5 minutes)
LOCK_TIMEOUT_SECONDS = 300
# Heartbeat interval in seconds
HEARTBEAT_INTERVAL_SECONDS = 300
LOCK_READ_RETRIES = 3
LOCK_READ_RETRY_SLEEP_SECONDS = 0.5
LOCK_REFRESH_RETRIES = 3
LOCK_REFRESH_RETRY_SLEEP_SECONDS = 0.2


@dataclass
class ProgressManager:
    """
    Manages evaluation progress with filesystem-based distributed synchronization.

    Handles the full lifecycle of evaluation tasks including:
    - Task configuration and seed management
    - Progress tracking and result storage
    - Worker-to-task mapping
    - Distributed locking for multi-server coordination

    Attributes:
        result_base_dir: Base directory for evaluation results (e.g., saved/eval_results)
        benchmark_id: Benchmark identifier
        run_id: Run identifier
    """

    result_base_dir: str
    benchmark_id: str
    run_id: str
    _pid: int = field(default_factory=os.getpid, repr=False)

    def __post_init__(self):
        """Initialize mutable state after dataclass creation."""
        self.lock = threading.Lock()
        self.config_list: list[dict] = []
        self.task_config_map: dict[str, dict] = {}
        self.task_seed_list: dict[str, list[str]] = {}
        self.task_num_test_list: dict[str, int] = {}
        self.result_list: dict[str, dict[str, float]] = {}
        self.worker_to_task_map: dict[str, tuple[str, str, dict]] = {}
        self.done = False
        self._host = socket.gethostname()
        atexit.register(self.cleanup_process_locks)

    @property
    def run_dir(self) -> Path:
        """Get the run directory path."""
        return Path(self.result_base_dir) / self.benchmark_id / self.run_id

    def _get_episode_dir(self, task_name: str, seed: str) -> Path:
        """Get the episode directory path."""
        return self.run_dir / task_name / seed

    def _get_result_info_path(self, task_name: str, seed: str) -> Path:
        """Get the result_info.json path for an episode."""
        return self._get_episode_dir(task_name, seed) / "result_info.json"

    def _get_lock_path(self, task_name: str, seed: str) -> Path:
        """Get the lock file path for an episode."""
        return self.run_dir / task_name / f"{seed}.lock"

    def _get_lock_guard_path(self, task_name: str, seed: str) -> str:
        """Get the sidecar file path used by SoftFileLock for an episode lock."""
        return str(self._get_lock_path(task_name, seed)) + ".lock"

    def _get_progress_file_path(self) -> Path:
        """Get the legacy progress.json path."""
        return self.run_dir / "progress.json"

    # ==================== Task Management ====================

    def add_evaluation_config(
        self,
        task_config_list: list[dict],
        task_seed_list: dict[str, list[str]],
        task_num_test_list: dict[str, int],
    ):
        """
        Add evaluation configs to the manager.

        Args:
            task_config_list: List of task configuration dicts
            task_seed_list: Dict mapping task_name to list of seed strings
            task_num_test_list: Dict mapping task_name to num_test count
        """
        with self.lock:
            self.config_list.extend(task_config_list)
            for task_name, seed_list in task_seed_list.items():
                if task_name not in self.task_seed_list:
                    self.task_seed_list[task_name] = []
                self.task_seed_list[task_name].extend(seed_list)
            for task_name, num_test in task_num_test_list.items():
                self.task_num_test_list[task_name] = num_test
            for task_config in task_config_list:
                self.task_config_map[task_config["task_name"]] = task_config
                if task_config["task_name"] not in self.result_list:
                    self.result_list[task_config["task_name"]] = {}
            self.done = False

    def load_progress(self):
        """
        Load progress from filesystem and filter out completed tasks.

        Scans episode folders first, falls back to legacy progress.json.
        """
        # Build all seeds dict for scanning
        all_seeds: dict[str, list[str]] = {}
        for task_name, num_test in self.task_num_test_list.items():
            all_seeds[task_name] = [str(i).zfill(3) for i in range(num_test)]

        scanned_results = self.scan_completed_episodes(all_seeds)

        # Update result_list with scanned/legacy results
        with self.lock:
            if scanned_results:
                print(f"Restoring progress from filesystem ({self.run_dir})")
                for task_name in self.result_list:
                    if task_name in scanned_results:
                        self.result_list[task_name] = scanned_results[task_name]

                # Filter out completed seeds from task_seed_list
                for task_name, seed_list in self.task_seed_list.items():
                    completed_seeds = set(self.result_list.get(task_name, {}).keys())
                    self.task_seed_list[task_name] = [
                        seed for seed in seed_list if seed not in completed_seeds
                    ]

    def _cleanup_empty_configs(self):
        """Remove configs whose seed lists are fully consumed."""
        self.config_list = [
            c
            for c in self.config_list
            if len(self.task_seed_list.get(c["task_name"], [])) > 0
        ]

    def reconcile_task_state_from_filesystem(self) -> bool:
        """
        Rebuild in-memory task/seed state from the configured task set and filesystem.

        Returns:
            True if every configured seed has a corresponding result_info.json on disk.
        """
        with self.lock:
            task_num_test_list = dict(self.task_num_test_list)
            task_config_map = dict(self.task_config_map)
            worker_to_task_map = dict(self.worker_to_task_map)

        all_seeds: dict[str, list[str]] = {}
        for task_name, num_test in task_num_test_list.items():
            all_seeds[task_name] = [str(i).zfill(3) for i in range(num_test)]

        scanned_results = self.scan_completed_episodes(all_seeds)

        with self.lock:
            rebuilt_task_seed_list: dict[str, list[str]] = {}
            rebuilt_config_list: list[dict] = []

            for task_name, all_task_seeds in all_seeds.items():
                completed_seeds = set(scanned_results.get(task_name, {}).keys())
                assigned_seeds = {
                    seed
                    for mapped_task_name, seed, _ in worker_to_task_map.values()
                    if mapped_task_name == task_name
                }
                pending_seeds = [
                    seed
                    for seed in all_task_seeds
                    if seed not in completed_seeds and seed not in assigned_seeds
                ]
                rebuilt_task_seed_list[task_name] = pending_seeds
                if pending_seeds and task_name in task_config_map:
                    rebuilt_config_list.append(task_config_map[task_name])

            self.result_list = {
                task_name: dict(scanned_results.get(task_name, {}))
                for task_name in task_num_test_list
            }
            self.task_seed_list = rebuilt_task_seed_list
            self.config_list = rebuilt_config_list
            self.done = all(
                len(self.result_list.get(task_name, {}))
                >= task_num_test_list[task_name]
                for task_name in task_num_test_list
            )

        return self.done

    def get_next_task(self, worker_id: str) -> tuple[dict | None, str | None]:
        """
        Get the next available task for a worker.

        Uses distributed locking to coordinate with other servers.
        Iterates through all tasks to find an available seed, skipping
        tasks whose seeds are all locked by other workers/servers.

        Args:
            worker_id: ID of the worker requesting a task

        Returns:
            Tuple of (config, seed) or (None, None) if no tasks available
        """
        with self.lock:
            # Remove configs whose seed lists are fully consumed
            self._cleanup_empty_configs()

            if len(self.config_list) == 0:
                self.done = True
                return None, None

            # Try each task to find an available (lockable) seed
            found_config = None
            found_seed = None

            for config in self.config_list:
                task_name = config["task_name"]

                task_seed = None
                seeds_to_remove: list[int] = []

                for idx, seed in enumerate(self.task_seed_list[task_name]):
                    if self.try_acquire_episode_lock(task_name, seed, worker_id):
                        # Check if already completed by another server
                        if self.is_episode_completed(task_name, seed):
                            # Read result, release lock, mark for removal
                            score_value = self.get_episode_score(task_name, seed)
                            if score_value is not None:
                                self.result_list[task_name][seed] = score_value
                            self.release_episode_lock(task_name, seed)
                            seeds_to_remove.append(idx)
                            continue

                        task_seed = seed
                        seeds_to_remove.append(idx)
                        break

                # Remove processed seeds (reverse to preserve indices)
                for idx in reversed(seeds_to_remove):
                    self.task_seed_list[task_name].pop(idx)

                if task_seed is not None:
                    found_config = config
                    found_seed = task_seed
                    break

            # Clean up any configs that became empty during iteration
            self._cleanup_empty_configs()

            if found_config is None:
                # All seeds across all tasks are currently locked
                if len(self.config_list) == 0:
                    self.done = True
                return None, None

            self.worker_to_task_map[worker_id] = (
                found_config["task_name"],
                found_seed,
                found_config,
            )

        self.print_progress()
        return found_config, found_seed

    def record_result(
        self, worker_id: str, score: float | None, release_lock: bool = True
    ) -> dict | None:
        """
        Record the result of a completed episode.

        Args:
            worker_id: ID of the worker that completed the episode
            result: Success rate value or None if failed

        Returns:
            Episode result dict or None
        """
        with self.lock:
            if worker_id not in self.worker_to_task_map:
                return None
            task_name, task_seed, config = self.worker_to_task_map[worker_id]

        episode_result = None
        if score is not None:
            episode_result = {
                "episode_id": os.path.join(
                    self.benchmark_id, self.run_id, task_name, task_seed
                ),
                "task_name": task_name,
                "seed": task_seed,
                "score": score,
                "sr": 1.0 if abs(score - 1) < 1e-6 else 0.0,
            }

        # Update state
        with self.lock:
            if score is not None:
                self.result_list[task_name][task_seed] = score
            else:
                self._restore_unfinished_episode_locked(task_name, task_seed, config)
            self.worker_to_task_map.pop(worker_id, None)

        # Release the episode lock unless the caller keeps it for async finalize.
        if release_lock:
            self.release_episode_lock(task_name, task_seed)

        return episode_result

    def _restore_unfinished_episode_locked(
        self, task_name: str, task_seed: str, config: dict
    ) -> None:
        """Put an assigned-but-unfinished episode back into the in-memory queue."""
        if task_seed in self.result_list.get(task_name, {}):
            return
        if task_seed not in self.task_seed_list.get(task_name, []):
            self.task_seed_list.setdefault(task_name, []).insert(0, task_seed)
        if not any(c["task_name"] == task_name for c in self.config_list):
            self.config_list.insert(0, config)
        self.done = False

    def get_worker_task(self, worker_id: str) -> tuple[str, str, dict] | None:
        """Get the current task assigned to a worker."""
        with self.lock:
            return self.worker_to_task_map.get(worker_id)

    def assign_task_to_worker(
        self, worker_id: str, task_name: str, task_seed: str, config: dict
    ):
        """Assign a task to a worker."""
        with self.lock:
            self.worker_to_task_map[worker_id] = (task_name, task_seed, config)

    def release_worker_task(self, worker_id: str) -> tuple[str, str, dict] | None:
        """
        Release a worker's current task and restore it to the queue.

        Args:
            worker_id: ID of the worker

        Returns:
            The released task tuple or None
        """
        with self.lock:
            if worker_id not in self.worker_to_task_map:
                return None

            task_name, task_seed, config = self.worker_to_task_map.pop(worker_id)

            # Release the lock for this task
            self.release_episode_lock(task_name, task_seed)

            # Restore to queue if not completed
            if task_seed not in self.result_list.get(task_name, {}):
                self._restore_unfinished_episode_locked(task_name, task_seed, config)

            return (task_name, task_seed, config)

    def cleanup_worker(self, worker_id: str):
        """
        Clean up all state associated with a worker.

        Called when a worker dies to release locks and restore episodes.

        Args:
            worker_id: ID of the worker to clean up
        """
        # Clean up locks held by this worker
        cleaned = self.cleanup_worker_locks(worker_id)
        with self.lock:
            for task_name, seed in cleaned:
                # Restore cleaned up episodes to task queue
                if seed not in self.result_list.get(task_name, {}):
                    if seed not in self.task_seed_list.get(task_name, []):
                        self.task_seed_list[task_name].insert(0, seed)

        # Release the worker's current task
        self.release_worker_task(worker_id)

    # ==================== Result Calculation ====================

    def calculate_result(self) -> dict[str, float]:
        """
        Calculate success rate per task.

        Returns:
            Dict mapping task_name (with prefix) to success rate
        """
        with self.lock:
            result_per_task = {}
            for task_name, result in self.result_list.items():
                if len(result) == 0:
                    result_per_task["*" + task_name] = {"score": 0.0, "sr": 0.0}
                elif len(result) < self.task_num_test_list[task_name]:
                    score_list = result.values()
                    sr_list = [
                        1.0 if abs(_score - 1) < 1e-6 else 0.0 for _score in score_list
                    ]
                    result_per_task[
                        f"*({len(result)}/{self.task_num_test_list[task_name]})"
                        + task_name
                    ] = {
                        "score": sum(score_list) / len(result),
                        "sr": sum(sr_list) / len(result),
                    }
                else:
                    score_list = result.values()
                    sr_list = [
                        1.0 if abs(_score - 1) < 1e-6 else 0.0 for _score in score_list
                    ]
                    result_per_task[
                        f"({len(result)}/{self.task_num_test_list[task_name]})"
                        + task_name
                    ] = {
                        "score": sum(score_list) / len(result),
                        "sr": sum(sr_list) / len(result),
                    }
            return result_per_task

    def check_finished(self) -> bool:
        """Check if all tasks are completed."""
        with self.lock:
            for task_name, result in self.result_list.items():
                if len(result) < self.task_num_test_list[task_name]:
                    return False
            return True

    def save_final_result(self, result_dir: str):
        """Save final evaluation result to file."""
        metric = self.calculate_result()
        result_file = os.path.join(
            result_dir, "result.json" if self.check_finished() else "progress.json"
        )
        self._atomic_json_write(result_file, metric)

    # ==================== Progress Display ====================

    def print_progress(self):
        """Print current progress to console."""
        with self.lock:
            print("=" * 10 + " Current ToDO-Task List " + "=" * 10)
            for task_name, task_seed_list in self.task_seed_list.items():
                print(f"{task_name}: {task_seed_list}")
            print("=" * 10 + " Progress " + "=" * 10)
            for task_name, result in self.result_list.items():
                print(f"{task_name}: {result}")
            print("=" * 10 + " Current Worker List " + "=" * 10)
            for w_id, (task_name, task_seed, _) in self.worker_to_task_map.items():
                print(f"{w_id}: {task_name}, {task_seed}")
            print("=" * 20)

    # ==================== Status API ====================

    def get_status(
        self, active_workers: list[str], refresh_filesystem: bool = False
    ) -> dict[str, Any]:
        """
        Get comprehensive job status for the /status endpoint.

        Args:
            active_workers: List of currently active worker IDs
            refresh_filesystem: When True, rescan result files and lock files.

        Returns:
            Status dict with status, counts, and results
        """
        with self.lock:
            task_num_test_list = dict(self.task_num_test_list)
            result_list = {
                task_name: dict(scores)
                for task_name, scores in self.result_list.items()
            }
            worker_to_task_map = dict(self.worker_to_task_map)

        if refresh_filesystem:
            all_seeds: dict[str, list[str]] = {}
            for task_name, num_test in task_num_test_list.items():
                all_seeds[task_name] = [str(i).zfill(3) for i in range(num_test)]
            completed = self.scan_completed_episodes(all_seeds)
            locked = self.get_locked_episodes()
        else:
            completed = result_list
            locked = {}
            for _, (task_name, task_seed, _) in worker_to_task_map.items():
                locked.setdefault(task_name, []).append(task_seed)

        # Count episodes
        total_episodes = sum(task_num_test_list.values())
        completed_episodes = sum(len(seeds) for seeds in completed.values())
        in_progress_episodes = sum(len(seeds) for seeds in locked.values())

        # Calculate results (score per task)
        results: dict[str, float] = {}
        for task_name, seeds_score in completed.items():
            if seeds_score:
                results[task_name] = sum(seeds_score.values()) / len(seeds_score)

        # Determine status
        if not task_num_test_list:
            status = "idle"
        elif completed_episodes == total_episodes:
            status = "complete"
        elif active_workers:
            status = "running"
        elif completed_episodes > 0 or in_progress_episodes > 0:
            status = "incomplete"
        else:
            status = "idle"

        return {
            "status": status,
            "benchmark_id": self.benchmark_id,
            "run_id": self.run_id,
            "total_episodes": total_episodes,
            "completed_episodes": completed_episodes,
            "in_progress_episodes": in_progress_episodes,
            "active_workers": active_workers,
            "results": results,
        }

    # ==================== Episode Completion Check ====================

    def is_episode_completed(self, task_name: str, seed: str) -> bool:
        """Check if an episode is completed by checking result_info.json existence."""
        result_info_path = self._get_result_info_path(task_name, seed)
        return result_info_path.exists()

    def get_episode_score(self, task_name: str, seed: str) -> float | None:
        """Get the score and success rate for a completed episode."""
        result_info_path = self._get_result_info_path(task_name, seed)
        if not result_info_path.exists():
            return None
        try:
            with open(result_info_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return float(data.get("score", 0.0))
        except (json.JSONDecodeError, ValueError, OSError):
            return None

    def scan_completed_episodes(
        self, task_seed_list: dict[str, list[str]]
    ) -> dict[str, dict[str, float]]:
        """Scan for completed episodes by checking score_info.json files."""
        results: dict[str, dict[str, float]] = {}
        for task_name, seeds in task_seed_list.items():
            results[task_name] = {}
            for seed in seeds:
                result_value = self.get_episode_score(task_name, seed)
                if result_value is not None:
                    results[task_name][seed] = result_value
        return results

    # ==================== Distributed Locking ====================

    def try_acquire_episode_lock(
        self, task_name: str, seed: str, worker_id: str
    ) -> bool:
        """
        Try to acquire a lock for an episode using atomic file creation.

        Returns True if lock was acquired, False if already locked.
        """
        lock_path = self._get_lock_path(task_name, seed)

        # Clean up stale lock first
        if self._is_lock_stale(lock_path):
            try:
                lock_path.unlink()
            except FileNotFoundError:
                # Expected if another process already cleaned the stale lock.
                pass
            except OSError as exc:
                print(f"Warning: failed to remove stale lock {lock_path}: {exc}")
        # Ensure parent directory exists
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_data = {
            "worker_id": worker_id,
            "timestamp": time.time(),
            "pid": self._pid,
            "host": self._host,
            "task_name": task_name,
            "seed": seed,
        }
        try:
            # Use O_CREAT | O_EXCL for atomic creation
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                os.write(fd, json.dumps(lock_data).encode("utf-8"))
            finally:
                os.close(fd)
            return True
        except FileExistsError:
            return False
        except OSError as e:
            print(f"Warning: Failed to acquire lock for {task_name}/{seed}: {e}")
            return False

    def release_episode_lock(self, task_name: str, seed: str) -> None:
        """Release the lock for an episode. Only releases if owned by this process."""
        lock_path = self._get_lock_path(task_name, seed)
        try:
            lock_data = self._read_lock_data(lock_path)
            if lock_data is None:
                return
            if lock_data.get("pid") != self._pid:
                return
            if lock_data.get("host") != self._host:
                return
            lock_path.unlink()
        except (json.JSONDecodeError, ValueError):
            # Corrupted lock file - only remove if stale
            if self._is_lock_stale(lock_path):
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    # Expected if the stale lock disappears between check and delete.
                    pass
                except OSError as exc:
                    print(
                        f"Warning: failed to remove corrupted lock {lock_path}: {exc}"
                    )
                    pass
        except OSError as exc:
            print(f"Warning: failed to release lock {task_name}/{seed}: {exc}")
            pass

    def _is_lock_stale(self, lock_path: Path) -> bool:
        """Check if a lock file is stale (old timestamp or dead PID)."""
        if not lock_path.exists():
            return False

        # Check mtime (heartbeat updates via utime) with retries
        mtime = 0
        for attempt in range(LOCK_READ_RETRIES):
            try:
                mtime = lock_path.stat().st_mtime
                break
            except OSError:
                if attempt < LOCK_READ_RETRIES - 1:
                    time.sleep(LOCK_READ_RETRY_SLEEP_SECONDS)
                    continue
                print("[lock] detected stale lock (stat failed) for ", lock_path)
                return True
        if time.time() - mtime > LOCK_TIMEOUT_SECONDS:
            print(
                "!!!!!!!!!!!!!! CLEANUP: Detected stale lock (timeout) for ", lock_path
            )
            return True

        # Consider completed episodes as stale locks
        try:
            task_name = lock_path.parent.name
            seed = lock_path.stem
            if self.is_episode_completed(task_name, seed):
                print("[lock] detected stale lock (already completed) for ", lock_path)
                return True
        except OSError:
            pass

        return False

    def cleanup_process_locks(self) -> None:
        """Remove locks owned by this process (best-effort cleanup)."""
        try:
            if not self.run_dir.exists():
                return
            for task_dir in self.run_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                for lock_file in task_dir.glob("*.lock"):
                    try:
                        with open(lock_file, "r", encoding="utf-8") as f:
                            lock_data = json.load(f)
                        if (
                            lock_data.get("pid") == self._pid
                            and lock_data.get("host") == self._host
                        ):
                            lock_file.unlink()
                    except (json.JSONDecodeError, OSError):
                        continue
        except OSError as exc:
            print(f"Warning: failed to clean up process locks in {self.run_dir}: {exc}")
            pass

    def cleanup_worker_locks(self, worker_id: str) -> list[tuple[str, str]]:
        """Clean up the lock held by a specific worker using worker_to_task_map."""
        cleaned = []

        with self.lock:
            if worker_id not in self.worker_to_task_map:
                return cleaned
            task_name, seed, _ = self.worker_to_task_map[worker_id]

        lock_path = self._get_lock_path(task_name, seed)
        if not lock_path.exists():
            return cleaned

        try:
            lock_path.unlink()
            cleaned.append((task_name, seed))
        except FileNotFoundError:
            # Lock file may vanish before this process removes it.
            pass
        except OSError as exc:
            print(f"Warning: failed to remove lock for worker {worker_id}: {exc}")
            pass

        return cleaned

    def refresh_episode_lock(self, task_name: str, seed: str) -> bool:
        """
        Refresh lock timestamp for an episode if owned by this process.

        Returns True if refreshed, False otherwise.
        """
        lock_path = self._get_lock_path(task_name, seed)
        last_reason = "unknown"
        for attempt in range(LOCK_REFRESH_RETRIES):
            lock_data = self._read_lock_data(lock_path)
            if lock_data is None:
                last_reason = "corrupted_or_missing_lock"
            elif lock_data.get("pid") != self._pid:
                last_reason = (
                    f"pid_mismatch file_pid={lock_data.get('pid')} self_pid={self._pid}"
                )
                break
            elif lock_data.get("host") != self._host:
                last_reason = f"host_mismatch file_host={lock_data.get('host')} self_host={self._host}"
                break
            else:
                try:
                    os.utime(lock_path, None)
                    return True
                except OSError as exc:
                    last_reason = f"utime_failed: {exc}"

            if attempt < LOCK_REFRESH_RETRIES - 1:
                time.sleep(LOCK_REFRESH_RETRY_SLEEP_SECONDS * (attempt + 1))
                continue

        print(f"[lock] heartbeat skipped {task_name}/{seed} {last_reason}")
        return False

    def refresh_worker_lock(self, worker_id: str) -> bool:
        """Refresh lock for the task currently assigned to a worker."""
        with self.lock:
            task = self.worker_to_task_map.get(worker_id)
        if task is None:
            return False
        task_name, seed, _ = task
        return self.refresh_episode_lock(task_name, seed)

    def _read_lock_data(self, lock_path: Path) -> dict | None:
        """Read lock JSON with retries to tolerate transient FS inconsistency."""
        for attempt in range(LOCK_READ_RETRIES):
            try:
                with open(lock_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                if attempt < LOCK_READ_RETRIES - 1:
                    time.sleep(LOCK_READ_RETRY_SLEEP_SECONDS * (2**attempt))
                    continue
                return None

    def get_locked_episodes(self) -> dict[str, list[str]]:
        """Get all currently locked episodes."""
        locked: dict[str, list[str]] = {}
        if not self.run_dir.exists():
            return locked

        for task_name in self.task_num_test_list:
            task_dir = self.run_dir / task_name
            if not task_dir.is_dir():
                continue

            for lock_file in task_dir.glob("*.lock"):
                if not self._is_lock_stale(lock_file):
                    if task_name not in locked:
                        locked[task_name] = []
                    locked[task_name].append(lock_file.stem)

        return locked

    # ==================== Utility ====================

    def _atomic_json_write(self, file_path: str, data: Any) -> None:
        """Atomically write JSON data to file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except OSError:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    # Temporary file may already be deleted by another cleanup path.
                    pass
                except OSError as exc:
                    print(
                        f"Warning: failed to cleanup temporary file {tmp_path}: {exc}"
                    )
            raise
