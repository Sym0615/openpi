"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from filelock import SoftFileLock, FileLock, Timeout
import os
from pathlib import Path
import pickle
import copy
import importlib
import tempfile

import csv
from huggingface_hub import HfApi
import json
import yaml
import trimesh


def increment_index_in_file(
    file_path: str,
    line_number: int = 49,
    keyword: str = "file = ",
    increment: int = 1,
    delimiter: str = "=",
) -> None:
    with open(file_path, "r") as file:
        lines = file.readlines()
    if line_number <= len(lines):
        line = lines[line_number - 1].strip()
        if line.startswith(keyword):
            try:
                index = int(line.split(delimiter)[1].strip())
                index += increment
                lines[line_number - 1] = f"{keyword}{index}\n"
                print(f"Updated line: {lines[line_number - 1].strip()}")
            except ValueError:
                print(f"Error: The value after '{keyword}' is not an integer.")
                return
        else:
            print(f"Error: The specified line does not start with '{keyword}'.")
            return
    else:
        print("Error: The specified line number is out of range.")
        return
    with open(file_path, "w") as file:
        file.writelines(lines)


def load_json(config_path: str) -> dict:
    with open(config_path, "r") as file:
        config = json.load(file)
    return config


def load_default_config(
    current_dir: str, config_name: str, anygrasp_mode: str = "default"
) -> dict:
    config_path = os.path.join(current_dir, "assets/configs", config_name)
    if not os.path.exists(config_path):
        config = {}
    else:
        with open(config_path, "r") as file:
            config = json.load(file)
    if "ANYGRASP_ADDR" not in config:
        config["ANYGRASP_ADDR"] = load_yaml(
            os.path.join(current_dir, "configs/miscs/anygrasp.yml")
        )[anygrasp_mode]
    if "ASSETS_DIR" not in config:
        config["ASSETS_DIR"] = os.path.join(current_dir, "saved/assets")
    if "DEMONSTRATION_DIR" not in config:
        make_dir(os.path.join(current_dir, "saved/demonstrations"))
        config["DEMONSTRATION_DIR"] = os.path.join(current_dir, "saved/demonstrations")
    if "EVAL_RESULT_DIR" not in config:
        make_dir(os.path.join(current_dir, "saved/eval_results"))
        config["EVAL_RESULT_DIR"] = os.path.join(current_dir, "saved/eval_results")
    if "TASKS_DIR" not in config:
        make_dir(os.path.join(current_dir, "saved/tasks"))
        config["TASKS_DIR"] = os.path.join(current_dir, "saved/tasks")
    if "TEST_USD_NAME" not in config:
        config["TEST_USD_NAME"] = "base"
    config["current_dir"] = current_dir
    return config


def load_task_config(config_path: str) -> dict:
    config_path_str = str(config_path)
    if config_path_str.endswith(".yml") or config_path_str.endswith(".yaml"):
        return load_yaml(config_path)
    elif config_path_str.endswith(".json"):
        return load_json(config_path)
    else:
        raise ValueError(f"Unsupported file type: {config_path_str}")


def load_yaml(config_path: str) -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return json.loads(json.dumps(config))


def read_task_csv_to_dict(file_path: str) -> dict:
    result = {}
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            scene = row["\ufeffscene"]
            instruction = row["instruction"]
            if scene not in result:
                result[scene] = {"instruction": instruction, "target": [], "anchor": []}
            targets = []
            anchors = []
            for i in range(1, 4):
                target_key = f"target_{i}"
                anchor_key = f"anchor_{i}"
                if row[target_key]:
                    targets.append(row[target_key].split("/"))
                if row[anchor_key]:
                    anchors.append(row[anchor_key].split("/"))
            result[scene]["target"].extend(targets)
            result[scene]["anchor"].extend(anchors)
    return result


def save_dict_to_json(data, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    # print(f"JSON saved: {file_path}")


def save_dict_to_json_atomic(data: dict, file_path: str) -> None:
    """Atomically save dict to JSON using write-then-rename pattern.

    This function writes to a temporary file first, then atomically renames
    it to the target path. This prevents data corruption if the process
    crashes mid-write.
    """
    dir_path = os.path.dirname(file_path) or "."
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", prefix=".json_", dir=dir_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, file_path)  # Atomic on POSIX
    except (OSError, TypeError, ValueError):
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            # Temp file may already be removed after write failure.
            pass
        except OSError as exc:
            print(f"Warning: failed to remove temporary file {tmp_path}: {exc}")
        raise


def save_dict_as_yaml(data: dict, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(data, yaml_file, allow_unicode=True, default_flow_style=False)


def save_dict_as_pkl(dict: dict, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(dict, f)


class _CompatUnpickler(pickle.Unpickler):
    """Handle numpy internal module path differences across environments."""

    _MODULE_ALIASES = {
        "numpy._core": "numpy.core",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy.core": "numpy._core",
        "numpy.core.multiarray": "numpy._core.multiarray",
        "numpy.core.numeric": "numpy._core.numeric",
    }

    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            alias = self._MODULE_ALIASES.get(module)
            if alias is None:
                raise
            imported = importlib.import_module(alias)
            return getattr(imported, name)


def load_dict_from_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        return _CompatUnpickler(f).load()


def check_glb_properties(file_path: str) -> tuple[bool, int] | tuple[None, None]:
    try:
        scene = trimesh.load(file_path)
        total_vertex_count = 0
        has_normal_map = False
        for name, mesh in scene.geometry.items():  # type: ignore[attr-defined]
            total_vertex_count += len(mesh.vertices)
            if isinstance(mesh, trimesh.Trimesh):
                if hasattr(mesh.visual, "material"):
                    material = mesh.visual.material  # type: ignore[attr-defined]
                    if material and getattr(material, "normalTexture", None):
                        has_normal_map = True
        return has_normal_map, total_vertex_count
    except (OSError, RuntimeError, ValueError) as e:
        print(f"Error processing {file_path}: {e}")
        return None, None


def is_glb_vaild(
    file_path: str, min_vertex_count: int = 1000, debug: bool = False
) -> bool:
    if not os.path.exists(file_path):
        return False
    has_normal_map, total_vertex_count = check_glb_properties(file_path)
    if total_vertex_count is None:
        return False
    if debug:
        if not has_normal_map and total_vertex_count < min_vertex_count:
            print(
                f"has_normal_map: {has_normal_map}, total_vertex_count: {total_vertex_count}"
            )
    return has_normal_map or (total_vertex_count > min_vertex_count)


def check_uids(sorted_uids: list[str] | None, directory: str, json_file: str) -> bool:
    if sorted_uids is None:
        print("sorted_uids is None")
        return False
    for uid in sorted_uids:
        if uid == "00000000000000000000000000000000":
            continue
        usd_file = os.path.join(directory, f"{uid}.usd")
        if not os.path.isfile(usd_file):
            print(f"file not exist: {usd_file}")
            return False
        with open(json_file, "r") as f:
            data = json.load(f)
            if uid not in data:
                print(f"key not exist: {uid}")
                return False
    return True


def make_dir(dir_path: str) -> None:
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def clean_stale_filelocks(lock_dir: str, timeout: float = 0.0) -> None:
    for root, dirs, files in os.walk(lock_dir):
        for file in files:
            if file.endswith(".lock"):
                lock_path = os.path.join(root, file)
                try:
                    with FileLock(lock_path, timeout=timeout):
                        print(f"[Clean] Removing stale lock: {lock_path}")
                        os.remove(lock_path)
                except Timeout:
                    print(f"[Keep] Lock in use: {lock_path}")


def record_log(log_path: str, info: str) -> None:
    lock_file = os.path.join(log_path, "log_soft.lock")
    try:
        with SoftFileLock(lock_file, timeout=600.0):
            log_pkl_path = os.path.join(log_path, "log.pkl")
            if os.path.exists(log_pkl_path):
                failed_log = load_dict_from_pkl(log_pkl_path)
            else:
                failed_log = {}
            if info not in failed_log:
                failed_log[info] = 1
            else:
                failed_log[info] += 1
            save_dict_as_pkl(failed_log, log_pkl_path)
    except Timeout as exc:
        raise TimeoutError(
            "Filelock timeout, try to delete the lock file by python "
            "standalone_tools/cleanup_lockfiles.py"
        ) from exc


def report_log(log_path: str) -> dict:
    log_pkl_path = os.path.join(log_path, "log.pkl")
    if os.path.exists(log_pkl_path):
        failed_log = load_dict_from_pkl(log_pkl_path)
    else:
        failed_log = {}
    return failed_log


def check_benchmark_version(benchmark_id: str) -> bool:
    api = HfApi()
    files = api.list_repo_files(repo_id=benchmark_id, repo_type="dataset")
    if ".version" in files:
        return True
    else:
        return False
