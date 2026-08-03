#!/usr/bin/env python3
# ruff: noqa
"""Local filesystem reader for LeRobot-style episode parquet/mp4 data.

This module can also be run as a smoke-testable CLI for datasets whose
directory layout matches either LeRobot
v2.1 or v3.0:

    <dataset_root>/meta/info.json
    <dataset_root>/meta/tasks.jsonl
    <dataset_root>/meta/episodes.jsonl
    <dataset_root>/data/chunk-000/episode_000000.parquet
    <dataset_root>/videos/chunk-000/<video_key>/episode_000000.mp4

Recommended mixture example:

    python -m openpi.training.petrel_lerobot_reader \
      --dataset-config configs/petrel/pi05_petrel.yaml \
      --index 0 --with-video
"""

from __future__ import annotations

import argparse
import bisect
from collections import OrderedDict
import copy
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from openpi.training import profiling as _training_profiling


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
V3_TASKS_PATH = "meta/tasks.parquet"
V3_EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

_PROFILE_ENABLED = False
_PROFILE_EVENTS: list[tuple[str, float, dict[str, Any]]] = []
OPENPI_REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT_PATHS = (str(OPENPI_REPO_ROOT / "data/petrel_cache"),)
FRAME_FILTER_FILENAMES = ("frame_filter.jsonl", "frame_filter_sidecar.jsonl")


def _is_file_accessible(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _local_path_candidates(path: str | Path | None) -> list[Path]:
    if path is None:
        return []
    text = str(Path(str(path)).expanduser())
    candidates = [text]
    for src_root in CACHE_ROOT_PATHS:
        src_root = src_root.rstrip("/")
        if text == src_root or text.startswith(src_root + "/"):
            suffix = text[len(src_root) :]
            for dst_root in CACHE_ROOT_PATHS:
                candidates.append(dst_root.rstrip("/") + suffix)
            break

    out: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            out.append(Path(candidate))
    return out


def _profile_record(name: str, elapsed_s: float, **info: Any) -> None:
    _training_profiling.record(name, elapsed_s, num_bytes=info.get("bytes"))
    if _PROFILE_ENABLED:
        _PROFILE_EVENTS.append((name, float(elapsed_s), info))


def _io_profile_name(backend: str, path: str | Path) -> str:
    suffix = str(path).lower().split("?", 1)[0]
    if suffix.endswith(".parquet"):
        kind = "parquet"
    elif suffix.endswith(".mp4"):
        kind = "video"
    elif "/meta/" in suffix or suffix.startswith("meta/"):
        kind = "metadata"
    else:
        kind = "other"
    return f"io.{backend}.{kind}_read"


def _profile_print() -> None:
    if not _PROFILE_ENABLED:
        return
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name, elapsed_s, _ in _PROFILE_EVENTS:
        totals[name] = totals.get(name, 0.0) + elapsed_s
        counts[name] = counts.get(name, 0) + 1
    print("[profile] summary", flush=True)
    for name, total in sorted(totals.items(), key=lambda item: item[1], reverse=True):
        print(f"[profile] {name}: total={total:.3f}s count={counts[name]}", flush=True)
    print("[profile] events", flush=True)
    for name, elapsed_s, info in _PROFILE_EVENTS:
        info_text = " ".join(f"{k}={v}" for k, v in info.items() if v is not None)
        print(f"[profile] {name}: {elapsed_s:.3f}s {info_text}".rstrip(), flush=True)


class LocalFileClient:
    """Read dataset objects from one isolated local filesystem root."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def resolve_path(self, path: str) -> Path:
        """Resolve a dataset path without allowing it to escape the configured root."""
        local_path = Path(path).expanduser()
        if not local_path.is_absolute():
            raise ValueError(f"Local dataset path must be absolute, got {path!r}.")
        resolved = local_path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"Local dataset path escapes configured root {self.root}: {resolved}")
        return resolved

    def get_bytes(self, path: str) -> bytes:
        resolved = self.resolve_path(path)
        try:
            t0 = time.perf_counter()
            data = resolved.read_bytes()
        except OSError as exc:
            raise FileNotFoundError(f"[LocalFileClient] failed to read {resolved}: {exc}") from exc
        _profile_record(
            _io_profile_name("local", resolved),
            time.perf_counter() - t0,
            bytes=len(data),
            path=resolved.name,
        )
        return data


def petrel_join(base_url: str, *parts: str) -> str:
    url = str(base_url).rstrip("/")
    for part in parts:
        url = url + "/" + str(part).lstrip("/")
    return url


def _is_absolute_dataset_path(path: str) -> bool:
    return Path(str(path)).is_absolute()


def _json_loads(raw: bytes) -> Any:
    t0 = time.perf_counter()
    out = json.loads(raw.decode("utf-8"))
    _profile_record("json.loads", time.perf_counter() - t0, bytes=len(raw))
    return out


def _jsonl_loads(raw: bytes) -> list[dict[str, Any]]:
    t0 = time.perf_counter()
    records = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    _profile_record("jsonl.loads", time.perf_counter() - t0, bytes=len(raw), records=len(records))
    return records


def _shape_tuple_inplace(info: dict[str, Any]) -> dict[str, Any]:
    for feature in info.get("features", {}).values():
        if "shape" in feature:
            feature["shape"] = tuple(feature["shape"])
    return info


def _merge_mapping(parent: Any, child: Any) -> dict[str, Any]:
    out = dict(parent) if isinstance(parent, dict) else {}
    if not isinstance(child, dict):
        return out
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_mapping(out[key], value)
        else:
            out[key] = value
    return out


def _modality_key(modalities: dict[str, Any], name: str, default: str) -> str:
    spec = modalities.get(name)
    if spec is None:
        return default
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        key = spec.get("key", spec.get("source_key"))
        if key in (None, "", "auto"):
            return default
        return str(key)
    return default


def _modality_keys(modalities: dict[str, Any], name: str, default: str) -> list[str]:
    spec = modalities.get(name)
    if spec is None:
        return [default]
    if isinstance(spec, str):
        return [spec] if spec not in ("", "auto") else [default]
    if isinstance(spec, dict):
        keys = spec.get("keys", spec.get("source_keys"))
        if isinstance(keys, (list, tuple)):
            out = [str(key) for key in keys if key not in (None, "", "auto")]
            return out or [default]
        key = spec.get("key", spec.get("source_key"))
        if key in (None, "", "auto"):
            return [default]
        return [str(key)]
    return [default]


def _modality_is_auto(modalities: dict[str, Any], name: str) -> bool:
    spec = modalities.get(name)
    if isinstance(spec, str):
        return spec == "auto"
    if isinstance(spec, dict):
        if "keys" in spec or "source_keys" in spec:
            keys = spec.get("keys", spec.get("source_keys"))
            return keys in (None, "", "auto", [])
        return spec.get("key", spec.get("source_key")) in (None, "", "auto")
    return False


def _feature_dtype(info: dict[str, Any], key: str) -> str:
    return str(info.get("features", {}).get(key, {}).get("dtype", "")).lower()


def _is_numeric_feature(info: dict[str, Any], key: str) -> bool:
    dtype = _feature_dtype(info, key)
    return dtype not in {"video", "image", "string", "str"}


def _feature_dim(info: dict[str, Any], key: str) -> int | None:
    shape = info.get("features", {}).get(key, {}).get("shape")
    if not shape:
        return None
    if isinstance(shape, int):
        return int(shape)
    if isinstance(shape, (list, tuple)) and len(shape) == 1:
        return int(shape[0])
    if isinstance(shape, (list, tuple)) and _is_numeric_feature(info, key):
        total = 1
        for dim in shape:
            total *= int(dim)
        return int(total)
    return None


def _feature_dims(info: dict[str, Any], keys: list[str]) -> int | None:
    total = 0
    for key in keys:
        dim = _feature_dim(info, key)
        if dim is None:
            return None
        total += int(dim)
    return total


def _pad_last_dim(x: torch.Tensor, target_dim: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    if target_dim is None:
        target_dim = int(x.shape[-1])
    if int(x.shape[-1]) > int(target_dim):
        raise ValueError(f"Cannot pad tensor with last dim {x.shape[-1]} to smaller target_dim={target_dim}")
    pad_dim = int(target_dim) - int(x.shape[-1])
    if pad_dim > 0:
        x = torch.nn.functional.pad(x, (0, pad_dim))
    mask = torch.zeros(int(target_dim), dtype=torch.bool, device=x.device)
    if pad_dim > 0:
        mask[-pad_dim:] = True
    return x, mask


def _safe_name(text: str) -> str:
    out = []
    for ch in str(text).strip("/"):
        out.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "_")
    return "_".join("".join(out).split("/"))


def _safe_rel_path(text: str) -> str:
    parts = [
        _safe_name(part)
        for part in str(text).strip("/").split("/")
        if part.strip("/") and part not in {".", ".."}
    ]
    return "/".join(part for part in parts if part)


def _relative_dataset_cache_path(dataset_root: str | None) -> str:
    return _safe_rel_path(str(dataset_root or ""))


def _format_template(
    template: str | None,
    *,
    name: str,
    path: str,
    remote: str | None = None,
    source: str | None = None,
    relative_remote: str | None = None,
    embodiment: str | None = None,
    **extra: Any,
) -> str | None:
    if template in (None, ""):
        return None
    clean_path = str(path).strip("/")
    clean_remote = _safe_rel_path(relative_remote or _relative_remote_cache_path(remote) or remote or "remote")
    values = {
        "name": _safe_name(name),
        "path": clean_path,
        "safe_path": _safe_name(clean_path),
        "remote": clean_remote,
        "relative_remote": clean_remote,
        "safe_remote": _safe_name(remote or clean_remote or "remote"),
        "source": _safe_name(source or remote or "source"),
        "embodiment": _safe_name(embodiment or "default"),
    }
    for key, value in extra.items():
        values[str(key)] = _safe_name(value) if value is not None else ""
    return str(template).format(**values)


def _weight_override_value(value: Any) -> tuple[float | None, bool | None]:
    if value is None:
        return None, None
    if isinstance(value, dict):
        weight = value.get("dataset_weight", value.get("weight"))
        distribute = value.get("distribute_weights")
        return (None if weight is None else float(weight), None if distribute is None else bool(distribute))
    return float(value), None


def _apply_weight_override(weight: float, distribute: bool, override: Any) -> tuple[float, bool]:
    new_weight, new_distribute = _weight_override_value(override)
    if new_weight is not None:
        weight = float(new_weight)
    if new_distribute is not None:
        distribute = bool(new_distribute)
    return float(weight), bool(distribute)


def _lookup_weight_override(overrides: Any, keys: list[Any] | tuple[Any, ...]) -> Any:
    if not isinstance(overrides, dict):
        return None
    for key in keys:
        if key is None:
            continue
        text = str(key)
        if text in overrides:
            return overrides[text]
        safe = _safe_name(text)
        if safe in overrides:
            return overrides[safe]
    return None


def _table_to_tensors(table: Any) -> dict[str, torch.Tensor | list[Any]]:
    out: dict[str, torch.Tensor | list[Any]] = {}
    for name in table.column_names:
        column = table[name]
        values = column.to_pylist()
        if not values:
            out[name] = torch.empty(0)
            continue
        first = values[0]
        if isinstance(first, str):
            out[name] = values
            continue
        arr = np.asarray(values)
        if arr.dtype == object:
            arr = np.stack(values)
        if arr.dtype.kind in {"U", "S", "O"}:
            out[name] = values
        else:
            if arr.dtype.kind == "f":
                arr = arr.astype(np.float32, copy=False)
            out[name] = torch.from_numpy(np.asarray(arr))
    return out


def _select_rows(data: dict[str, torch.Tensor | list[Any]], key: str, indices: list[int]) -> torch.Tensor | list[Any]:
    if key not in data:
        raise KeyError(f"Missing column {key!r}; available columns: {sorted(data.keys())}")
    value = data[key]
    if isinstance(value, torch.Tensor):
        return value[torch.as_tensor(indices, dtype=torch.long)]
    return [value[i] for i in indices]


def _as_feature_matrix(value: torch.Tensor, key: str) -> torch.Tensor:
    if value.ndim == 1:
        return value.unsqueeze(-1)
    if value.ndim > 2:
        return value.flatten(1)
    if value.ndim == 0:
        raise ValueError(f"Column {key!r} produced a scalar tensor; expected one value per frame.")
    return value


def _select_rows_concat(data: dict[str, torch.Tensor | list[Any]], keys: list[str], indices: list[int]) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for key in keys:
        selected = _select_rows(data, key, indices)
        if not isinstance(selected, torch.Tensor):
            raise TypeError(f"Column {key!r} is not numeric and cannot be used as state/action.")
        parts.append(_as_feature_matrix(selected, key))
    if not parts:
        raise ValueError("No feature keys configured for state/action.")
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


def _parquet_num_rows(data: dict[str, torch.Tensor | list[Any]]) -> int:
    lengths: list[int] = []
    for value in data.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            lengths.append(int(value.shape[0]))
        elif isinstance(value, list):
            lengths.append(len(value))
    if not lengths:
        raise RuntimeError("Could not infer parquet row count from loaded episode columns.")
    return min(lengths)


CANONICAL_LAYOUT_DIMS = {
    "robot": 80,
    "openpi": 32,
}


def _canonical_adapter_dim(adapter: dict[str, Any]) -> int | None:
    if not isinstance(adapter, dict):
        return None
    dim = adapter.get("dim", adapter.get("target_dim"))
    if dim is not None:
        return int(dim)
    layout = adapter.get("layout")
    if layout is None:
        return None
    return CANONICAL_LAYOUT_DIMS.get(str(layout))


def _slice_last_dim(x: torch.Tensor, raw_slice: list[Any] | tuple[Any, ...] | None) -> torch.Tensor:
    x = _as_feature_matrix(x, "canonical")
    if raw_slice is None:
        return x
    if not isinstance(raw_slice, (list, tuple)) or len(raw_slice) != 2:
        raise ValueError(f"raw_slice must be [start, end], got {raw_slice!r}")
    start = raw_slice[0]
    end = raw_slice[1]
    start = None if start is None else int(start)
    end = None if end is None else int(end)
    return x[..., start:end]


def _select_feature(
    data: dict[str, torch.Tensor | list[Any]],
    key: str,
    indices: list[int],
    raw_slice: list[Any] | tuple[Any, ...] | None = None,
) -> torch.Tensor:
    value = _select_rows(data, key, indices)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Column {key!r} is not numeric and cannot be used by canonical_adapter.")
    return _slice_last_dim(value, raw_slice).to(torch.float32)


def _component_raw_slice(component: dict[str, Any], kind: str) -> list[Any] | tuple[Any, ...] | None:
    return component.get(f"{kind}_raw_slice") or component.get("raw_slice")


def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_to_rotvec_wxyz(q: torch.Tensor) -> torch.Tensor:
    q = _normalize_quat(q.to(torch.float32))
    sign = torch.where(q[..., :1] < 0, -1.0, 1.0)
    q = q * sign
    w = q[..., :1].clamp(-1.0, 1.0)
    xyz = q[..., 1:]
    sin_half = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w)
    scale = torch.where(sin_half > 1e-8, angle / sin_half, 2.0 * torch.ones_like(sin_half))
    return xyz * scale


def _quat_to_rotvec_xyzw(q: torch.Tensor) -> torch.Tensor:
    return _quat_to_rotvec_wxyz(torch.cat([q[..., 3:4], q[..., :3]], dim=-1))


def _quat_conj_wxyz(q: torch.Tensor) -> torch.Tensor:
    q = _normalize_quat(q.to(torch.float32))
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = _normalize_quat(a.to(torch.float32))
    b = _normalize_quat(b.to(torch.float32))
    aw, ax, ay, az = a[..., 0:1], a[..., 1:2], a[..., 2:3], a[..., 3:4]
    bw, bx, by, bz = b[..., 0:1], b[..., 1:2], b[..., 2:3], b[..., 3:4]
    return _normalize_quat(
        torch.cat(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dim=-1,
        )
    )


def _rpy_to_rotvec(rpy: torch.Tensor) -> torch.Tensor:
    # Roll-pitch-yaw in XYZ convention, converted through quaternion.
    half = rpy.to(torch.float32) * 0.5
    cr, cp, cy = torch.cos(half[..., 0:1]), torch.cos(half[..., 1:2]), torch.cos(half[..., 2:3])
    sr, sp, sy = torch.sin(half[..., 0:1]), torch.sin(half[..., 1:2]), torch.sin(half[..., 2:3])
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return _quat_to_rotvec_wxyz(torch.cat([qw, qx, qy, qz], dim=-1))


def _pose_to_xyz_rotvec(value: torch.Tensor, pose_input: str) -> torch.Tensor:  # return xyz_rotvec
    value = _as_feature_matrix(value, "pose").to(torch.float32)
    if pose_input == "xyz_rotvec":
        if value.shape[-1] != 6:
            raise ValueError(f"xyz_rotvec pose expects 6 dims, got {value.shape[-1]}")
        return value
    if pose_input == "xyz_quat_wxyz":
        if value.shape[-1] != 7:
            raise ValueError(f"xyz_quat_wxyz pose expects 7 dims, got {value.shape[-1]}")
        return torch.cat([value[..., :3], _quat_to_rotvec_wxyz(value[..., 3:7])], dim=-1)
    if pose_input == "xyz_quat_xyzw":
        if value.shape[-1] != 7:
            raise ValueError(f"xyz_quat_xyzw pose expects 7 dims, got {value.shape[-1]}")
        return torch.cat([value[..., :3], _quat_to_rotvec_xyzw(value[..., 3:7])], dim=-1)
    if pose_input in {"xyz_rpy", "delta_xyz_rpy"}:
        if value.shape[-1] != 6:
            raise ValueError(f"{pose_input} pose expects 6 dims, got {value.shape[-1]}")
        return torch.cat([value[..., :3], _rpy_to_rotvec(value[..., 3:6])], dim=-1)
    raise ValueError(f"Unsupported canonical_adapter pose_input={pose_input!r}")


def _pose_output_format(kind: str, adapter: dict[str, Any], component: dict[str, Any]) -> str:
    return str(
        component.get(f"{kind}_pose_output")
        or component.get("pose_output")
        or adapter.get(f"{kind}_pose_output")
        or adapter.get("pose_output")
        or "xyz_rotvec"
    )


def _rotvec_to_rot6d(rotvec: torch.Tensor) -> torch.Tensor:
    rotation = _rotvec_to_matrix(rotvec)
    return rotation[..., :, :2].transpose(-1, -2).reshape(*rotation.shape[:-2], 6)


def _format_xyz_rotvec_pose(pose: torch.Tensor, pose_output: str) -> torch.Tensor:
    pose_output = str(pose_output)
    if pose_output in {"xyz_rotvec", "delta_xyz_rotvec"}:
        return pose
    if pose_output == "xyz_rot6d":
        return torch.cat([pose[..., :3], _rotvec_to_rot6d(pose[..., 3:6])], dim=-1)
    if pose_output in {"xyz_rpy", "delta_xyz_rpy"}:
        return torch.cat([pose[..., :3], _rotvec_to_rpy(pose[..., 3:6])], dim=-1)
    raise ValueError(f"Unsupported canonical_adapter pose_output={pose_output!r}")


def _rotvec_to_rpy(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert a rotation vector to roll/pitch/yaw using the reader's XYZ convention."""
    rotation = _rotvec_to_matrix(rotvec)
    sin_pitch = (-rotation[..., 2, 0]).clamp(-1.0, 1.0)
    pitch = torch.asin(sin_pitch)
    cos_pitch = torch.cos(pitch)
    regular = cos_pitch.abs() > 1e-6
    roll_regular = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
    yaw_regular = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    # At gimbal lock, choose yaw=0 and absorb the remaining rotation into roll.
    roll_lock = torch.atan2(-rotation[..., 1, 2], rotation[..., 1, 1])
    roll = torch.where(regular, roll_regular, roll_lock)
    yaw = torch.where(regular, yaw_regular, torch.zeros_like(yaw_regular))
    return torch.stack([roll, pitch, yaw], dim=-1)


def _pose_xyz_quat_wxyz(value: torch.Tensor, pose_input: str) -> tuple[torch.Tensor, torch.Tensor]:
    value = _as_feature_matrix(value, "pose").to(torch.float32)
    if pose_input == "xyz_quat_wxyz":
        if value.shape[-1] != 7:
            raise ValueError(f"xyz_quat_wxyz pose expects 7 dims, got {value.shape[-1]}")
        return value[..., :3], _normalize_quat(value[..., 3:7])
    if pose_input == "xyz_quat_xyzw":
        if value.shape[-1] != 7:
            raise ValueError(f"xyz_quat_xyzw pose expects 7 dims, got {value.shape[-1]}")
        return value[..., :3], _normalize_quat(torch.cat([value[..., 6:7], value[..., 3:6]], dim=-1))
    raise ValueError(f"state-derived action requires quaternion pose input, got {pose_input!r}")


def _pose_delta_xyz_rotvec(current: torch.Tensor, future: torch.Tensor, pose_input: str) -> torch.Tensor:
    cur_xyz, cur_quat = _pose_xyz_quat_wxyz(current, pose_input)
    next_xyz, next_quat = _pose_xyz_quat_wxyz(future, pose_input)
    delta_quat = _quat_mul_wxyz(_quat_conj_wxyz(cur_quat), next_quat)
    return torch.cat([next_xyz - cur_xyz, _quat_to_rotvec_wxyz(delta_quat)], dim=-1)


def _skew_from_vector(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zero = torch.zeros_like(x)
    return torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero],
        dim=-1,
    ).reshape(*v.shape[:-1], 3, 3)


def _rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    rv = rotvec.to(torch.float32)
    theta = rv.norm(dim=-1, keepdim=True)
    axis = rv / theta.clamp_min(1e-8)
    k = _skew_from_vector(axis)
    eye = torch.eye(3, dtype=rv.dtype, device=rv.device).expand(*rv.shape[:-1], 3, 3)
    theta_m = theta.unsqueeze(-1)
    return eye + torch.sin(theta_m) * k + (1.0 - torch.cos(theta_m)) * (k @ k)


def _matrix_to_rotvec(rotation: torch.Tensor) -> torch.Tensor:
    r = rotation.to(torch.float32)
    trace = r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    vee = torch.stack(
        [r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]],
        dim=-1,
    )
    sin_theta = torch.sin(theta)
    scale = torch.where(sin_theta.abs() > 1e-6, theta / (2.0 * sin_theta).clamp_min(1e-8), torch.full_like(theta, 0.5))
    return vee * scale.unsqueeze(-1)


def _rotation_matrix_from_config(value: Any, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if value is None:
        return torch.eye(3, dtype=dtype, device=device)
    if isinstance(value, dict):
        value = value.get("matrix", value.get("rotation", value.get("name", value.get("preset"))))
    if isinstance(value, str):
        name = value.strip().lower()
        presets = {
            "identity": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "opencv": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "ros_to_opencv": [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
            "robot_to_opencv": [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
            "z_up_to_opencv": [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
            "x_forward_y_left_z_up_to_opencv": [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
            "robot_y_left_to_opencv": [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
            "robot_y_right_to_opencv": [[0, 1, 0], [0, 0, -1], [1, 0, 0]],
            "opencv_to_ros": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
            "opencv_to_robot": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
            "opencv_to_robot_y_left": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
            "opencv_to_robot_y_right": [[0, 0, 1], [1, 0, 0], [0, -1, 0]],
            # EEF-local transforms. These matrices are source_ee_from_canonical_ee:
            # their columns are canonical OpenCV-EEF axes expressed in the source EEF frame.
            # Use the preset whose source EEF axis should become canonical +Z
            # (gripper/finger direction) before entering the network.
            "ee_source_x_to_canonical_z": [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
            "ee_source_y_to_canonical_z": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
            "ee_source_z_to_canonical_z": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "ee_source_neg_x_to_canonical_z": [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
            "ee_source_neg_y_to_canonical_z": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
            "ee_source_neg_z_to_canonical_z": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        }
        if name not in presets:
            raise ValueError(f"Unknown frame transform preset {value!r}")
        value = presets[name]
    mat = torch.as_tensor(value, dtype=dtype, device=device)
    if mat.shape != (3, 3):
        raise ValueError(f"frame transform matrix must be 3x3, got {tuple(mat.shape)}")
    return mat


def _merge_frame_transform(adapter: dict[str, Any], component: dict[str, Any], kind: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for source in (
        adapter.get("frame_transform"),
        adapter.get(f"{kind}_frame_transform"),
        component.get("frame_transform"),
        component.get(f"{kind}_frame_transform"),
    ):
        if isinstance(source, dict):
            cfg.update(source)
    return cfg


def _pose_is_delta(kind: str, pose_input: str, adapter: dict[str, Any], component: dict[str, Any]) -> bool:
    semantic = _pose_semantics(kind, adapter, component)
    if semantic is not None:
        return semantic in {"delta", "relative", "velocity"}
    return kind == "action" and str(pose_input).startswith("delta_")


def _pose_semantics(kind: str, adapter: dict[str, Any], component: dict[str, Any]) -> str | None:
    semantic = (
        component.get(f"{kind}_pose_semantics")
        or component.get("pose_semantics")
        or adapter.get(f"{kind}_pose_semantics")
        or adapter.get("pose_semantics")
    )
    if semantic is not None:
        return str(semantic).strip().lower()
    return None


def _transform_xyz_rotvec_pose(pose: torch.Tensor, frame_transform: dict[str, Any], *, is_delta: bool) -> torch.Tensor:
    if not frame_transform:
        return pose
    dtype, device = pose.dtype, pose.device
    world_from_source = _rotation_matrix_from_config(
        frame_transform.get("world_from_source", frame_transform.get("canonical_world_from_source_world")),
        dtype=dtype,
        device=device,
    )
    canonical_ee_from_source_cfg = frame_transform.get(
        "canonical_ee_from_source_ee",
        frame_transform.get("canonical_local_from_source_local"),
    )
    if canonical_ee_from_source_cfg is not None:
        canonical_ee_from_source = _rotation_matrix_from_config(
            canonical_ee_from_source_cfg, dtype=dtype, device=device
        )
        source_ee_from_canonical = canonical_ee_from_source.transpose(-1, -2)
    else:
        source_ee_from_canonical = _rotation_matrix_from_config(
            frame_transform.get(
                "source_ee_from_canonical_ee", frame_transform.get("source_local_from_canonical_local")
            ),
            dtype=dtype,
            device=device,
        )
        canonical_ee_from_source = source_ee_from_canonical.transpose(-1, -2)

    xyz = pose[..., :3]
    rot = _rotvec_to_matrix(pose[..., 3:6])
    if is_delta:
        delta_frame = str(
            frame_transform.get(
                "delta_frame",
                frame_transform.get("action_delta_frame", frame_transform.get("delta_pose_frame", "world")),
            )
        ).lower()
        if delta_frame in {"eef_local", "ee_local", "local", "source_ee", "source_local"}:
            xyz_out = torch.einsum("ij,...j->...i", canonical_ee_from_source, xyz)
            rot_out = canonical_ee_from_source @ rot @ canonical_ee_from_source.transpose(-1, -2)
        else:
            xyz_out = torch.einsum("ij,...j->...i", world_from_source, xyz)
            rot_out = world_from_source @ rot @ world_from_source.transpose(-1, -2)
    else:
        xyz_out = torch.einsum("ij,...j->...i", world_from_source, xyz)
        rot_out = world_from_source @ rot @ source_ee_from_canonical
    return torch.cat([xyz_out, _matrix_to_rotvec(rot_out)], dim=-1)


def _pose_delta_from_transformed_xyz_rotvec(current_pose: torch.Tensor, future_pose: torch.Tensor) -> torch.Tensor:
    current_rot = _rotvec_to_matrix(current_pose[..., 3:6])
    future_rot = _rotvec_to_matrix(future_pose[..., 3:6])
    current_from_world = current_rot.transpose(-1, -2)
    delta_xyz_world = future_pose[..., :3] - current_pose[..., :3]
    delta_xyz_local = torch.einsum("...ij,...j->...i", current_from_world, delta_xyz_world)
    delta_rot = current_from_world @ future_rot
    return torch.cat([delta_xyz_local, _matrix_to_rotvec(delta_rot)], dim=-1)


def _pose_delta_world_from_transformed_xyz_rotvec(
    current_pose: torch.Tensor, future_pose: torch.Tensor
) -> torch.Tensor:
    current_rot = _rotvec_to_matrix(current_pose[..., 3:6])
    future_rot = _rotvec_to_matrix(future_pose[..., 3:6])
    delta_xyz_world = future_pose[..., :3] - current_pose[..., :3]
    delta_rot = current_rot.transpose(-1, -2) @ future_rot
    return torch.cat([delta_xyz_world, _matrix_to_rotvec(delta_rot)], dim=-1)


def _assign_target_indices(
    out: torch.Tensor, dim_mask: torch.Tensor, target_indices: list[Any] | tuple[Any, ...], values: torch.Tensor
) -> None:
    if not isinstance(target_indices, (list, tuple)) or not target_indices:
        raise ValueError(f"target_indices must be a non-empty list, got {target_indices!r}")
    indices = [int(index) for index in target_indices]
    if len(set(indices)) != len(indices):
        raise ValueError(f"target_indices contains duplicate dimensions: {target_indices!r}")
    if min(indices) < 0 or max(indices) >= out.shape[-1]:
        raise ValueError(f"target_indices {target_indices!r} exceed output dimension {out.shape[-1]}")
    values = _as_feature_matrix(values, "target").to(out.dtype)
    if values.shape[-1] != len(indices):
        raise ValueError(f"target_indices {target_indices!r} expects {len(indices)} dims, got {values.shape[-1]}")
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=out.device)
    out.index_copy_(-1, index_tensor, values)
    dim_mask[index_tensor.cpu()] = False


def _adapter_columns(adapter: dict[str, Any] | None) -> list[str]:
    if not adapter:
        return []
    columns: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value and value not in columns:
            columns.append(value)

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.endswith("_key"):
                    add(value)
                else:
                    visit(value)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)

    visit(adapter)
    return columns


def _decode_mp4_path_lerobot(
    path: str | Path,
    timestamps: list[float],
    *,
    tolerance_s: float = 0.02,
    backend: str | None = None,
) -> torch.Tensor:
    if not timestamps:
        return torch.empty(0, 3, 0, 0)

    del backend
    t0 = time.perf_counter()
    import av

    _profile_record("video.import_av", time.perf_counter() - t0)
    t0 = time.perf_counter()
    container = av.open(str(path), mode="r")
    try:
        stream = container.streams.video[0]
        try:
            stream.thread_type = "NONE"
        except Exception:
            pass
        try:
            stream.codec_context.thread_count = 1
        except Exception:
            pass
        first_ts = float(min(timestamps))
        last_ts = float(max(timestamps))
        if stream.time_base is not None:
            seek_offset = int(first_ts / float(stream.time_base))
            container.seek(seek_offset, any_frame=False, backward=True, stream=stream)

        loaded_frames = []
        loaded_ts = []
        fallback_fps = float(stream.average_rate) if stream.average_rate is not None else 30.0
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame.pts is not None and stream.time_base is not None:
                ts = float(frame.pts * stream.time_base)
            elif frame.time is not None:
                ts = float(frame.time)
            else:
                ts = frame_idx / fallback_fps
            if ts + tolerance_s < first_ts:
                continue
            loaded_frames.append(frame.to_ndarray(format="rgb24"))
            loaded_ts.append(ts)
            if ts >= last_ts + tolerance_s:
                break
    finally:
        container.close()

    if not loaded_frames:
        raise RuntimeError(f"PyAV decoded no frames from mp4: {path}")
    loaded_ts_arr = np.asarray(loaded_ts, dtype=np.float32)
    query_ts = np.asarray(timestamps, dtype=np.float32)
    closest = np.abs(query_ts[:, None] - loaded_ts_arr[None]).argmin(axis=1)
    max_err = float(np.abs(query_ts - loaded_ts_arr[closest]).max())
    if max_err > max(tolerance_s, 1.0 / 30.0 + tolerance_s):
        _profile_record("video.timestamp_warning", 0.0, max_err=max_err)
    arr = np.stack([loaded_frames[int(i)] for i in closest], axis=0)
    frames = torch.from_numpy(arr).permute(0, 3, 1, 2).to(torch.float32) / 255.0
    _profile_record(
        "video.decode",
        time.perf_counter() - t0,
        frames=len(timestamps),
        loaded=len(loaded_frames),
        backend="pyav_path",
    )
    del loaded_frames, loaded_ts, loaded_ts_arr, query_ts, closest
    return frames


def _decode_mp4_bytes_lerobot(
    raw: bytes,
    timestamps: list[float],
    *,
    tolerance_s: float = 0.02,
    backend: str | None = None,
) -> torch.Tensor:
    if not timestamps:
        return torch.empty(0, 3, 0, 0)
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        t0 = time.perf_counter()
        f.write(raw)
        f.flush()
        _profile_record("video.tempfile_write", time.perf_counter() - t0, bytes=len(raw))
        return _decode_mp4_path_lerobot(f.name, timestamps, tolerance_s=tolerance_s, backend=backend)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    remote_root: str
    stats_path: str | None = "dataset_stats.json"
    local_stats_path: str | None = None
    local_meta_dir: str | None = None
    path_index_dir: str | None = None
    local_text_embedding_cache_dir: str | None = None
    context_len: int = 128
    text_encoder_id: str = None
    modalities: dict[str, Any] | None = None
    canonical_adapter: dict[str, Any] | None = None
    format_version: int | None = None
    embodiment: str = "default"
    control_schema: str | None = None
    stats_group: str | None = None
    stats_group_fallbacks: tuple[str, ...] = ()
    dataset_weight: float = 1.0
    distribute_weights: bool = False
    action_target_dim: int | None = None
    state_target_dim: int | None = None
    source_name: str | None = None
    group_id: int = 0
    local_root: str | None = None


class PetrelLeRobotDataset(torch.utils.data.Dataset):
    """Map-style reader for one locally mounted LeRobot v2.1/v3.0 dataset."""

    def __init__(
        self,
        spec: DatasetSpec,
        *,
        episodes: list[int] | None = None,
        num_frames: int = 33,
        action_size: int | None = None,
        global_sample_stride: int = 1,
        parquet_cache_size: int = 8,
        path_index: list[dict[str, Any]] | None = None,
        use_path_index_cache: bool = True,
        shared_parquet_cache: Any | None = None,
        local_video_path_decode: bool = True,
    ):
        self.spec = spec
        self.local_video_path_decode = bool(local_video_path_decode)
        self._client: LocalFileClient | None = None
        self.name = spec.name  # petrel/dataset_groups/xxx-dataset/xxx-task.yaml-datasets-name
        self.remote_root = spec.remote_root.rstrip("/") # dataset_groups/xxx-dataset/xxx-task.yaml-(root+datasets/path)
        self.modalities = spec.modalities or {} # /petrel/(dataset_groups/xxx-task.yaml+/control_profiles/xxx-dataset.yaml)-(fields+fields)
        self.canonical_adapter = spec.canonical_adapter or {} # /petrel/(dataset_groups/xxx-task.yaml+/control_profiles/xxx-dataset.yaml)-(mapping+mapping_overrides)
        self.canonical_dim = _canonical_adapter_dim(self.canonical_adapter)
        self.use_canonical_adapter = self.canonical_dim is not None
        # List: dataset_groups/xxx-task.yaml-state.keys
        self.state_keys = _modality_keys(self.modalities, "state", "observation.state")
        # List: dataset_groups/xxx-task.yaml-action.keys
        self.action_keys = _modality_keys(self.modalities, "action", "action")
        self.state_keys_config = list(self.state_keys)
        self.action_keys_config = list(self.action_keys)
        self.state_key = self.state_keys[0]
        self.action_key = self.action_keys[0]
        # str dataset_groups/xxx-task.yaml-timestamp.key
        self.timestamp_key = _modality_key(self.modalities, "timestamp", "timestamp")
        # str dataset_groups/xxx-task.yaml-frame_idex.key
        self.frame_index_key = _modality_key(self.modalities, "frame_index", "frame_index")
        # str dataset_groups/xxx-task.yaml-task_index.key
        self.task_index_key = _modality_key(self.modalities, "task_index", "task_index")
        self.video_key = _modality_key(self.modalities, "video", "")
        # List: dataset_groups/xxx-task.yaml-video.keys
        self.video_keys_config = _modality_keys(self.modalities, "video", "")
        self.num_frames_per_sample = int(num_frames)
        self.action_size = int(action_size if action_size is not None else num_frames - 1)
        self.global_sample_stride = int(global_sample_stride)
        self.parquet_cache_size = max(0, int(parquet_cache_size))
        self._parquet_cache: OrderedDict[int, dict[str, torch.Tensor | list[Any]]] = OrderedDict()
        # This reader is reconstructed independently inside each DataLoader worker.
        # Keep the cache process-local instead of using a multiprocessing proxy.
        del shared_parquet_cache
        self._v3_data_file_range_cache: dict[int, tuple[int, int] | None] = {}
        self._v3_data_file_range_cache_max = 1024
        self.use_path_index_cache = bool(use_path_index_cache)
        self._loaded_path_index_dir: Path | None = None
        self.frame_filter_clean_segments: dict[int, tuple[tuple[int, int], ...]] = {}
        self.frame_filter_path: Path | None = None

        t0 = time.perf_counter()
        # Dict data/petrel_cache/meta/InternData-A1_v30/articulation_tasks/lift2/close_the_microwave_left_arm_part1/info.json
        self.info = self._read_json("meta/info.json")
        self.info = _shape_tuple_inplace(self.info)
        self.format_major = self._detect_format_major()
        available_video_keys = [key for key, ft in self.info.get("features", {}).items() if ft.get("dtype") == "video"]
        configured_video_keys = [key for key in self.video_keys_config if key]
        if configured_video_keys:
            missing_video_keys = [key for key in configured_video_keys if key not in available_video_keys]
            if missing_video_keys:
                raise KeyError(
                    f"Dataset {self.name} configured video keys not found in meta/info.json: "
                    f"missing={missing_video_keys}, available={available_video_keys}"
                )
            self.video_keys = configured_video_keys
        else:
            self.video_keys = available_video_keys
        if not self.video_key and self.video_keys:
            self.video_key = self.video_keys[0]
        self.fps = float(self.info.get("fps", 30.0))
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        self._resolve_feature_keys()
        self.state_dim = _feature_dims(self.info, self.state_keys)  # info state shape count
        self.action_dim = _feature_dims(self.info, self.action_keys)  # info action shape count
        if self.use_canonical_adapter:
            self.state_target_dim = int(self.canonical_dim)  # 80
            self.action_target_dim = int(self.canonical_dim)  # 80
        else:
            self.state_target_dim = int(spec.state_target_dim) if spec.state_target_dim is not None else self.state_dim
            self.action_target_dim = (
                int(spec.action_target_dim) if spec.action_target_dim is not None else self.action_dim
            )
        _profile_record(
            "dataset.read_info",
            time.perf_counter() - t0,
            dataset=self.name,
            version=self.info.get("codebase_version"),
            format_major=self.format_major,
            state_key="+".join(self.state_keys),
            action_key="+".join(self.action_keys),
        )
        t0 = time.perf_counter()
        # Dict data/petrel_cache/meta/InternData-A1_v30/articulation_tasks/lift2/close_the_microwave_left_arm_part1/tasks.parquet
        # {task_index: task_prompt}
        self.tasks = self._read_tasks() 
        self.task_to_index = {task: task_index for task_index, task in self.tasks.items()}
        _profile_record("dataset.read_tasks", time.perf_counter() - t0, dataset=self.name, count=len(self.tasks))
        # which dataset use segment text
        self.segment_action_text_enabled = self._should_use_segment_action_text()   # return true or false
        self.subtask_action_segments: dict[int, tuple[tuple[int, int, str], ...]] = {}

        # openpi/data/petrel_cache/path_index/InternData-A1_v30/articulation_tasks/franka/close_the_electriccooker/path_index.jsonl
        loaded_path_index = path_index if path_index is not None else self._load_local_path_index()
        if loaded_path_index is not None:
            self._apply_path_index(loaded_path_index, episodes=episodes)
            self._load_subtask_action_segments()
            _profile_record("dataset.path_index_hit", 0.0, dataset=self.name)
            return

        t0 = time.perf_counter()
        self.episodes_dict = self._read_episodes()
        _profile_record(
            "dataset.read_episodes", time.perf_counter() - t0, dataset=self.name, count=len(self.episodes_dict)
        )

        path_index = self.export_path_index()
        self._write_local_path_index(path_index)
        self._apply_path_index(path_index, episodes=episodes)
        self._load_subtask_action_segments()

    def _path_index_paths(self) -> list[Path]:
        if not self.use_path_index_cache or not self.spec.path_index_dir:
            return []
        return [path / "path_index.jsonl" for path in _local_path_candidates(self.spec.path_index_dir)]

    def _path_index_path(self) -> Path | None:
        paths = self._path_index_paths()
        return paths[0] if paths else None

    def _path_index_row_is_usable(self, row: dict[str, Any]) -> bool:
        if "episode_index" not in row:
            return False
        if "frame_count" not in row and "length" not in row:
            return False
        if not row.get("data_path"):
            return False
        video_paths = row.get("video_paths")
        if self.video_keys and not isinstance(video_paths, dict):
            return False
        return all(key in video_paths for key in self.video_keys)

    def _load_local_path_index(self) -> list[dict[str, Any]] | None:
        for cache_path in self._path_index_paths():
            if not _is_file_accessible(cache_path):
                continue
            rows: list[dict[str, Any]] = []
            try:
                with cache_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        if not isinstance(row, dict) or not self._path_index_row_is_usable(row):
                            return None
                        rows.append(row)
            except Exception:
                continue
            self._loaded_path_index_dir = cache_path.parent
            return rows or None
        return None

    def _frame_filter_paths(self) -> list[Path]:
        roots: list[Path] = []
        if self._loaded_path_index_dir is not None:
            roots.append(self._loaded_path_index_dir)
        roots.extend(path.parent for path in self._path_index_paths())
        out: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            for name in FRAME_FILTER_FILENAMES:
                path = root / name
                key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                out.append(path)
        return out

    @staticmethod
    def _normalize_clean_segments(raw_segments: Any, episode_len: int) -> tuple[tuple[int, int], ...]:
        length = max(0, int(episode_len))
        if raw_segments in (None, ""):
            return ((0, length),) if length > 0 else tuple()
        if not isinstance(raw_segments, (list, tuple)):
            return ((0, length),) if length > 0 else tuple()

        segments: list[tuple[int, int]] = []
        for segment in raw_segments:
            if not isinstance(segment, (list, tuple)) or len(segment) != 2:
                continue
            start = max(0, min(length, int(segment[0])))
            end = max(0, min(length, int(segment[1])))
            if end > start:
                segments.append((start, end))
        if not segments:
            return tuple()
        segments.sort()

        merged: list[tuple[int, int]] = []
        for start, end in segments:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return tuple(merged)

    def _load_local_frame_filter(self) -> None:
        self.frame_filter_clean_segments = {}
        self.frame_filter_path = None
        for filter_path in self._frame_filter_paths():
            if not _is_file_accessible(filter_path):
                continue
            loaded: dict[int, tuple[tuple[int, int], ...]] = {}
            try:
                with filter_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        if not isinstance(row, dict) or "episode_index" not in row:
                            continue
                        episode_index = int(row["episode_index"])
                        episode = self.episodes_dict.get(episode_index)
                        if episode is None:
                            continue
                        episode_len = int(episode["length"])
                        segments = self._normalize_clean_segments(row.get("clean_segments"), episode_len)
                        loaded[episode_index] = segments
            except Exception:
                continue
            self.frame_filter_clean_segments = loaded
            self.frame_filter_path = filter_path
            _profile_record(
                "dataset.frame_filter_hit",
                0.0,
                dataset=self.name,
                path=str(filter_path),
                episodes=len(loaded),
            )
            return

    def _write_local_path_index(self, rows: list[dict[str, Any]]) -> None:
        for cache_path in self._path_index_paths():
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    dir=cache_path.parent,
                    prefix=cache_path.name + ".tmp.",
                    delete=False,
                ) as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    tmp_path = Path(handle.name)
                tmp_path.replace(cache_path)
                _profile_record(
                    "dataset.path_index_write", 0.0, dataset=self.name, path=str(cache_path), episodes=len(rows)
                )
                return
            except Exception:
                continue

    def _apply_path_index(self, rows: list[dict[str, Any]], *, episodes: list[int] | None = None) -> None:
        episodes_dict: dict[int, dict[str, Any]] = {}
        for row in rows:
            episode_index = int(row["episode_index"])
            item: dict[str, Any] = {
                "episode_index": episode_index,
                "length": int(row.get("frame_count", row.get("length"))),
                "data_path": str(row["data_path"]),
                "video_paths": {str(key): str(value) for key, value in dict(row.get("video_paths") or {}).items()},
            }
            for key in (
                "task_index",
                "data_from_index",
                "data_to_index",
                "data_file_ordinal",
                "start_frame",
                "source_episode_index",
                "source_start_frame",
            ):
                if key in row:
                    item[key] = int(row[key])
            if isinstance(row.get("video_timestamp_offsets"), dict):
                item["video_timestamp_offsets"] = {
                    str(key): float(value) for key, value in row["video_timestamp_offsets"].items()
                }
            episodes_dict[episode_index] = item
        self.episodes_dict = episodes_dict  # Dict {episode_index: items}
        self.selected_episodes = list(episodes) if episodes is not None else sorted(self.episodes_dict)
        self.episode_lengths = [int(self.episodes_dict[ep]["length"]) for ep in self.selected_episodes]
        self.cumulative_lengths = np.cumsum(self.episode_lengths).tolist()
        self.trajectory_ids = np.asarray(self.selected_episodes, dtype=np.int64)  #[0, 1, 2, 3 ...]
        self.trajectory_lengths = np.asarray(self.episode_lengths, dtype=np.int64)  # [episode0_length, episode1_length, ....]
        self.max_delta_index = max(self.num_frames_per_sample - 1, self.action_size - 1) * self.global_sample_stride
        self._load_local_frame_filter()  # TODO

    def source_episode_index(self, episode_index: int) -> int:
        episode = self.episodes_dict.get(int(episode_index))
        if isinstance(episode, dict) and "source_episode_index" in episode:
            return int(episode["source_episode_index"])
        return int(episode_index)

    def episode_start_frame(self, episode_index: int) -> int:
        episode = self.episodes_dict.get(int(episode_index))
        if isinstance(episode, dict):
            return max(0, int(episode.get("start_frame", 0) or 0))
        return 0

    def source_frame_index(self, episode_index: int, frame_idx: int) -> int:
        return self.episode_start_frame(int(episode_index)) + int(frame_idx)

    def source_frame_indices(self, episode_index: int, frame_indices: list[int] | torch.Tensor) -> list[int]:
        start = self.episode_start_frame(int(episode_index))
        if isinstance(frame_indices, torch.Tensor):
            values = frame_indices.detach().to(device="cpu", dtype=torch.long).tolist()
        else:
            values = list(frame_indices)
        return [start + int(value) for value in values]

    def _should_use_segment_action_text(self) -> bool:
        source = str(self.spec.source_name or "").lower()
        root = str(self.remote_root or "").lower()
        name = str(self.name or "").lower()
        return (
            source.startswith("agibotworld")
            or source.startswith("galaxea")
            or "agibotworld" in root
            or "/galaxea/" in root
            or name.startswith("task_")
        )

    def _load_subtask_action_segments(self) -> None:
        self.subtask_action_segments = {}
        if not bool(getattr(self, "segment_action_text_enabled", False)):
            return
        try:
            records = self._read_jsonl("meta/episodes_lang_mem.jsonl")
        except FileNotFoundError:
            return
        except Exception:
            return

        loaded: dict[int, tuple[tuple[int, int, str], ...]] = {}
        for row in records:
            if not isinstance(row, dict) or "episode_index" not in row:
                continue
            segments: list[tuple[int, int, str]] = []
            for segment in row.get("action_config") or []:
                if not isinstance(segment, dict):
                    continue
                text = str(segment.get("action_text") or "").strip()
                if not text:
                    continue
                try:
                    start = int(segment.get("start_frame"))
                    end = int(segment.get("end_frame"))
                except Exception:
                    continue
                if end > start:
                    segments.append((start, end, text))
            if segments:
                segments.sort(key=lambda item: (item[0], item[1]))
                loaded[int(row["episode_index"])] = tuple(segments)
        # self.subtask_action_segments = {
        #     episode_index: (
        #         (start_frame, end_frame, action_text),
        #         ...
        #     )
        # }
        self.subtask_action_segments = loaded
        _profile_record(
            "dataset.subtask_action_segments",
            0.0,
            dataset=self.name,
            episodes=len(loaded),
        )

    def _subtask_segments_for_episode(self, episode_index: int) -> tuple[tuple[int, int, str], ...]:
        if not self.subtask_action_segments:
            return tuple()
        source_episode = self.source_episode_index(int(episode_index))
        return self.subtask_action_segments.get(int(source_episode), tuple())

    def _subtask_for_source_frame(self, episode_index: int, source_frame_idx: int) -> tuple[int, int, str] | None:
        segments = self._subtask_segments_for_episode(int(episode_index))
        if not segments:
            return None
        frame = int(source_frame_idx)
        for start, end, text in segments:
            if start <= frame < end:
                return start, end, text
        if frame >= int(segments[-1][1]):
            return segments[-1]
        return None

    def _valid_start_intervals_for_episode(
        self,
        episode_index: int,
        episode_len: int,
        allow_padding_at_end: bool = False,
        *,
        extra_span: int = 0,
    ) -> tuple[tuple[int, int], ...]:
        episode_len = max(0, int(episode_len))
        if episode_len <= 0:
            return tuple()

        segments = self._subtask_segments_for_episode(int(episode_index))
        if not segments:
            max_start = episode_len - 1 if allow_padding_at_end else episode_len - 1 - int(self.max_delta_index)
            max_start -= max(0, int(extra_span))
            return ((0, int(max_start)),) if max_start >= 0 else tuple()

        source_start = self.episode_start_frame(int(episode_index))
        required_delta = int(self.max_delta_index) + max(0, int(extra_span))
        clean_segments = self.clean_segments_for_episode(int(episode_index), episode_len)  # TODO
        intervals: list[tuple[int, int]] = []
        for seg_start, seg_end, _ in segments:
            subtask_start = max(0, int(seg_start) - source_start)
            subtask_end = min(episode_len, int(seg_end) - source_start)
            for clean_start, clean_end in clean_segments:
                local_start = max(subtask_start, int(clean_start))
                local_end = min(subtask_end, int(clean_end))
                max_start = local_end - 1 - required_delta
                if max_start >= local_start:
                    intervals.append((local_start, max_start))
        return tuple(intervals)

    def sample_valid_start_for_trajectory_pos(
        self,
        trajectory_pos: int,
        rng: np.random.Generator,
        allow_padding_at_end: bool = False,
        *,
        extra_span: int = 0,
    ) -> int | None:
        episode_index = int(self.trajectory_ids[int(trajectory_pos)])
        episode_len = int(self.trajectory_lengths[int(trajectory_pos)])
        intervals = self._valid_start_intervals_for_episode(
            episode_index,
            episode_len,
            allow_padding_at_end,
            extra_span=extra_span,
        )
        if not intervals:
            return None
        counts = np.asarray([end - start + 1 for start, end in intervals], dtype=np.float64)
        seg_idx = int(rng.choice(len(intervals), p=counts / counts.sum()))
        start, end = intervals[seg_idx]
        return int(rng.integers(int(start), int(end) + 1))

    def clean_segments_for_episode(
        self,
        episode_index: int,
        episode_len: int | None = None,
    ) -> tuple[tuple[int, int], ...]:
        episode_index = int(episode_index)
        if episode_len is None:
            episode_len = int(self.episodes_dict[episode_index]["length"])
        segments = self.frame_filter_clean_segments.get(episode_index)
        if segments is None:
            length = max(0, int(episode_len))
            return ((0, length),) if length > 0 else tuple()
        return segments

    def clean_frame_mask(
        self,
        episode_index: int,
        frame_indices: list[int] | torch.Tensor,
        *,
        episode_len: int | None = None,
    ) -> torch.Tensor:
        if isinstance(frame_indices, torch.Tensor):
            indices = frame_indices.detach().to(device="cpu", dtype=torch.long)
        else:
            indices = torch.as_tensor(frame_indices, dtype=torch.long)
        mask = torch.zeros(indices.shape, dtype=torch.bool)
        for start, end in self.clean_segments_for_episode(episode_index, episode_len):
            mask |= (indices >= int(start)) & (indices < int(end))
        return mask

    def _episode_task_index_from_metadata(self, episode: dict[str, Any]) -> int | None:
        for key in ("task_index", "tasks_index", "task_indices"):
            value = self._scalar(episode.get(key))
            if value is not None:
                return int(value)

        tasks = episode.get("tasks")
        task: str | None = None
        if isinstance(tasks, str):
            task = tasks
        elif isinstance(tasks, (list, tuple)) and tasks:
            task = str(tasks[0])
        if task is None:
            return None
        return int(self.task_to_index[task]) if task in self.task_to_index else None

    def _path_index_row(self, episode_index: int, episode: dict[str, Any]) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "episode_index": int(episode_index),
            "frame_count": int(episode["length"]),
            "start_frame": 0,
            "data_path": self._remote_path(self._data_rel_path(int(episode_index))),
            "video_paths": {
                key: self._remote_path(self._video_rel_path(int(episode_index), key)) for key in self.video_keys
            },
        }
        task_index = self._episode_task_index_from_metadata(episode)
        if task_index is not None:
            entry["task_index"] = task_index
        if self.is_v3:
            bounds = self._v3_episode_data_bounds(int(episode_index))
            if bounds is not None:
                entry["data_from_index"] = int(bounds[0])
                entry["data_to_index"] = int(bounds[1])
            entry["data_file_ordinal"] = int(self._v3_data_file_ordinal_from_episode(int(episode_index)))
            video_offsets: dict[str, float] = {}
            for key in self.video_keys:
                offset = self._video_timestamp_offset(int(episode_index), key)
                if offset:
                    video_offsets[key] = float(offset)
            if video_offsets:
                entry["video_timestamp_offsets"] = video_offsets
        return entry

    def export_path_index(self) -> list[dict[str, Any]]:
        return [self._path_index_row(int(key), self.episodes_dict[key]) for key in sorted(self.episodes_dict)]

    @property
    def client(self) -> LocalFileClient:
        if self._client is None:
            if not self.spec.local_root:
                raise ValueError(f"Local dataset {self.name!r} is missing local_root.")
            self._client = LocalFileClient(self.spec.local_root)
        return self._client

    @property
    def is_v3(self) -> bool:
        return self.format_major >= 3

    def _detect_format_major(self) -> int:
        if self.spec.format_version is not None:
            return int(self.spec.format_version)
        version = str(self.info.get("codebase_version", "v2.1")).strip().lower().lstrip("v")
        try:
            return int(version.split(".", 1)[0])
        except ValueError:
            return 2

    def set_target_dims(self, *, action_target_dim: int | None = None, state_target_dim: int | None = None) -> None:
        if action_target_dim is not None:
            self.action_target_dim = int(action_target_dim)
        if state_target_dim is not None:
            self.state_target_dim = int(state_target_dim)

    def _feature_sort_key(self, key: str) -> tuple[int, int, str]:
        lowered = key.lower()
        if ".left_" in lowered or lowered.endswith(".left") or "left" in lowered:
            side = 0
        elif ".right_" in lowered or lowered.endswith(".right") or "right" in lowered:
            side = 1
        else:
            side = 2
        if "joint" in lowered or "qpos" in lowered or "arm" in lowered:
            kind = 0
        elif "end_effector" in lowered or "eef" in lowered or ".ee" in lowered:
            kind = 1
        elif "gripper" in lowered:
            kind = 2
        else:
            kind = 3
        return side, kind, lowered

    def _infer_vector_keys(self, name: str, candidates: tuple[str, ...]) -> list[str]:
        features = self.info.get("features", {})
        for key in candidates:
            if key in features and _is_numeric_feature(self.info, key):
                return [key]

        skip_tokens = ("timestamp", "frame_index", "episode_index", "task_index", "index")
        keys: list[str] = []
        if name == "action":
            prefixes = ("action", "actions")
        else:
            prefixes = ("observation.state", "observation.states", "states", "state")

        for key in features:
            lowered = key.lower()
            if any(token == lowered or lowered.endswith("." + token) for token in skip_tokens):
                continue
            if not _is_numeric_feature(self.info, key):
                continue
            if any(lowered == prefix or lowered.startswith(prefix + ".") for prefix in prefixes):
                keys.append(key)

        if keys:
            return sorted(keys, key=self._feature_sort_key)

        lowered_name = name.lower()
        for key in features:
            lower_key = key.lower()
            if lowered_name in lower_key and _is_numeric_feature(self.info, key):
                return [key]
        return []

    def _resolve_feature_key(self, name: str, current: str, candidates: tuple[str, ...]) -> str:
        features = self.info.get("features", {})
        if current in features:
            return current
        if not _modality_is_auto(self.modalities, name) and current not in ("", "auto"):
            return current
        inferred = self._infer_vector_keys(name, candidates)
        return inferred[0] if inferred else current

    def _resolve_vector_keys(self, name: str, current: list[str], candidates: tuple[str, ...]) -> list[str]:
        features = self.info.get("features", {})
        if current and all(key in features for key in current):
            return current
        if not _modality_is_auto(self.modalities, name) and current and current != [""]:
            return current
        inferred = self._infer_vector_keys(name, candidates)
        return inferred or current

    def _resolve_feature_keys(self) -> None:
        self.state_keys = self._resolve_vector_keys(
            "state",
            self.state_keys,
            (
                "observation.state",
                "observation.states",
                "observation.robot_state",
                "observation.joint_state",
                "observation.ee_state",
            ),
        )
        self.action_keys = self._resolve_vector_keys("action", self.action_keys, ("action", "actions"))
        self.state_key = self.state_keys[0]
        self.action_key = self.action_keys[0]
        self.timestamp_key = self._resolve_feature_key("timestamp", self.timestamp_key, ("timestamp",))
        self.frame_index_key = self._resolve_feature_key("frame_index", self.frame_index_key, ("frame_index",))
        self.task_index_key = self._resolve_feature_key("task_index", self.task_index_key, ("task_index",))

    def _remote_path(self, rel_or_abs_path: str) -> str:
        text = str(rel_or_abs_path)
        if _is_absolute_dataset_path(text):
            return text
        return petrel_join(self.remote_root, text)

    def _local_meta_paths(self, rel_path: str) -> list[Path]:
        if not self.spec.local_meta_dir or not str(rel_path).startswith("meta/"):
            return []
        suffix = str(rel_path)[len("meta/") :]
        return [path / suffix for path in _local_path_candidates(self.spec.local_meta_dir)]

    def _local_meta_path(self, rel_path: str) -> Path | None:
        paths = self._local_meta_paths(rel_path)
        return paths[0] if paths else None

    def _read_bytes(self, rel_path: str) -> bytes:
        local_paths = self._local_meta_paths(rel_path)
        for local_path in local_paths:
            if not _is_file_accessible(local_path):
                continue
            t0 = time.perf_counter()
            raw = local_path.read_bytes()
            _profile_record(
                _io_profile_name("metadata_cache", local_path),
                time.perf_counter() - t0,
                bytes=len(raw),
                path=local_path.name,
            )
            return raw

        raw = self.client.get_bytes(self._remote_path(rel_path))
        for local_path in local_paths:
            t0 = time.perf_counter()
            tmp_path: Path | None = None
            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    "wb",
                    dir=local_path.parent,
                    prefix=local_path.name + ".tmp.",
                    delete=False,
                ) as handle:
                    handle.write(raw)
                    tmp_path = Path(handle.name)
                tmp_path.replace(local_path)
                _profile_record(
                    "local_meta.write_bytes", time.perf_counter() - t0, bytes=len(raw), path=local_path.name
                )
                break
            except Exception:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)
                continue
        return raw

    def _read_json(self, rel_path: str) -> dict[str, Any]:
        return _json_loads(self._read_bytes(rel_path))

    def load_dataset_stats(self) -> dict[str, Any]:
        local_stats_path = Path(self.spec.local_stats_path).expanduser() if self.spec.local_stats_path else None
        local_stats_paths = _local_path_candidates(local_stats_path)
        for candidate in local_stats_paths:
            if _is_file_accessible(candidate):
                return json.loads(candidate.read_text(encoding="utf-8"))

        if not self.spec.stats_path:
            if self.is_v3:
                stats = _json_loads(self._read_bytes("meta/stats.json"))
            else:
                raise FileNotFoundError(f"dataset {self.name} has no stats_path or local_stats_path configured")
        else:
            stats_path = str(self.spec.stats_path)
            if "://" not in stats_path:
                local_candidate = None
                for candidate in _local_path_candidates(stats_path):
                    if _is_file_accessible(candidate):
                        local_candidate = candidate
                        break
                if local_candidate is not None:
                    stats = json.loads(local_candidate.read_text(encoding="utf-8"))
                else:
                    stats = _json_loads(self._read_bytes(stats_path))
            else:
                stats = _json_loads(self._read_bytes(stats_path))

        for candidate in local_stats_paths:
            try:
                candidate.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = candidate.with_suffix(candidate.suffix + ".tmp")
                tmp_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                tmp_path.replace(candidate)
                break
            except Exception:
                continue
        return stats

    def _read_jsonl(self, rel_path: str) -> list[dict[str, Any]]:
        return _jsonl_loads(self._read_bytes(rel_path))

    @staticmethod
    def _task_text_from_metadata(item: dict[str, Any]) -> str | None:
        def normalize(value: Any) -> str | None:
            if value is None:
                return None
            if isinstance(value, float) and math.isnan(value):
                return None
            if isinstance(value, (list, tuple)):
                for child in value:
                    text = normalize(child)
                    if text:
                        return text
                return None
            if isinstance(value, dict):
                for key in ("task", "instruction", "language_instruction", "prompt", "description", "text"):
                    text = normalize(value.get(key))
                    if text:
                        return text
                return None
            text = str(value).strip()
            if not text or text.lower() in {"none", "null", "nan"}:
                return None
            return text

        for key in (
            "task",
            "language_instruction",
            "instruction",
            "prompt",
            "description",
            "text",
            "__index_level_0__",
        ):
            text = normalize(item.get(key))
            if text:
                return text
        return None

    def _fallback_task_text(self) -> str:
        return ""

    def _read_tasks(self) -> dict[int, str]:
        if self.is_v3:
            try:
                raw = self._read_bytes(V3_TASKS_PATH)
            except FileNotFoundError:
                records = self._read_jsonl("meta/tasks.jsonl")
            else:
                t0 = time.perf_counter()
                table = pq.read_table(BytesIO(raw))
                records = table.to_pylist()
                _profile_record("tasks.read_parquet", time.perf_counter() - t0, records=len(records), bytes=len(raw))
        else:
            records = self._read_jsonl("meta/tasks.jsonl")

        t0 = time.perf_counter()
        out: dict[int, str] = {}
        for item in records:
            if "task_index" not in item:
                raise KeyError(f"task metadata row missing task_index: keys={list(item)}")
            task = self._task_text_from_metadata(item)
            task_index = int(item["task_index"])
            if task is None:
                task = self._fallback_task_text()
            out[task_index] = task
        out = dict(sorted(out.items()))
        _profile_record("tasks.sort_index", time.perf_counter() - t0, records=len(records))
        return out

    def _text_embedding_cache_paths(self, prompt: str) -> list[Path]:
        if not self.spec.local_text_embedding_cache_dir:
            raise ValueError(f"dataset {self.name} has no local_text_embedding_cache_dir configured")
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        filename = f"{hashed}.t5_len{int(self.spec.context_len)}.{self.spec.text_encoder_id}.pt"
        return [path / filename for path in _local_path_candidates(self.spec.local_text_embedding_cache_dir)]

    def _text_embedding_cache_path(self, prompt: str) -> Path:
        paths = self._text_embedding_cache_paths(prompt)
        for path in paths:
            if _is_file_accessible(path):
                return path
        return paths[0]

    def iter_text_tasks_for_cache(self) -> list[str]:
        tasks: list[str] = []
        seen: set[str] = set()
        for task in self.tasks.values():
            text = str(task).strip()
            if text and text not in seen:
                seen.add(text)
                tasks.append(text)
        for segments in self.subtask_action_segments.values():
            for _, _, task in segments:
                text = str(task).strip()
                if text and text not in seen:
                    seen.add(text)
                    tasks.append(text)
        return tasks

    def _get_cached_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        cache_path = self._text_embedding_cache_path(prompt)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first or point local_text_embedding_cache_dir to the right local cache."
            )
        t0 = time.perf_counter()
        payload = torch.load(cache_path, map_location="cpu")
        _profile_record("text_cache.torch_load", time.perf_counter() - t0, path=cache_path.name)
        context = payload["context"].clone()
        context_mask = payload["mask"].bool().clone()
        if context.ndim != 2:
            raise ValueError(f"Cached `context` must be 2D [L, D], got {tuple(context.shape)} in {cache_path}")
        if context_mask.ndim != 1:
            raise ValueError(f"Cached `mask` must be 1D [L], got {tuple(context_mask.shape)} in {cache_path}")
        if context.shape[0] != int(self.spec.context_len):
            raise ValueError(
                f"Cached context_len mismatch: expected {self.spec.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != int(self.spec.context_len):
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.spec.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )
        context[~context_mask] = 0.0
        return context, context_mask

    def _read_episodes(self) -> dict[int, dict[str, Any]]:
        if self.is_v3:
            records = self._read_v3_episode_records()
        else:
            records = self._read_jsonl("meta/episodes.jsonl")
        t0 = time.perf_counter()
        out = {int(item["episode_index"]): item for item in sorted(records, key=lambda x: int(x["episode_index"]))}
        _profile_record("episodes.sort_index", time.perf_counter() - t0, records=len(records))
        return out

    def _read_v3_episode_records(self) -> list[dict[str, Any]]:
        target = int(self.info.get("total_episodes") or 0)
        template = str(self.info.get("episodes_path") or V3_EPISODES_PATH)
        records: list[dict[str, Any]] = []
        file_ordinal = 0
        while True:
            if target > 0 and len(records) >= target:
                break
            if target > 0 and file_ordinal >= target:
                raise FileNotFoundError(
                    f"Could not collect all v3 episode metadata for {self.name}: "
                    f"got {len(records)}/{target} records before scanning {file_ordinal} files"
                )
            chunk_index = file_ordinal // self.chunks_size
            file_index = file_ordinal % self.chunks_size
            rel_path = template.format(chunk_index=chunk_index, file_index=file_index, episode_chunk=chunk_index)
            try:
                raw = self._read_bytes(rel_path)
            except FileNotFoundError:
                if records and target == 0:
                    break
                if records and target > 0 and len(records) >= target:
                    break
                raise
            t0 = time.perf_counter()
            table = pq.read_table(BytesIO(raw))
            chunk_records = table.to_pylist()
            records.extend(chunk_records)
            _profile_record(
                "episodes.read_parquet",
                time.perf_counter() - t0,
                records=len(chunk_records),
                bytes=len(raw),
                file=rel_path.rsplit("/", 1)[-1],
            )
            file_ordinal += 1
        return records[:target] if target > 0 else records

    @staticmethod
    def _scalar(value: Any, default: Any = None) -> Any:
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        if isinstance(value, np.ndarray):
            return value.reshape(-1)[0].item() if value.size else default
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value

    def _episode_chunk(self, episode_index: int) -> int:
        return int(episode_index) // self.chunks_size

    def _v3_data_rel_path_for_file_ordinal(self, episode_index: int, file_ordinal: int) -> str:
        template = self.info["data_path"]
        file_ordinal = max(0, int(file_ordinal))
        chunk_index = file_ordinal // self.chunks_size
        file_index = file_ordinal % self.chunks_size
        return template.format(
            chunk_index=chunk_index,
            file_index=file_index,
            episode_chunk=chunk_index,
            episode_index=episode_index,
        )

    def _v3_episode_data_bounds(self, episode_index: int) -> tuple[int, int] | None:
        episode = self.episodes_dict[int(episode_index)]
        if "data_from_index" in episode and "data_to_index" in episode:
            return int(episode["data_from_index"]), int(episode["data_to_index"])
        start = self._scalar(
            episode.get("dataset_from_index"), episode.get("data/from_index", episode.get("data/from"))
        )
        end = self._scalar(episode.get("dataset_to_index"), episode.get("data/to_index", episode.get("data/to")))
        if start is None or end is None:
            return None
        return int(start), int(end)

    def _cache_v3_data_file_range(self, file_ordinal: int, value: tuple[int, int] | None) -> None:
        self._v3_data_file_range_cache[int(file_ordinal)] = value
        while len(self._v3_data_file_range_cache) > self._v3_data_file_range_cache_max:
            self._v3_data_file_range_cache.pop(next(iter(self._v3_data_file_range_cache)))

    def _v3_data_file_ordinal_from_episode(self, episode_index: int) -> int:
        episode = self.episodes_dict[int(episode_index)]
        if "data_file_ordinal" in episode:
            return int(episode["data_file_ordinal"])
        chunk_index = int(self._scalar(episode.get("data/chunk_index"), 0))
        file_index = int(self._scalar(episode.get("data/file_index"), 0))
        return chunk_index * self.chunks_size + file_index

    def _v3_data_file_index_range(self, episode_index: int, file_ordinal: int) -> tuple[int, int] | None:
        file_ordinal = int(file_ordinal)
        if file_ordinal in self._v3_data_file_range_cache:
            return self._v3_data_file_range_cache[file_ordinal]
        rel_path = self._v3_data_rel_path_for_file_ordinal(episode_index, file_ordinal)
        try:
            raw = self._read_bytes(rel_path)
        except FileNotFoundError:
            self._cache_v3_data_file_range(file_ordinal, None)
            return None
        parquet_file = pq.ParquetFile(BytesIO(raw))
        if "index" not in parquet_file.schema_arrow.names:
            self._cache_v3_data_file_range(file_ordinal, None)
            return None
        table = parquet_file.read(columns=["index"])
        if table.num_rows == 0:
            self._cache_v3_data_file_range(file_ordinal, None)
            return None
        values = table["index"].to_numpy(zero_copy_only=False)
        out = (int(values.min()), int(values.max()) + 1)
        self._cache_v3_data_file_range(file_ordinal, out)
        return out

    def _v3_find_data_rel_path_by_index_range(self, episode_index: int, *, max_scan_files: int = 128) -> str | None:
        bounds = self._v3_episode_data_bounds(episode_index)
        if bounds is None:
            return None
        start, end = bounds
        base_ordinal = self._v3_data_file_ordinal_from_episode(episode_index)
        for file_ordinal in range(base_ordinal, base_ordinal + max_scan_files):
            file_range = self._v3_data_file_index_range(episode_index, file_ordinal)
            if file_range is None:
                if file_ordinal > base_ordinal:
                    break
                continue
            low, high = file_range
            if high <= start:
                continue
            if low >= end:
                break
            return self._v3_data_rel_path_for_file_ordinal(episode_index, file_ordinal)
        return None

    def _data_rel_path(self, episode_index: int) -> str:
        episode = self.episodes_dict.get(int(episode_index))
        if isinstance(episode, dict):
            if episode.get("data_path"):
                return str(episode["data_path"])
            if episode.get("data_rel_path"):
                return str(episode["data_rel_path"])
        template = self.info["data_path"]
        if self.is_v3:
            if episode is None:
                raise KeyError(f"Missing episode metadata for {episode_index}")
            chunk_index = int(self._scalar(episode.get("data/chunk_index"), 0))
            file_index = int(self._scalar(episode.get("data/file_index"), 0))
            return template.format(
                chunk_index=chunk_index,
                file_index=file_index,
                episode_chunk=chunk_index,
                episode_index=episode_index,
            )
        return template.format(episode_chunk=self._episode_chunk(episode_index), episode_index=episode_index)

    def _video_rel_path(self, episode_index: int, video_key: str) -> str:
        episode = self.episodes_dict.get(int(episode_index))
        if isinstance(episode, dict):
            paths = episode.get("video_paths")
            if isinstance(paths, dict) and video_key in paths:
                return str(paths[video_key])
            rel_paths = episode.get("video_rel_paths")
            if isinstance(rel_paths, dict) and video_key in rel_paths:
                return str(rel_paths[video_key])
        template = self.info["video_path"]
        if self.is_v3:
            if episode is None:
                raise KeyError(f"Missing episode metadata for {episode_index}")
            chunk_index = int(
                self._scalar(episode.get(f"videos/{video_key}/chunk_index"), episode.get("video/chunk_index", 0))
            )
            file_index = int(
                self._scalar(episode.get(f"videos/{video_key}/file_index"), episode.get("video/file_index", 0))
            )
            return template.format(
                chunk_index=chunk_index,
                file_index=file_index,
                episode_chunk=chunk_index,
                episode_index=episode_index,
                video_key=video_key,
            )
        return template.format(
            episode_chunk=self._episode_chunk(episode_index),
            episode_index=episode_index,
            video_key=video_key,
        )

    def _needed_parquet_columns(self) -> list[str]:
        columns = [
            *self.state_keys,
            *self.action_keys,
            *_adapter_columns(self.canonical_adapter),
            self.timestamp_key,
            self.frame_index_key,
            self.task_index_key,
        ]
        if self.is_v3:
            columns.extend(["episode_index", "index"])
        out: list[str] = []
        for col in columns:
            if col and col not in out:
                out.append(col)
        return out

    def _filter_episode_table(self, table: Any, episode_index: int) -> Any:
        if not self.is_v3:
            return table

        source_episode_index = self.source_episode_index(int(episode_index))
        bounds = self._v3_episode_data_bounds(int(episode_index))
        if bounds is not None and "index" in table.column_names:
            start, end = bounds
            mask = pc.and_(pc.greater_equal(table["index"], int(start)), pc.less(table["index"], int(end)))
            filtered = table.filter(mask)
            if filtered.num_rows > 0:
                return filtered

        if "episode_index" in table.column_names:
            return table.filter(pc.equal(table["episode_index"], int(source_episode_index)))
        return table

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_client"] = None
        state["_parquet_cache"] = OrderedDict()
        return state

    def _load_episode_parquet(self, episode_index: int) -> dict[str, torch.Tensor | list[Any]]:
        episode_index = int(episode_index)
        if self.parquet_cache_size > 0:
            cached = self._parquet_cache.get(episode_index)
            if cached is not None:
                self._parquet_cache.move_to_end(episode_index)
                _profile_record("parquet.cache_hit", 0.0, dataset=self.name, episode=episode_index)
                return cached

        _profile_record("parquet.cache_miss", 0.0, dataset=self.name, episode=episode_index)
        data = self._load_episode_parquet_uncached(episode_index)
        if self.parquet_cache_size > 0:
            self._parquet_cache[episode_index] = data
            self._parquet_cache.move_to_end(episode_index)
            while len(self._parquet_cache) > self.parquet_cache_size:
                evicted_episode, _ = self._parquet_cache.popitem(last=False)
                _profile_record("parquet.cache_evict", 0.0, dataset=self.name, episode=evicted_episode)
        return data

    def _load_episode_parquet_uncached(self, episode_index: int) -> dict[str, torch.Tensor | list[Any]]:
        rel_path = self._data_rel_path(episode_index)

        def read_filtered_table(path: str) -> tuple[Any, int, int]:
            raw = self._read_bytes(path)
            t0 = time.perf_counter()
            parquet_file = pq.ParquetFile(BytesIO(raw))
            available_columns = set(parquet_file.schema_arrow.names)
            columns = [col for col in self._needed_parquet_columns() if col in available_columns]
            _profile_record(
                "parquet.open",
                time.perf_counter() - t0,
                episode=episode_index,
                bytes=len(raw),
                columns=len(columns),
                path=path,
            )
            t0 = time.perf_counter()
            table = parquet_file.read(columns=columns)
            table = self._filter_episode_table(table, int(episode_index))
            _profile_record(
                "parquet.read_table",
                time.perf_counter() - t0,
                episode=episode_index,
                bytes=len(raw),
                columns=len(columns),
                rows=table.num_rows,
                path=path,
            )
            return table, len(raw), len(columns)

        table, _, _ = read_filtered_table(rel_path)
        if table.num_rows == 0 and self.is_v3:
            resolved_rel_path = self._v3_find_data_rel_path_by_index_range(int(episode_index))
            if resolved_rel_path and resolved_rel_path != rel_path:
                rel_path = resolved_rel_path
                table, _, _ = read_filtered_table(rel_path)

        if table.num_rows == 0:
            raise RuntimeError(f"No rows for episode {episode_index} in {rel_path}")
        t0 = time.perf_counter()
        data = _table_to_tensors(table)
        _profile_record("parquet.to_tensors", time.perf_counter() - t0, episode=episode_index, rows=table.num_rows)
        return data

    def _load_episode_video_bytes(self, episode_index: int, video_key: str) -> bytes:
        rel_path = self._video_rel_path(episode_index, video_key)
        raw = self._read_bytes(rel_path)
        _profile_record("video.files_loaded", 0.0, episode=episode_index, bytes=len(raw), key=video_key)
        return raw

    def decode_episode_video(self, episode_index: int, video_key: str, timestamps: list[float]) -> torch.Tensor:
        """Decode requested frames, using direct path seek for the local backend."""
        rel_path = self._video_rel_path(episode_index, video_key)
        if self.local_video_path_decode:
            client = self.client
            path = client.resolve_path(self._remote_path(rel_path))
            started = time.perf_counter()
            try:
                frames = _decode_mp4_path_lerobot(path, timestamps)
            except Exception as exc:
                _profile_record(
                    "video.local_path_decode_fallback",
                    time.perf_counter() - started,
                    episode=episode_index,
                    frames=len(timestamps),
                    key=video_key,
                    path=path.name,
                )
                logging.warning("Direct local video decode failed for %s; falling back to bytes: %s", path, exc)
            else:
                _profile_record(
                    "video.local_path_decode_total",
                    time.perf_counter() - started,
                    episode=episode_index,
                    frames=len(timestamps),
                    key=video_key,
                    path=path.name,
                )
                return frames
        raw = self._load_episode_video_bytes(episode_index, video_key)
        return _decode_mp4_bytes_lerobot(raw, timestamps)

    def _episode_task(
        self, episode_index: int, frame_idx: int, parquet: dict[str, torch.Tensor | list[Any]]
    ) -> tuple[int, str]:
        subtask = self._subtask_for_source_frame(int(episode_index), int(frame_idx))
        if subtask is not None:
            _, _, task = subtask
            return int(self.task_to_index.get(task, -1)), task

        if self.task_index_key in parquet:
            task_index = int(_select_rows(parquet, self.task_index_key, [frame_idx])[0].item())
            return task_index, self.tasks[task_index]

        episode = self.episodes_dict[int(episode_index)]
        value = episode.get("task_index")
        if value is not None:
            task_index = int(self._scalar(value))
            return task_index, self.tasks[task_index]
        task = episode.get("task")
        if isinstance(task, str):
            return int(self.task_to_index.get(task, -1)), task
        for key in ("tasks_index", "task_indices"):
            value = self._scalar(episode.get(key))
            if value is not None:
                task_index = int(value)
                return task_index, self.tasks[task_index]

        tasks = episode.get("tasks")
        if isinstance(tasks, str):
            return int(self.task_to_index.get(tasks, -1)), tasks
        if isinstance(tasks, (list, tuple)) and tasks:
            task = str(tasks[0])
            return int(self.task_to_index.get(task, -1)), task
        raise KeyError(f"Could not resolve task for episode {episode_index}")

    def _video_timestamp_offset(self, episode_index: int, video_key: str) -> float:
        if not self.is_v3:
            return 0.0
        episode = self.episodes_dict[int(episode_index)]
        offsets = episode.get("video_timestamp_offsets")
        if isinstance(offsets, dict) and video_key in offsets:
            return float(offsets[video_key])
        for key in (f"videos/{video_key}/from_timestamp", f"videos/{video_key}/start_timestamp", "from_timestamp"):
            value = self._scalar(episode.get(key))
            if value is not None:
                return float(value)
        return 0.0

    def _global_to_episode_frame(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"index {idx} out of bounds for dataset length {len(self)}")
        pos = bisect.bisect_right(self.cumulative_lengths, idx)
        start = 0 if pos == 0 else self.cumulative_lengths[pos - 1]
        return self.selected_episodes[pos], idx - start

    def _query_indices(self, frame_idx: int, episode_len: int, horizon: int) -> tuple[list[int], torch.Tensor]:
        raw = [frame_idx + i * self.global_sample_stride for i in range(horizon)]
        clamped = [max(0, min(episode_len - 1, i)) for i in raw]
        is_pad = torch.as_tensor([i < 0 or i >= episode_len for i in raw], dtype=torch.bool)
        return clamped, is_pad

    def _query_indices_batch(
        self,
        starts: torch.Tensor,
        episode_len: int,
        horizon: int,
    ) -> tuple[list[int], torch.Tensor]:
        offsets = torch.arange(int(horizon), dtype=torch.long) * int(self.global_sample_stride)
        raw = starts.to(torch.long)[:, None] + offsets[None, :]
        is_pad = (raw < 0) | (raw >= int(episode_len))
        clamped = raw.clamp(min=0, max=max(0, int(episode_len) - 1))
        return clamped.reshape(-1).tolist(), is_pad

    def _canonical_component_tensor(
        self,
        parquet: dict[str, torch.Tensor | list[Any]],
        component: dict[str, Any],
        indices: list[int],
        kind: str,
    ) -> torch.Tensor | None:
        key = component.get(f"{kind}_key") or component.get("key")
        if not key:
            return None
        return _select_feature(parquet, str(key), indices, _component_raw_slice(component, kind))

    def _canonical_pose_input(self, component: dict[str, Any], kind: str) -> str:
        return str(
            component.get(f"{kind}_pose_input")
            or component.get("pose_input")
            or self.canonical_adapter.get(f"{kind}_pose_input")
            or (self.canonical_adapter.get("action_pose_input") if kind == "action" else None)
            or self.canonical_adapter.get("pose_input")
            or "xyz_rotvec"
        )

    def _canonical_frame_transform(self, component: dict[str, Any], kind: str) -> dict[str, Any]:
        return _merge_frame_transform(self.canonical_adapter, component, kind)

    def _canonical_transform_pose(   # 改变坐标轴的含义 统一不同数据集的运动方向
        self,
        pose: torch.Tensor,
        component: dict[str, Any],
        kind: str,
        pose_input: str,
        *,
        is_delta: bool | None = None,
    ) -> torch.Tensor:
        if is_delta is None:
            is_delta = _pose_is_delta(kind, pose_input, self.canonical_adapter, component)
        return _transform_xyz_rotvec_pose(
            pose,
            self._canonical_frame_transform(component, kind),
            is_delta=bool(is_delta),
        )

    def _canonical_absolute_pose_from_component(
        self,
        parquet: dict[str, torch.Tensor | list[Any]],
        component: dict[str, Any],
        indices: list[int],
        kind: str,
    ) -> torch.Tensor | None:
        pose_input = self._canonical_pose_input(component, kind)
        direct = self._canonical_component_tensor(parquet, component, indices, kind)
        if direct is not None:
            pose = _pose_to_xyz_rotvec(direct, pose_input)
            return self._canonical_transform_pose(pose, component, kind, pose_input, is_delta=False)

        pos_key = component.get(f"{kind}_position_key") or component.get("position_key")
        rot_key = (
            component.get(f"{kind}_rotation_key")
            or component.get(f"{kind}_orientation_key")
            or component.get("rotation_key")
            or component.get("orientation_key")
        )
        if not pos_key or not rot_key:
            return None

        pos_slice = component.get(f"{kind}_position_slice") or component.get("position_slice")
        rot_slice = (
            component.get(f"{kind}_rotation_slice")
            or component.get(f"{kind}_orientation_slice")
            or component.get("rotation_slice")
            or component.get("orientation_slice")
        )
        pos = _select_feature(parquet, str(pos_key), indices, pos_slice)
        rot = _select_feature(parquet, str(rot_key), indices, rot_slice)
        if pos.shape[-1] != 3:
            raise ValueError(f"Canonical ee position expects 3 dims from {pos_key!r}, got {pos.shape[-1]}")
        if rot.shape[-1] == 4:
            rotvec = _quat_to_rotvec_xyzw(rot) if "xyzw" in pose_input else _quat_to_rotvec_wxyz(rot)
        elif rot.shape[-1] == 3:
            rotvec = _rpy_to_rotvec(rot)
        else:
            raise ValueError(f"Canonical ee rotation expects 3 or 4 dims from {rot_key!r}, got {rot.shape[-1]}")
        pose = torch.cat([pos, rotvec], dim=-1)
        return self._canonical_transform_pose(pose, component, kind, pose_input, is_delta=False)

    def _canonical_pose_from_component(
        self,
        parquet: dict[str, torch.Tensor | list[Any]],
        component: dict[str, Any],
        indices: list[int],
        kind: str,
        next_indices: list[int] | None = None,
        reference_indices: list[int] | None = None,
    ) -> torch.Tensor | None:
        pose_input = self._canonical_pose_input(component, kind)  # mapping pose input
        selected_indices = indices
        if kind == "action" and component.get("action_from_next_action"):
            if next_indices is None:
                raise ValueError(f"canonical_adapter for {self.name} uses action_from_next_action without next_indices")
            selected_indices = next_indices

        if kind == "action":
            if component.get("action_from_state_delta"):
                if next_indices is None:
                    raise ValueError(
                        f"canonical_adapter for {self.name} uses action_from_state_delta without next_indices"
                    )
                current_pose = self._canonical_absolute_pose_from_component(parquet, component, indices, "state")
                future_pose = self._canonical_absolute_pose_from_component(parquet, component, next_indices, "state")
                if current_pose is None or future_pose is None:
                    return None
                pose = _pose_delta_world_from_transformed_xyz_rotvec(current_pose, future_pose)
                return _format_xyz_rotvec_pose(pose, _pose_output_format(kind, self.canonical_adapter, component))

            semantic = _pose_semantics(kind, self.canonical_adapter, component)
            if semantic in {"absolute", "absolute_target", "target", "target_pose"}:
                current_pose = self._canonical_absolute_pose_from_component(
                    parquet,
                    component,
                    reference_indices if reference_indices is not None else indices,
                    "state",
                )
                target_pose = self._canonical_absolute_pose_from_component(
                    parquet, component, selected_indices, "action"
                )
                if current_pose is None or target_pose is None:
                    return None
                pose = _pose_delta_world_from_transformed_xyz_rotvec(current_pose, target_pose)  #绝对动作转换为相对动作
                return _format_xyz_rotvec_pose(pose, _pose_output_format(kind, self.canonical_adapter, component)) 

            state_delta_key = component.get("state_delta_key")
            if state_delta_key:
                if next_indices is None:
                    raise ValueError(f"canonical_adapter for {self.name} uses state_delta_key without next_indices")
                current = _select_feature(
                    parquet, str(state_delta_key), indices, _component_raw_slice(component, "state")
                )
                future = _select_feature(
                    parquet, str(state_delta_key), next_indices, _component_raw_slice(component, "state")
                )
                pose = _pose_delta_xyz_rotvec(current, future, pose_input)
                pose = self._canonical_transform_pose(pose, component, kind, pose_input, is_delta=True)
                return _format_xyz_rotvec_pose(pose, _pose_output_format(kind, self.canonical_adapter, component))

            state_future_key = component.get("state_future_key")
            if state_future_key:
                if next_indices is None:
                    raise ValueError(f"canonical_adapter for {self.name} uses state_future_key without next_indices")
                future = _select_feature(
                    parquet, str(state_future_key), next_indices, _component_raw_slice(component, "state")
                )
                pose = _pose_to_xyz_rotvec(future, pose_input)
                pose = self._canonical_transform_pose(pose, component, kind, pose_input, is_delta=False)
                return _format_xyz_rotvec_pose(pose, _pose_output_format(kind, self.canonical_adapter, component))

        direct = self._canonical_component_tensor(parquet, component, selected_indices, kind)
        if direct is not None:
            pose = _pose_to_xyz_rotvec(direct, pose_input)  
            pose = self._canonical_transform_pose(pose, component, kind, pose_input)
            return _format_xyz_rotvec_pose(pose, _pose_output_format(kind, self.canonical_adapter, component))

        pos_key = component.get(f"{kind}_position_key") or component.get("position_key")
        rot_key = (
            component.get(f"{kind}_rotation_key")
            or component.get(f"{kind}_orientation_key")
            or component.get("rotation_key")
            or component.get("orientation_key")
        )
        if not pos_key or not rot_key:
            return None

        pos_slice = component.get(f"{kind}_position_slice") or component.get("position_slice")
        rot_slice = (
            component.get(f"{kind}_rotation_slice")
            or component.get(f"{kind}_orientation_slice")
            or component.get("rotation_slice")
            or component.get("orientation_slice")
        )
        pos = _select_feature(parquet, str(pos_key), selected_indices, pos_slice)
        rot = _select_feature(parquet, str(rot_key), selected_indices, rot_slice)
        if pos.shape[-1] != 3:
            raise ValueError(f"Canonical ee position expects 3 dims from {pos_key!r}, got {pos.shape[-1]}")
        if rot.shape[-1] == 4:
            pose_input = str(self.canonical_adapter.get("pose_input") or "split_xyz_xyzw")
            rotvec = _quat_to_rotvec_xyzw(rot) if "xyzw" in pose_input else _quat_to_rotvec_wxyz(rot)
        elif rot.shape[-1] == 3:
            rotvec = _rpy_to_rotvec(rot)
        else:
            raise ValueError(f"Canonical ee rotation expects 3 or 4 dims from {rot_key!r}, got {rot.shape[-1]}")
        pose = torch.cat([pos, rotvec], dim=-1)
        pose = self._canonical_transform_pose(pose, component, kind, pose_input)
        return _format_xyz_rotvec_pose(pose, _pose_output_format(kind, self.canonical_adapter, component))

    def _canonical_project(
        self,
        parquet: dict[str, torch.Tensor | list[Any]],
        indices: list[int],
        kind: str,
        next_indices: list[int] | None = None,
        reference_indices: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_canonical_adapter:
            raise RuntimeError("canonical projection requested without canonical_adapter")
        dim = int(self.canonical_dim)
        out = torch.zeros((len(indices), dim), dtype=torch.float32)
        dim_mask = torch.ones(dim, dtype=torch.bool)
        slots = self.canonical_adapter.get("slots")
        if not isinstance(slots, dict):
            raise ValueError(f"canonical_adapter for {self.name} is missing slots")

        skip_component_keys = {
            "pad",
            "target_slot",
            "target_indices",
            "state_target_indices",
            "action_target_indices",
            "notes",
            "note",
        }
        for slot_name, slot_cfg in slots.items():
            if not isinstance(slot_cfg, dict) or slot_cfg.get("pad") is True:
                continue
            for component_name, component in slot_cfg.items():
                if component_name in skip_component_keys:
                    continue
                component = slot_cfg.get(component_name)
                if not isinstance(component, dict) or component.get("pad") is True:
                    continue
                target_indices = component.get(f"{kind}_target_indices") or component.get("target_indices")
                if target_indices is None:
                    continue
                if component_name == "ee":
                    values = self._canonical_pose_from_component(
                        parquet,
                        component,
                        indices,
                        kind,
                        next_indices,
                        reference_indices,
                    )
                elif kind == "action" and component.get("action_from_next_state"):
                    if next_indices is None:
                        raise ValueError(
                            f"canonical_adapter for {self.name} uses action_from_next_state without next_indices"
                        )
                    values = self._canonical_component_tensor(parquet, component, next_indices, "state")
                elif kind == "action" and component.get("action_from_next_action"):
                    if next_indices is None:
                        raise ValueError(
                            f"canonical_adapter for {self.name} uses action_from_next_action without next_indices"
                        )
                    values = self._canonical_component_tensor(parquet, component, next_indices, "action")
                else:
                    values = self._canonical_component_tensor(parquet, component, indices, kind)
                if values is None:
                    continue
                _assign_target_indices(out, dim_mask, target_indices, values)
        return out, dim_mask

    def _canonical_action_delta_from_states(
        self,
        current_state: torch.Tensor,
        future_state: torch.Tensor,
        dim_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "_canonical_action_delta_from_states is deprecated. "
            "Use _canonical_project(..., kind='action') so state/action slices can differ."
        )

    def __len__(self) -> int:
        return int(sum(self.episode_lengths))

    def valid_start_counts(self, allow_padding_at_end: bool = False) -> np.ndarray:
        if self.subtask_action_segments:
            counts = []
            for trajectory_pos, episode_index in enumerate(self.trajectory_ids):
                episode_len = int(self.trajectory_lengths[int(trajectory_pos)])
                intervals = self._valid_start_intervals_for_episode(
                    int(episode_index),
                    episode_len,
                    allow_padding_at_end,
                )
                counts.append(sum(end - start + 1 for start, end in intervals))
            return np.asarray(counts, dtype=np.int64)
        if allow_padding_at_end:
            return np.maximum(self.trajectory_lengths, 0).astype(np.int64)
        return np.maximum(self.trajectory_lengths - int(self.max_delta_index), 0).astype(np.int64)

    def max_start_for_trajectory_pos(self, trajectory_pos: int, allow_padding_at_end: bool = False) -> int:
        if self.subtask_action_segments:
            episode_index = int(self.trajectory_ids[int(trajectory_pos)])
            episode_len = int(self.trajectory_lengths[int(trajectory_pos)])
            intervals = self._valid_start_intervals_for_episode(
                episode_index,
                episode_len,
                allow_padding_at_end,
            )
            return max((end for _, end in intervals), default=-1)
        length = int(self.trajectory_lengths[int(trajectory_pos)])
        if allow_padding_at_end:
            return length - 1
        return length - 1 - int(self.max_delta_index)

    def get_episode_data(self, selected_episode_pos: int) -> dict[str, torch.Tensor | list[Any]]:
        episode_index = self.selected_episodes[selected_episode_pos]
        return self._load_episode_parquet(episode_index)

    def get_step_item(self, episode_index: int, frame_idx: int) -> dict[str, Any]:
        parquet = self._load_episode_parquet(episode_index)
        return self._get_step_item_from_parquet(episode_index, frame_idx, parquet)

    def get_step_items(self, episode_index: int, frame_indices: list[int]) -> list[dict[str, Any]]:
        """Build multiple frames from one episode while loading its parquet only once."""
        if not frame_indices:
            return []
        parquet = self._load_episode_parquet(episode_index)
        return [
            self._get_step_item_from_parquet(episode_index, int(frame_idx), parquet) for frame_idx in frame_indices
        ]

    def _get_step_item_from_parquet(
        self,
        episode_index: int,
        frame_idx: int,
        parquet: dict[str, torch.Tensor | list[Any]],
    ) -> dict[str, Any]:
        episode_meta_len = int(self.episodes_dict[episode_index]["length"])
        episode_start = self.episode_start_frame(int(episode_index))
        parquet_rows = _parquet_num_rows(parquet)
        episode_len = min(episode_meta_len, max(0, parquet_rows - episode_start))
        if episode_len <= 0:
            raise RuntimeError(f"Episode {episode_index} has no readable rows.")
        frame_idx = max(0, min(episode_len - 1, int(frame_idx)))

        state_indices, state_pad = self._query_indices(frame_idx, episode_len, self.num_frames_per_sample)
        action_indices, action_pad = self._query_indices(frame_idx, episode_len, self.action_size)
        source_state_indices = self.source_frame_indices(episode_index, state_indices)
        source_action_indices = self.source_frame_indices(episode_index, action_indices)
        source_action_reference_indices = [source_state_indices[0]] * len(source_action_indices)
        action_next_local_indices = [min(episode_len - 1, idx + self.global_sample_stride) for idx in action_indices]
        source_action_next_indices = self.source_frame_indices(episode_index, action_next_local_indices)

        action_clean = self.clean_frame_mask(
            int(episode_index),
            action_indices,
            episode_len=episode_len,
        )
        next_clean = self.clean_frame_mask(
            int(episode_index),
            action_next_local_indices,
            episode_len=episode_len,
        )
        action_pad = action_pad | ~(action_clean & next_clean)

        if self.use_canonical_adapter:
            state, state_dim_is_pad = self._canonical_project(parquet, source_state_indices, "state")
            action, action_dim_is_pad = self._canonical_project(
                parquet,
                source_action_indices,
                "action",
                source_action_next_indices,
                source_action_reference_indices,
            )
        else:
            state, state_dim_is_pad = _pad_last_dim(
                _select_rows_concat(parquet, self.state_keys, source_state_indices), self.state_target_dim
            )
            action, action_dim_is_pad = _pad_last_dim(
                _select_rows_concat(parquet, self.action_keys, source_action_indices), self.action_target_dim
            )

        item: dict[str, Any] = {
            "dataset_name": self.name,
            "embodiment": self.spec.embodiment,
            "control_schema": self.spec.control_schema or self.spec.embodiment,
            "stats_group": self.spec.stats_group or self.name,
            "stats_group_fallbacks": self.spec.stats_group_fallbacks,
            "canonical_action_layout": self.canonical_adapter.get("layout", ""),
            "action_delta_from_state_indices": tuple(
                int(index) for index in self.canonical_adapter.get("delta_from_state_indices", ())
            ),
            "episode_index": torch.tensor(episode_index, dtype=torch.long),
            "sample_frame_index": torch.tensor(frame_idx, dtype=torch.long),
            "source_episode_index": torch.tensor(self.source_episode_index(episode_index), dtype=torch.long),
            "source_frame_index": torch.tensor(self.source_frame_index(episode_index, frame_idx), dtype=torch.long),
            "observation.state": state,
            "action": action,
            "observation.state_is_pad": state_pad,
            "action_is_pad": action_pad,
            "observation.state_dim_is_pad": state_dim_is_pad,
            "action_dim_is_pad": action_dim_is_pad,
        }
        if self.timestamp_key in parquet:
            item["timestamp"] = _select_rows(parquet, self.timestamp_key, source_state_indices)
        if self.frame_index_key in parquet:
            item["frame_index"] = _select_rows(parquet, self.frame_index_key, source_state_indices)
        else:
            item["frame_index"] = torch.as_tensor(source_state_indices, dtype=torch.long)
        task_index, task = self._episode_task(episode_index, self.source_frame_index(episode_index, frame_idx), parquet)
        item["task_index"] = torch.tensor(task_index, dtype=torch.long)
        item["task"] = task
        prompt = DEFAULT_PROMPT.format(task=task)
        item["prompt"] = prompt
        if self.spec.local_text_embedding_cache_dir:
            context, context_mask = self._get_cached_text_context(prompt)
            item["context"] = context
            item["context_mask"] = context_mask
        return item

    def iter_episode_stat_batches(
        self,
        episode_index: int,
        *,
        allow_padding_at_end: bool = False,
        max_windows_per_batch: int = 4096,
        max_start: int | None = None,
        start_indices: list[int] | np.ndarray | torch.Tensor | None = None,
    ):
        """Yield canonical state/action windows for one episode in vectorized batches.

        This is intentionally image-free and prompt-free; it mirrors get_step_item's
        state/action construction for normalization stats without recomputing
        overlapping windows one start frame at a time.
        """

        episode_meta_len = int(self.episodes_dict[int(episode_index)]["length"])
        parquet = self._load_episode_parquet(int(episode_index))
        episode_start = self.episode_start_frame(int(episode_index))
        episode_len = min(episode_meta_len, max(0, _parquet_num_rows(parquet) - episode_start))
        if episode_len <= 0:
            raise RuntimeError(f"Episode {episode_index} has no readable rows.")

        if max_start is None:
            max_start = episode_len - 1 if allow_padding_at_end else episode_len - 1 - int(self.max_delta_index)
        max_start = min(int(max_start), episode_len - 1)
        if max_start < 0:
            return

        if start_indices is None:
            all_starts = torch.arange(0, max_start + 1, dtype=torch.long)
        else:
            all_starts = torch.as_tensor(start_indices, dtype=torch.long).flatten()
            if all_starts.numel() == 0:
                return
            if bool(((all_starts < 0) | (all_starts > max_start)).any()):
                raise IndexError(f"Stats start indices must be in [0, {max_start}] for episode {episode_index}.")
        batch_size = max(1, int(max_windows_per_batch))
        for offset in range(0, int(all_starts.numel()), batch_size):
            starts = all_starts[offset : offset + batch_size]
            num_windows = int(starts.numel())

            state_indices, state_pad = self._query_indices_batch(starts, episode_len, self.num_frames_per_sample)
            action_indices, action_pad = self._query_indices_batch(starts, episode_len, self.action_size)
            source_state_indices = self.source_frame_indices(int(episode_index), state_indices)
            source_action_indices = self.source_frame_indices(int(episode_index), action_indices)
            state_reference_indices = (
                torch.as_tensor(source_state_indices, dtype=torch.long)
                .reshape(num_windows, self.num_frames_per_sample)[:, 0]
            )
            source_action_reference_indices = (
                state_reference_indices[:, None]
                .expand(num_windows, self.action_size)
                .reshape(-1)
                .tolist()
            )
            action_next_indices = (
                torch.as_tensor(action_indices, dtype=torch.long)
                .add(int(self.global_sample_stride))
                .clamp(max=episode_len - 1)
                .tolist()
            )
            source_action_next_indices = self.source_frame_indices(int(episode_index), action_next_indices)
            action_clean = self.clean_frame_mask(int(episode_index), action_indices, episode_len=episode_len)
            next_clean = self.clean_frame_mask(int(episode_index), action_next_indices, episode_len=episode_len)
            action_pad = action_pad | ~(action_clean.reshape_as(action_pad) & next_clean.reshape_as(action_pad))

            if self.use_canonical_adapter:
                state_flat, state_dim_is_pad = self._canonical_project(parquet, source_state_indices, "state")
                state = state_flat.reshape(num_windows, self.num_frames_per_sample, -1)

                action_flat, action_dim_is_pad = self._canonical_project(
                    parquet,
                    source_action_indices,
                    "action",
                    source_action_next_indices,
                    source_action_reference_indices,
                )
                action = action_flat.reshape(num_windows, self.action_size, -1)
            else:
                state_flat, state_dim_is_pad = _pad_last_dim(
                    _select_rows_concat(parquet, self.state_keys, source_state_indices),
                    self.state_target_dim,
                )
                action_flat, action_dim_is_pad = _pad_last_dim(
                    _select_rows_concat(parquet, self.action_keys, source_action_indices),
                    self.action_target_dim,
                )
                state = state_flat.reshape(num_windows, self.num_frames_per_sample, -1)
                action = action_flat.reshape(num_windows, self.action_size, -1)

            yield {
                "dataset_name": self.name,
                "stats_group": self.spec.stats_group or self.name,
                "stats_group_fallbacks": self.spec.stats_group_fallbacks,
                "action_delta_from_state_indices": tuple(
                    int(index) for index in self.canonical_adapter.get("delta_from_state_indices", ())
                ),
                "episode_index": torch.full((num_windows,), int(episode_index), dtype=torch.long),
                "sample_frame_index": starts,
                "observation.state": state,
                "action": action,
                "observation.state_is_pad": state_pad,
                "action_is_pad": action_pad,
                "observation.state_dim_is_pad": state_dim_is_pad,
                "action_dim_is_pad": action_dim_is_pad,
                "_stats_num_samples": num_windows,
            }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode_index, frame_idx = self._global_to_episode_frame(int(idx))
        return self.get_step_item(episode_index, frame_idx)

    def get_video_for_item(
        self,
        item: dict[str, Any],
        video_key: str | None = None,
        sample_indices: list[int] | tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if not self.video_keys:
            raise RuntimeError("dataset has no video keys")
        video_key = video_key or self.video_key or self.video_keys[0]
        episode_index = int(item["episode_index"].item())
        if "timestamp" in item:
            timestamps = [float(x) for x in item["timestamp"].tolist()]
        else:
            frame_indices = [int(x) for x in item["frame_index"].tolist()]
            timestamps = [idx / self.fps for idx in frame_indices]
        if sample_indices is not None:
            timestamps = [timestamps[int(idx)] for idx in sample_indices if 0 <= int(idx) < len(timestamps)]
        offset = self._video_timestamp_offset(episode_index, video_key)
        if offset:
            timestamps = [offset + ts for ts in timestamps]
        return self.decode_episode_video(episode_index, video_key, timestamps)


def safe_hash(input_tuple: tuple[Any, ...]) -> int:
    digest = hashlib.sha256(repr(input_tuple).encode("utf-8")).hexdigest()
    return int(digest, 16) & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class MixedPetrelLeRobotDataset(torch.utils.data.Dataset):
    """DreamZero-style map dataset mixture for remote Petrel datasets.

    During training, each global index deterministically samples:
      1. a dataset according to dataset_sampling_weights;
      2. an episode according to trajectory_sampling_weights;
      3. a valid start frame in that episode.

    This mirrors DreamZero's mixture policy while keeping this file standalone.
    """

    def __init__(
        self,
        datasets: list[PetrelLeRobotDataset],
        dataset_weights: list[float] | None = None,
        *,
        training: bool = True,
        balance_dataset_weights: bool = False,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        allow_padding_at_end: bool = False,
        epoch_length: int | None = None,
        eval_episodes_per_dataset: int | None = None,
        eval_frames_per_episode: int | None = None,
        eval_frame_policy: str = "first",
        eval_scope: str = "dataset",
        eval_segments_per_source: int | None = None,
        eval_total_samples: int | None = None,
        eval_fixed_samples: list[dict[str, Any]] | None = None,
        eval_fixed_per_dataset: bool = False,
        eval_fixed_episode_pos: int | None = 0,
        eval_fixed_frame_index: int = 0,
        eval_exclude_from_training: str | None = None,
        training_block_size: int = 1,
        training_frame_stride: int = 1,
    ):
        if not datasets:
            raise ValueError("at least one dataset is required")
        self.datasets = datasets  # List: [PetrelLeRobotDataset]
        self.training = bool(training)
        self.balance_dataset_weights = bool(balance_dataset_weights)
        self.balance_trajectory_weights = bool(balance_trajectory_weights)
        self.seed = int(seed)
        self.allow_padding_at_end = bool(allow_padding_at_end)
        self.training_block_size = max(1, int(training_block_size)) if self.training else 1
        self.training_frame_stride = max(1, int(training_frame_stride))
        self.prefer_sequential_indices = self.training and self.training_block_size > 1
        self.epoch = 0
        # info action shape sum
        max_action_dim = max((int(ds.action_dim or 0) for ds in datasets), default=0) or None
        # info state shape sum 
        max_state_dim = max((int(ds.state_dim or 0) for ds in datasets), default=0) or None
        for ds in datasets:
            ds.set_target_dims(
                action_target_dim=ds.action_target_dim or max_action_dim,
                state_target_dim=ds.state_target_dim or max_state_dim,
            )
        eval_candidate_steps: list[tuple[int, int, int]] = []
        if eval_fixed_samples:
            eval_candidate_steps = self._build_fixed_eval_steps(eval_fixed_samples)
        elif eval_fixed_per_dataset:
            eval_candidate_steps = self._build_per_dataset_fixed_eval_steps(
                episode_pos=eval_fixed_episode_pos,
                frame_index=eval_fixed_frame_index,
            )

        exclude_mode = str(eval_exclude_from_training or "").lower()
        self.excluded_eval_steps: set[tuple[int, int, int]] = set()
        self.excluded_eval_episodes: set[tuple[int, int]] = set()
        if self.training and eval_candidate_steps and exclude_mode not in {"", "none", "false", "off"}:
            if exclude_mode in {"episode", "episodes", "trajectory", "trajectories"}:
                self.excluded_eval_episodes = {
                    (ds_pos, episode_index) for ds_pos, episode_index, _ in eval_candidate_steps
                }
            elif exclude_mode in {"sample", "samples", "step", "steps"}:
                self.excluded_eval_steps = set(eval_candidate_steps)
            else:
                raise ValueError(
                    f"Unsupported eval_exclude_from_training={eval_exclude_from_training!r}; use episode/sample/none"
                )

        if self.training:
            self.dataset_lengths = np.asarray(
                [int(self._training_valid_start_counts(ds_pos).sum()) for ds_pos in range(len(datasets))],
                dtype=np.float64,
            )
        else:
            self.dataset_lengths = np.asarray([len(ds) for ds in datasets], dtype=np.float64)
        weights = np.asarray(
            dataset_weights if dataset_weights is not None else [1.0] * len(datasets), dtype=np.float64
        )
        if weights.shape != self.dataset_lengths.shape:
            raise ValueError(
                f"dataset_weights shape {weights.shape} does not match datasets {self.dataset_lengths.shape}"
            )
        weights = weights * (self.dataset_lengths > 0)
        if self.balance_dataset_weights:
            weights = weights * self.dataset_lengths
        if float(weights.sum()) <= 0:
            raise ValueError("dataset sampling weights must sum to a positive value")
        self.dataset_sampling_weights = weights / weights.sum()

        self.trajectory_sampling_weights: list[np.ndarray] = []
        for ds_pos, ds in enumerate(datasets):
            valid_start_counts = (
                self._training_valid_start_counts(ds_pos).astype(np.float64)
                if self.training
                else ds.valid_start_counts(self.allow_padding_at_end).astype(np.float64)
            )
            traj_weights = (valid_start_counts > 0).astype(np.float64)
            if self.balance_trajectory_weights:
                traj_weights *= valid_start_counts
            if float(traj_weights.sum()) <= 0:
                if self.training:
                    self.trajectory_sampling_weights.append(np.zeros(len(ds.trajectory_ids), dtype=np.float64))
                    continue
                raise ValueError(f"dataset {ds.name} has no valid trajectories")
            self.trajectory_sampling_weights.append(traj_weights / traj_weights.sum())

        default_epoch_length = (
            int((self.dataset_lengths * self.dataset_sampling_weights).sum())
            if self.training
            else int(self.dataset_lengths.sum())
        )
        self.epoch_length = int(epoch_length) if epoch_length is not None else max(default_epoch_length, 1)
        self.cumulative_lengths = np.cumsum(self.dataset_lengths.astype(np.int64)).tolist()
        self.eval_steps: list[tuple[int, int, int]] | None = None
        if not self.training and eval_candidate_steps:
            self.eval_steps = eval_candidate_steps
            if not self.eval_steps:
                raise ValueError("fixed eval subset is empty")
        elif not self.training and (
            eval_episodes_per_dataset is not None
            or eval_frames_per_episode is not None
            or eval_segments_per_source is not None
            or eval_total_samples is not None
            or str(eval_scope or "dataset").lower() != "dataset"
        ):
            self.eval_steps = self._build_eval_steps(
                episodes_per_dataset=eval_episodes_per_dataset,
                frames_per_episode=eval_frames_per_episode,
                frame_policy=eval_frame_policy,
                scope=eval_scope,
                segments_per_source=eval_segments_per_source,
                total_samples=eval_total_samples,
            )
            if not self.eval_steps:
                raise ValueError("eval subset is empty; check episode/frame limits and allow_padding_at_end")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _training_valid_start_counts(self, ds_pos: int) -> np.ndarray:
        ds = self.datasets[int(ds_pos)]
        counts = ds.valid_start_counts(self.allow_padding_at_end).astype(np.int64, copy=True)
        if self.excluded_eval_episodes:
            for traj_pos, episode_index in enumerate(ds.trajectory_ids):
                if (int(ds_pos), int(episode_index)) in self.excluded_eval_episodes:
                    counts[traj_pos] = 0
        if self.excluded_eval_steps:
            for excluded_ds_pos, episode_index, _ in self.excluded_eval_steps:
                if int(excluded_ds_pos) != int(ds_pos):
                    continue
                matches = np.where(ds.trajectory_ids == int(episode_index))[0]
                if len(matches):
                    traj_pos = int(matches[0])
                    counts[traj_pos] = max(0, int(counts[traj_pos]) - 1)
        return counts

    def _is_training_excluded(self, ds_pos: int, episode_index: int, frame_idx: int) -> bool:
        if (int(ds_pos), int(episode_index)) in self.excluded_eval_episodes:
            return True
        if (int(ds_pos), int(episode_index), int(frame_idx)) in self.excluded_eval_steps:
            return True
        return False

    def _resolve_fixed_eval_dataset_pos(self, sample: dict[str, Any]) -> int:
        if "spec_index" in sample:
            pos = int(sample["spec_index"])
            if pos < 0 or pos >= len(self.datasets):
                raise IndexError(f"fixed eval spec_index={pos} out of range for {len(self.datasets)} datasets")
            return pos
        if "dataset_index" in sample:
            pos = int(sample["dataset_index"])
            if pos < 0 or pos >= len(self.datasets):
                raise IndexError(f"fixed eval dataset_index={pos} out of range for {len(self.datasets)} datasets")
            return pos
        name = sample.get("dataset", sample.get("name"))
        if name is None:
            raise KeyError("fixed eval sample needs one of spec_index, dataset_index, dataset, or name")
        matches = [idx for idx, ds in enumerate(self.datasets) if ds.name == str(name)]
        if not matches:
            raise KeyError(f"fixed eval dataset {name!r} not found")
        if len(matches) > 1:
            raise ValueError(f"fixed eval dataset {name!r} is ambiguous; use spec_index instead")
        return matches[0]

    def _build_fixed_eval_steps(self, samples: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
        steps: list[tuple[int, int, int]] = []
        for sample_idx, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise TypeError(f"fixed eval sample {sample_idx} must be a mapping")
            ds_pos = self._resolve_fixed_eval_dataset_pos(sample)
            ds = self.datasets[ds_pos]
            if "episode_index" in sample:
                episode_index = int(sample["episode_index"])
            else:
                episode_pos = int(sample.get("episode_pos", sample.get("trajectory_pos", 0)))
                if episode_pos < 0 or episode_pos >= len(ds.trajectory_ids):
                    raise IndexError(f"fixed eval episode_pos={episode_pos} out of range for dataset {ds.name}")
                episode_index = int(ds.trajectory_ids[episode_pos])
            if episode_index not in ds.episodes_dict:
                raise KeyError(f"fixed eval episode_index={episode_index} not found in dataset {ds.name}")
            traj_matches = np.where(ds.trajectory_ids == episode_index)[0]
            if len(traj_matches) == 0:
                raise KeyError(f"fixed eval episode_index={episode_index} is not selected in dataset {ds.name}")
            traj_pos = int(traj_matches[0])
            max_start = ds.max_start_for_trajectory_pos(traj_pos, self.allow_padding_at_end)
            frame_idx = int(sample.get("frame_index", sample.get("start_frame", 0)))
            if frame_idx < 0 or frame_idx > max_start:
                raise IndexError(
                    f"fixed eval frame_index={frame_idx} invalid for dataset={ds.name} "
                    f"episode={episode_index}; valid range is 0..{max_start}"
                )
            steps.append((ds_pos, episode_index, frame_idx))
        return steps

    def _build_per_dataset_fixed_eval_steps(
        self, *, episode_pos: int | None = 0, frame_index: int = 0
    ) -> list[tuple[int, int, int]]:
        steps: list[tuple[int, int, int]] = []
        for ds_pos, ds in enumerate(self.datasets):
            valid_counts = ds.valid_start_counts(self.allow_padding_at_end)
            valid_positions = [pos for pos, count in enumerate(valid_counts) if int(count) > 0]
            if not valid_positions:
                raise ValueError(f"dataset {ds.name} has no valid trajectory for fixed per-dataset eval")
            if episode_pos is None:
                traj_pos = valid_positions[0]
            else:
                pos = int(episode_pos)
                if pos < 0 or pos >= len(valid_positions):
                    raise IndexError(f"fixed eval episode_pos={pos} out of valid range for dataset {ds.name}")
                traj_pos = valid_positions[pos]
            episode_index = int(ds.trajectory_ids[traj_pos])
            max_start = ds.max_start_for_trajectory_pos(traj_pos, self.allow_padding_at_end)
            frame_idx = int(frame_index)
            if frame_idx < 0 or frame_idx > max_start:
                raise IndexError(
                    f"fixed eval frame_index={frame_idx} invalid for dataset={ds.name} "
                    f"episode={episode_index}; valid range is 0..{max_start}"
                )
            steps.append((ds_pos, episode_index, frame_idx))
        return steps

    def _eval_frame_indices(
        self,
        max_start: int,
        frames_per_episode: int | None,
        frame_policy: str,
        rng: np.random.Generator | None = None,
    ) -> list[int]:
        if max_start < 0:
            return []
        n = max(1, int(frames_per_episode or 1))
        policy = str(frame_policy or "first").lower()
        if policy == "first":
            return list(range(0, min(n, max_start + 1)))
        if policy == "middle":
            if n == 1:
                return [max_start // 2]
            values = np.linspace(0, max_start, num=min(n, max_start + 1))
            return sorted({int(round(v)) for v in values})
        if policy == "last":
            start = max(0, max_start - n + 1)
            return list(range(start, max_start + 1))
        if policy == "uniform":
            values = np.linspace(0, max_start, num=min(n, max_start + 1))
            return sorted({int(round(v)) for v in values})
        if policy == "random":
            rng = rng or np.random.default_rng(self.seed)
            count = min(n, max_start + 1)
            return sorted(int(v) for v in rng.choice(max_start + 1, size=count, replace=False))
        raise ValueError(f"Unsupported eval_frame_policy={frame_policy!r}; use first/middle/last/uniform/random")

    @staticmethod
    def _allocate_eval_counts(weights: list[float], total: int) -> list[int]:
        weights_arr = np.asarray(weights, dtype=np.float64)
        counts = np.zeros(len(weights_arr), dtype=np.int64)
        total = max(0, int(total))
        positive = weights_arr > 0
        if total <= 0 or not bool(positive.any()):
            return counts.tolist()

        remaining_total = total
        if total >= int(positive.sum()):
            counts[positive] = 1
            remaining_total -= int(positive.sum())

        if remaining_total > 0:
            active_weights = np.where(positive, weights_arr, 0.0)
            raw = active_weights / float(active_weights.sum()) * float(remaining_total)
            add = np.floor(raw).astype(np.int64)
            counts += add
            remainder = remaining_total - int(add.sum())
            if remainder > 0:
                fractions = raw - add
                order = np.argsort(-fractions)
                for idx in order[:remainder]:
                    if positive[int(idx)]:
                        counts[int(idx)] += 1
        return counts.tolist()

    def _sample_one_source_eval_step(
        self, group_id: Any, ds_positions: list[int], sample_index: int
    ) -> tuple[int, int, int] | None:
        valid_totals = []
        for ds_pos in ds_positions:
            counts = self.datasets[ds_pos].valid_start_counts(self.allow_padding_at_end).astype(np.float64)
            valid_totals.append(float(counts.sum()))
        weights = np.asarray(valid_totals, dtype=np.float64)
        if float(weights.sum()) <= 0:
            return None
        rng = np.random.default_rng(safe_hash(("eval_source", self.seed, group_id, int(sample_index))))
        ds_pos = int(rng.choice(ds_positions, p=weights / weights.sum()))
        ds = self.datasets[ds_pos]
        valid_counts = ds.valid_start_counts(self.allow_padding_at_end).astype(np.float64)
        if float(valid_counts.sum()) <= 0:
            return None
        traj_pos = int(rng.choice(len(ds.trajectory_ids), p=valid_counts / valid_counts.sum()))
        episode_index = int(ds.trajectory_ids[traj_pos])
        frame_idx = ds.sample_valid_start_for_trajectory_pos(
            traj_pos,
            rng,
            self.allow_padding_at_end,
        )
        if frame_idx is None:
            return None
        return ds_pos, episode_index, frame_idx

    def _build_eval_steps(
        self,
        *,
        episodes_per_dataset: int | None = None,
        frames_per_episode: int | None = None,
        frame_policy: str = "first",
        scope: str = "dataset",
        segments_per_source: int | None = None,
        total_samples: int | None = None,
    ) -> list[tuple[int, int, int]]:
        steps: list[tuple[int, int, int]] = []
        scope = str(scope or "dataset").lower()
        grouped_scopes = {"source", "group", "source_file", "source_type", "type", "source_entry"}
        total_samples_int = None if total_samples in (None, "") else int(total_samples)
        if scope in grouped_scopes or (total_samples_int is not None and total_samples_int > 0):
            groups: dict[Any, list[int]] = {}
            for ds_pos, ds in enumerate(self.datasets):
                if scope in {"source_type", "type", "source_entry"}:
                    key = str(ds.spec.source_name or ds.spec.group_id)
                elif scope in {"dataset", "leaf", "dataset_name"}:
                    key = f"{ds_pos}:{ds.name}"
                else:
                    key = int(ds.spec.group_id)
                groups.setdefault(key, []).append(ds_pos)
            group_items = sorted(groups.items(), key=lambda item: str(item[0]))
            if total_samples_int is not None and total_samples_int > 0:
                group_weights = [
                    sum(float(self.dataset_sampling_weights[ds_pos]) for ds_pos in ds_positions)
                    for _, ds_positions in group_items
                ]
                counts = self._allocate_eval_counts(group_weights, total_samples_int)
                for (group_id, ds_positions), count in zip(group_items, counts):
                    for sample_index in range(int(count)):
                        step = self._sample_one_source_eval_step(group_id, ds_positions, sample_index)
                        if step is not None:
                            steps.append(step)
                return steps

            n = max(1, int(segments_per_source or 1))
            for group_id, ds_positions in group_items:
                for sample_index in range(n):
                    step = self._sample_one_source_eval_step(group_id, ds_positions, sample_index)
                    if step is not None:
                        steps.append(step)
            return steps

        if scope != "dataset":
            raise ValueError(f"Unsupported eval scope={scope!r}; use dataset/source/source_type")
        for ds_pos, ds in enumerate(self.datasets):
            valid_counts = ds.valid_start_counts(self.allow_padding_at_end)
            valid_positions = [pos for pos, count in enumerate(valid_counts) if int(count) > 0]
            if episodes_per_dataset is not None and int(episodes_per_dataset) > 0:
                valid_positions = valid_positions[: int(episodes_per_dataset)]
            for traj_pos in valid_positions:
                episode_index = int(ds.trajectory_ids[traj_pos])
                max_start = ds.max_start_for_trajectory_pos(traj_pos, self.allow_padding_at_end)
                rng = np.random.default_rng(safe_hash(("eval_dataset", self.seed, int(ds_pos), int(episode_index))))
                for frame_idx in self._eval_frame_indices(max_start, frames_per_episode, frame_policy, rng):
                    steps.append((ds_pos, episode_index, int(frame_idx)))
        return steps

    def __len__(self) -> int:
        if self.eval_steps is not None:
            return len(self.eval_steps)
        return self.epoch_length if self.training else int(self.dataset_lengths.sum())

    def _sample_training_step(self, index: int) -> tuple[int, int, int]:
        block_size = self.training_block_size
        block_id = int(index) // block_size
        block_offset = int(index) % block_size
        for attempt in range(128):
            rng = np.random.default_rng(safe_hash((self.epoch, int(block_id), self.seed, attempt, "episode_block")))
            dataset_index = int(rng.choice(len(self.datasets), p=self.dataset_sampling_weights))
            ds = self.datasets[dataset_index]
            traj_weights = self.trajectory_sampling_weights[dataset_index]
            if float(traj_weights.sum()) <= 0:
                continue
            traj_pos = int(rng.choice(len(ds.trajectory_ids), p=traj_weights))
            episode_index = int(ds.trajectory_ids[traj_pos])
            span = (block_size - 1) * self.training_frame_stride
            base_frame = ds.sample_valid_start_for_trajectory_pos(
                traj_pos,
                rng,
                self.allow_padding_at_end,
                extra_span=span,
            )
            if base_frame is None:
                continue
            frame_idx = int(base_frame) + block_offset * self.training_frame_stride
            if not self._is_training_excluded(dataset_index, episode_index, frame_idx):
                return dataset_index, episode_index, frame_idx
        raise RuntimeError("could not sample a non-eval training step after 128 attempts")

    def _sample_step(self, index: int) -> tuple[int, int, int]:
        if self.eval_steps is not None:
            if index < 0 or index >= len(self.eval_steps):
                raise IndexError(f"index {index} out of bounds for eval subset length {len(self.eval_steps)}")
            return self.eval_steps[index]
        if self.training:
            return self._sample_training_step(int(index))

        ds_pos = bisect.bisect_right(self.cumulative_lengths, index)
        start = 0 if ds_pos == 0 else self.cumulative_lengths[ds_pos - 1]
        episode_index, frame_idx = self.datasets[ds_pos]._global_to_episode_frame(index - start)
        return ds_pos, int(episode_index), int(frame_idx)

    def sample_step(self, index: int) -> tuple[int, int, int]:
        """Resolve a mixture index without reading parquet or video data."""
        return self._sample_step(int(index))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ds_pos, episode_index, frame_idx = self.sample_step(int(idx))
        item = self.datasets[ds_pos].get_step_item(episode_index, frame_idx)
        item["dataset_index"] = torch.tensor(ds_pos, dtype=torch.long)
        return item

    def get_video_for_item(self, item: dict[str, Any]) -> torch.Tensor:
        ds_pos = int(item["dataset_index"].item())
        return self.datasets[ds_pos].get_video_for_item(item)


def _distributed_dataset_weights(
    specs: list[DatasetSpec],
    datasets: list[PetrelLeRobotDataset],
    *,
    allow_padding_at_end: bool = False,
) -> list[float]:
    weights = [float(spec.dataset_weight) for spec in specs]
    valid_lengths = [max(int(ds.valid_start_counts(allow_padding_at_end).sum()), 1) for ds in datasets]  
    for group_id in sorted({spec.group_id for spec in specs if spec.distribute_weights}):
        indices = [i for i, spec in enumerate(specs) if spec.group_id == group_id]
        total_len = sum(valid_lengths[i] for i in indices)
        if total_len <= 0:
            continue
        group_weight = float(specs[indices[0]].dataset_weight)
        for i in indices:
            weights[i] = group_weight * (valid_lengths[i] / total_len)
    return weights


def _read_yaml_or_json(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_include_path(path: str | Path, config_dir: Path) -> Path:
    out = Path(path).expanduser()
    if not out.is_absolute():
        out = config_dir / out
    return out

def _merge_config_includes(payload: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    projection_files = payload.get("projection_files", payload.get("projection_file", []))
    if projection_files in (None, ""):
        projection_files = []
    if isinstance(projection_files, (str, Path)):
        projection_files = [projection_files]
    if not isinstance(projection_files, list):
        raise TypeError("projection_file/projection_files must be a path or list of paths")

    for include in projection_files:
        include_path = _resolve_include_path(include, config_dir)
        fragment = _read_yaml_or_json(include_path)
        if not fragment:
            continue
        if not isinstance(fragment, dict):
            raise TypeError(f"Projection include must be a mapping: {include_path}")
        for key in ("canonical_action_space", "control_schemas"):
            if key in fragment:
                existing = payload.get(key)
                value = fragment[key]
                if isinstance(existing, dict) and isinstance(value, dict):
                    merged = dict(existing)
                    merged.update(value)
                    payload[key] = merged
                else:
                    payload[key] = value

    embodiments = dict(payload.get("embodiments", {}) or {})
    for include in payload.get("embodiment_files", []) or []:
        include_path = _resolve_include_path(include, config_dir)
        fragment = _read_yaml_or_json(include_path)
        if not fragment:
            continue
        if not isinstance(fragment, dict):
            raise TypeError(f"Embodiment include must be a mapping: {include_path}")
        if isinstance(fragment.get("embodiments"), dict):
            embodiments.update(fragment["embodiments"])
        elif "name" in fragment:
            fragment = dict(fragment)
            name = str(fragment.pop("name"))
            embodiments[name] = fragment
        else:
            embodiments.update(fragment)
    if embodiments:
        payload["embodiments"] = embodiments

    ego_cfg = payload.get("ego_data", {}) or {}
    if isinstance(ego_cfg, dict):
        ego_enabled = bool(ego_cfg.get("enabled", True))
        ego_source_names = {str(name) for name in ego_cfg.get("source_files", ["egodex", "xperience"])}
    else:
        ego_enabled = bool(ego_cfg)
        ego_source_names = {"egodex", "xperience"}

    def iter_source_includes(source_files: Any) -> list[str]:
        if source_files in (None, ""):
            return []
        if isinstance(source_files, (list, tuple)):
            return [str(item) for item in source_files]
        if not isinstance(source_files, dict):
            raise TypeError("source_files must be a list or a mapping")

        includes: list[str] = []
        for name, value in source_files.items():
            include = str(name)
            enabled = True
            if isinstance(value, bool):
                enabled = value
            elif isinstance(value, (int, float)):
                enabled = bool(value)
            elif isinstance(value, str):
                text = value.strip().lower()
                if text in {"false", "off", "no", "0", "disabled"}:
                    enabled = False
                elif text in {"true", "on", "yes", "1", "enabled"}:
                    enabled = True
                else:
                    include = value
            elif isinstance(value, dict):
                enabled = bool(value.get("enabled", True))
                include = str(value.get("path", value.get("file", value.get("include", include))))
            elif value is None:
                enabled = False
            else:
                raise TypeError(f"Unsupported source_files.{name!s} value type: {type(value).__name__}")
            if not enabled:
                continue
            if "/" not in include and not Path(include).suffix:
                include = f"sources/{include}.yaml"
            includes.append(include)
        return includes

    sources = list(payload.get("sources", []) or [])
    for include in iter_source_includes(payload.get("source_files", [])):
        include_path = _resolve_include_path(include, config_dir)
        source_file = include_path.stem
        if not ego_enabled and source_file in ego_source_names:
            continue
        fragment = _read_yaml_or_json(include_path)
        if not fragment:
            continue

        def annotate_source(item: Any) -> Any:
            if isinstance(item, dict):
                item = dict(item)
                item.setdefault("_source_file", source_file)
            return item

        if isinstance(fragment, dict) and isinstance(fragment.get("sources"), list):
            sources.extend(annotate_source(item) for item in fragment["sources"])
        elif isinstance(fragment, list):
            sources.extend(annotate_source(item) for item in fragment)
        elif isinstance(fragment, dict):
            sources.append(annotate_source(fragment))
        else:
            raise TypeError(f"Source include must be a mapping or list: {include_path}")
    if sources:
        payload["sources"] = sources

    return payload


def _resolve_local_cache_template(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else OPENPI_REPO_ROOT / path)


def _pose_output_dim(value: str) -> int:
    value = str(value)
    if value in {"xyz_rotvec", "delta_xyz_rotvec", "xyz_rpy", "delta_xyz_rpy"}:
        return 6
    if value == "xyz_rot6d":
        return 9
    raise ValueError(f"Unsupported source_control_space pose representation {value!r}.")


def _control_slot_indices(value: Any, *, dim: int, path: str) -> list[int]:
    if not isinstance(value, dict) or set(value) != {"indices"}:
        raise TypeError(
            f"source_control_space slot {path} must be a mapping containing only indices, got {value!r}."
        )
    raw_indices = value["indices"]
    if not isinstance(raw_indices, (list, tuple)):
        raise TypeError(f"source_control_space slot {path}.indices must be a list.")
    indices = [int(index) for index in raw_indices]
    if not indices:
        raise ValueError(f"source_control_space slot {path} cannot be empty.")
    if len(set(indices)) != len(indices):
        raise ValueError(f"source_control_space slot {path} contains duplicate indices: {indices}.")
    if min(indices) < 0 or max(indices) >= dim:
        raise ValueError(f"source_control_space slot {path}={indices} exceeds dim={dim}.")
    return indices


def _validate_source_control_space(value: Any, contract: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("Petrel YAML source_control_space must be a mapping.")
    name = value.get("name")
    dim = int(value.get("dim", 0))
    slots = value.get("slots")
    if not isinstance(name, str) or not name:
        raise ValueError("source_control_space.name must be a non-empty string.")
    if dim <= 0:
        raise ValueError("source_control_space.dim must be positive.")
    if not isinstance(slots, dict) or not slots:
        raise ValueError("source_control_space.slots must be a non-empty mapping.")
    if dim != int(contract["state_dim"]) or dim != int(contract["action_dim"]):
        raise ValueError(
            "source_control_space.dim must equal "
            f"openpi state/action dim; got source={dim}, state={contract['state_dim']}, action={contract['action_dim']}."
        )

    normalized_slots: dict[str, dict[str, list[int]]] = {}
    claimed: dict[int, str] = {}
    for side_name, side in slots.items():
        if not isinstance(side_name, str) or not isinstance(side, dict):
            raise TypeError("source_control_space.slots entries must map names to component mappings.")
        normalized_side: dict[str, list[int]] = {}
        for component_name, component_slot in side.items():
            path = f"{side_name}.{component_name}"
            indices = _control_slot_indices(component_slot, dim=dim, path=path)
            for index in indices:
                previous = claimed.get(index)
                if previous is not None:
                    raise ValueError(f"source_control_space slots {previous} and {path} overlap at dimension {index}.")
                claimed[index] = path
            normalized_side[str(component_name)] = indices
        normalized_slots[str(side_name)] = normalized_side

    pose = value.get("pose", {})
    if not isinstance(pose, dict):
        raise TypeError("source_control_space.pose must be a mapping.")
    state_pose = str(pose.get("state", "xyz_rotvec"))
    action_pose = str(pose.get("action", "delta_xyz_rotvec"))
    _pose_output_dim(state_pose)
    _pose_output_dim(action_pose)
    action_semantics = value.get("action_semantics", {})
    if not isinstance(action_semantics, dict):
        raise TypeError("source_control_space.action_semantics must be a mapping.")
    supported_semantics = {"absolute", "delta"}
    invalid_semantics = {
        str(component): str(semantic)
        for component, semantic in action_semantics.items()
        if str(semantic) not in supported_semantics
    }
    if invalid_semantics:
        raise ValueError(
            "source_control_space.action_semantics values must be absolute or delta; "
            f"got {invalid_semantics}."
        )
    return {
        **value,
        "name": name,
        "dim": dim,
        "slots": slots,
        "slot_indices": normalized_slots,
        "pose": {"state": state_pose, "action": action_pose},
        "action_semantics": {str(key): str(item) for key, item in action_semantics.items()},
        "missing_policy": str(value.get("missing_policy", "pad_and_mask")),
        "camera_frame_delta": bool(value.get("camera_frame_delta", False)),
    }


def _raw_component_width(component: dict[str, Any], kind: str, fallback: int) -> int:
    raw_slice = _component_raw_slice(component, kind)
    if raw_slice is None:
        return fallback
    if not isinstance(raw_slice, (list, tuple)) or len(raw_slice) != 2:
        raise ValueError(f"{kind}_raw_slice must be [start, end], got {raw_slice!r}.")
    start = 0 if raw_slice[0] is None else int(raw_slice[0])
    end = fallback if raw_slice[1] is None else int(raw_slice[1])
    if end <= start:
        raise ValueError(f"{kind}_raw_slice must have positive width, got {raw_slice!r}.")
    return end - start


def _resolve_control_adapter(mapping: dict[str, Any], control_space: dict[str, Any], *, profile_name: str) -> dict[str, Any]:
    adapter = copy.deepcopy(mapping)
    adapter["layout"] = control_space["name"]
    adapter["dim"] = int(control_space["dim"])
    adapter["missing_policy"] = control_space["missing_policy"]
    adapter["camera_frame_delta"] = control_space["camera_frame_delta"]
    adapter["state_pose_output"] = control_space["pose"]["state"]
    adapter["action_pose_output"] = control_space["pose"]["action"]
    slots = adapter.get("slots")
    if not isinstance(slots, dict) or not slots:
        raise ValueError(f"Control profile {profile_name!r} mapping.slots must be non-empty.")

    claimed: dict[str, dict[int, str]] = {"state": {}, "action": {}}
    delta_from_state_indices: list[int] = []
    for side_name, side in slots.items():
        if not isinstance(side, dict):
            raise TypeError(f"Control profile {profile_name!r} slot {side_name!r} must be a mapping.")
        for component_name, component in side.items():
            if component_name == "pad" or not isinstance(component, dict) or component.get("pad") is True:
                continue
            target_slot = component.get("target_slot")
            if not isinstance(target_slot, str) or not target_slot:
                raise ValueError(
                    f"Control profile {profile_name!r} {side_name}.{component_name} requires target_slot."
                )
            target_parts = target_slot.split(".")
            if len(target_parts) != 2:
                raise ValueError(f"target_slot must use '<side>.<component>', got {target_slot!r}.")
            try:
                available_indices = control_space["slot_indices"][target_parts[0]][target_parts[1]]
            except KeyError as exc:
                raise KeyError(
                    f"Control profile {profile_name!r} target_slot {target_slot!r} is not defined by "
                    "source_control_space."
                ) from exc

            for kind in ("state", "action"):
                key = component.get(f"{kind}_key") or component.get("key")
                if not key:
                    continue
                if component_name == "ee":
                    width = _pose_output_dim(control_space["pose"][kind])
                else:
                    width = _raw_component_width(component, kind, len(available_indices))
                if width > len(available_indices):
                    raise ValueError(
                        f"Control profile {profile_name!r} {side_name}.{component_name} {kind} width={width} "
                        f"does not fit target_slot {target_slot} width={len(available_indices)}."
                    )
                selected_indices = list(available_indices[:width])
                component[f"{kind}_target_indices"] = selected_indices
                for index in selected_indices:
                    previous = claimed[kind].get(index)
                    if previous is not None:
                        raise ValueError(
                            f"Control profile {profile_name!r} maps both {previous} and "
                            f"{side_name}.{component_name} to {kind} dimension {index}."
                        )
                    claimed[kind][index] = f"{side_name}.{component_name}"

            target_semantic = control_space["action_semantics"].get(str(component_name), "absolute")
            source_semantic = str(component.get("action_semantics", "absolute"))
            if component_name != "ee" and target_semantic == "delta" and source_semantic == "absolute":
                delta_from_state_indices.extend(component.get("action_target_indices", ()))
            elif component_name != "ee" and target_semantic != source_semantic:
                raise ValueError(
                    f"Control profile {profile_name!r} cannot convert {side_name}.{component_name} "
                    f"from {source_semantic!r} to {target_semantic!r}."
                )
    adapter["delta_from_state_indices"] = sorted(set(delta_from_state_indices))
    return adapter

def _translate_openpi_petrel_config(payload: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    """Validate openpi_petrel_v1 and translate it to the reader's internal dataset specs."""
    dataset_id, contract = _validate_openpi_petrel_metadata(payload)
    storage = payload.get("storage")
    sampling = payload.get("sampling")
    mixture = payload.get("mixture")
    norm_grouping = payload.get("norm_stats", {})
    dataset_files = payload.get("dataset_files")
    control_space = _validate_source_control_space(payload.get("source_control_space"), contract)
    for name, value in (
        ("storage", storage),
        ("sampling", sampling),
        ("mixture", mixture),
    ):
        if not isinstance(value, dict):
            raise TypeError(f"Petrel YAML {name} must be a mapping.")
    if not isinstance(norm_grouping, dict):
        raise TypeError("Petrel YAML norm_stats must be a mapping when provided.")
    if "control_profiles" in payload:
        raise ValueError(
            "Global inline control_profiles are not supported; use control_profile_files to register shared "
            "profiles or define exactly one selected profile inside each dataset-group YAML."
        )
    control_profile_files = payload.get("control_profile_files", [])
    if control_profile_files in (None, ""):
        control_profile_files = []
    elif isinstance(control_profile_files, (str, Path)):
        control_profile_files = [control_profile_files]
    elif not isinstance(control_profile_files, list):
        raise TypeError("Petrel YAML control_profile_files must be a path or list of paths.")

    registered_profiles: dict[str, dict[str, Any]] = {}
    for include in control_profile_files:
        registry_path = _resolve_include_path(include, config_dir)
        registry = _read_yaml_or_json(registry_path)
        if not isinstance(registry, dict):
            raise TypeError(f"Control profile registry must contain a mapping: {registry_path}")
        if registry.get("schema") != "openpi_petrel_control_profile_registry_v1":
            raise ValueError(
                "Control profile registry must declare "
                f"schema: openpi_petrel_control_profile_registry_v1: {registry_path}"
            )
        profiles = registry.get("control_profiles")
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError(f"Control profile registry requires a non-empty control_profiles mapping: {registry_path}")
        profile_defaults = registry.get("profile_defaults", {})
        if not isinstance(profile_defaults, dict):
            raise TypeError(f"Control profile registry profile_defaults must be a mapping: {registry_path}")
        for profile_name, profile in profiles.items():
            if not isinstance(profile_name, str) or not profile_name:
                raise ValueError(f"Control profile registry contains an invalid profile name: {registry_path}")
            if not isinstance(profile, dict):
                raise TypeError(f"Control profile {profile_name!r} must be a mapping: {registry_path}")
            profile = _merge_mapping(profile_defaults, profile)
            existing = registered_profiles.get(profile_name)
            if existing is not None and existing != profile:
                raise ValueError(
                    f"Conflicting registered definitions for control profile {profile_name!r}: {registry_path}"
                )
            registered_profiles[profile_name] = profile
    if not isinstance(dataset_files, dict):
        raise TypeError("Petrel YAML dataset_files must be a mapping.")
    if "dataset_groups" in payload:
        raise ValueError("Inline dataset_groups are not supported; put every dataset in its own YAML file.")
    dataset_dir_value = dataset_files.get("directory")
    if not isinstance(dataset_dir_value, str) or not dataset_dir_value:
        raise ValueError("Petrel YAML dataset_files.directory must be a non-empty path.")
    dataset_dir = Path(dataset_dir_value).expanduser()
    if not dataset_dir.is_absolute():
        dataset_dir = config_dir / dataset_dir
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Petrel dataset YAML directory not found: {dataset_dir}")
    pattern = str(dataset_files.get("pattern", "*.yaml"))
    recursive = bool(dataset_files.get("recursive", True))
    paths = sorted(dataset_dir.rglob(pattern) if recursive else dataset_dir.glob(pattern))
    if not paths:
        raise ValueError(f"No dataset YAML files matched {pattern!r} under {dataset_dir}.")

    groups: list[dict[str, Any]] = []
    profiles: dict[str, dict[str, Any]] = {}
    seen_remote_paths: set[tuple[str, str]] = set()
    for dataset_path in paths:
        group_payload = _read_yaml_or_json(dataset_path)
        if not isinstance(group_payload, dict):
            raise TypeError(f"Dataset group YAML must contain a mapping: {dataset_path}")
        if group_payload.get("schema") != "openpi_petrel_dataset_group_v1":
            raise ValueError(f"Dataset group YAML must declare schema: openpi_petrel_dataset_group_v1: {dataset_path}")
        required_strings = ("name", "category", "profile", "format", "root")
        for key in required_strings:
            if not isinstance(group_payload.get(key), str) or not group_payload[key]:
                raise ValueError(f"Dataset group YAML {dataset_path} requires non-empty {key!r}.")
        profile_name = group_payload["profile"]
        group_profiles = group_payload.get("control_profiles")
        if group_profiles is None:
            profile = registered_profiles.get(profile_name)
            if profile is None:
                raise KeyError(
                    f"Dataset group YAML {dataset_path} selects unregistered profile {profile_name!r}; "
                    "define it inline under control_profiles or register it with control_profile_files."
                )
        else:
            if not isinstance(group_profiles, dict) or not group_profiles:
                raise ValueError(f"Dataset group YAML {dataset_path} control_profiles must be a non-empty mapping.")
            if set(group_profiles) != {profile_name}:
                raise ValueError(
                    f"Dataset group YAML {dataset_path} must define exactly its selected profile {profile_name!r}; "
                    f"got {sorted(group_profiles)}."
                )
            profile = group_profiles[profile_name]
        if not isinstance(profile, dict):
            raise TypeError(f"{dataset_path} control_profiles.{profile_name} must be a mapping.")
        mapping = profile.get("mapping")
        fields = profile.get("fields")
        if not isinstance(mapping, dict) or not isinstance(fields, dict):
            raise ValueError(f"{dataset_path} control profile {profile_name!r} requires mapping and fields mappings.")
        existing_profile = profiles.get(profile_name)
        if existing_profile is not None and existing_profile != profile:
            raise ValueError(
                f"Conflicting definitions for control profile {profile_name!r}; "
                f"dataset group {dataset_path} does not match an earlier group."
            )
        profiles[profile_name] = profile
        datasets = group_payload.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            raise ValueError(f"Dataset group YAML {dataset_path} requires a non-empty datasets list.")
        normalized_datasets: list[dict[str, Any]] = []
        for child_index, child in enumerate(datasets):
            if not isinstance(child, dict):
                raise TypeError(f"{dataset_path} datasets[{child_index}] must be a mapping.")
            if set(child) != {"name", "path"}:
                raise ValueError(
                    f"{dataset_path} datasets[{child_index}] may contain only name and path; got {sorted(child)}."
                )
            if not isinstance(child["name"], str) or not child["name"]:
                raise ValueError(f"{dataset_path} datasets[{child_index}].name must be non-empty.")
            if not isinstance(child["path"], str) or not child["path"]:
                raise ValueError(f"{dataset_path} datasets[{child_index}].path must be non-empty.")
            remote_key = (group_payload["root"].rstrip("/"), child["path"].strip("/"))
            if remote_key in seen_remote_paths:
                raise ValueError(f"Duplicate remote dataset in group YAML files: {remote_key}")
            seen_remote_paths.add(remote_key)
            normalized_datasets.append({"name": child["name"], "path": child["path"]})
        groups.append(
            {
                "name": group_payload["name"],
                "category": group_payload["category"],
                "profile": group_payload["profile"],
                "format": group_payload["format"],
                "root": group_payload["root"],
                "datasets": normalized_datasets,
                "enabled": group_payload.get("enabled", True),
                "fields": group_payload.get("fields"),
                "mapping_overrides": group_payload.get("mapping_overrides"),
                "control_type": group_payload.get("control_type"),
                "stats_group": group_payload.get("stats_group"),
                "stats_group_template": group_payload.get("stats_group_template"),
                "robot_id": group_payload.get("robot_id"),
            }
        )

    legacy_profiles: dict[str, Any] = {}
    for profile_name, profile in profiles.items():
        if not isinstance(profile_name, str) or not isinstance(profile, dict):
            raise TypeError("Each control_profiles entry must map a name to a mapping.")
        mapping = profile.get("mapping")
        fields = profile.get("fields")
        if not isinstance(mapping, dict) or not isinstance(fields, dict):
            raise ValueError(f"control_profiles.{profile_name} requires mapping and fields mappings.")
        resolved_mapping = _resolve_control_adapter(mapping, control_space, profile_name=profile_name)
        legacy_profiles[profile_name] = {
            "control_schema": profile.get("control_type"),
            "canonical_adapter": resolved_mapping,
            "modalities": fields,
            "stats_group": profile.get("stats_group"),
            "stats_group_template": profile.get("stats_group_template"),
        }

    legacy_sources: list[dict[str, Any]] = []
    known_profiles = set(legacy_profiles)
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise TypeError(f"dataset_groups[{index}] must be a mapping.")
        profile_name = group.get("profile")
        if profile_name not in known_profiles:
            raise KeyError(f"dataset_groups[{index}].profile={profile_name!r} is not defined.")
        root = group.get("root")
        datasets = group.get("datasets")
        if not isinstance(root, str) or not root:
            raise ValueError(f"dataset_groups[{index}].root must be a non-empty string.")
        root_path = Path(root)
        if root_path.is_absolute() or ":" in root.split("/", 1)[0] or ".." in root_path.parts:
            raise ValueError(
                f"dataset_groups[{index}].root must be relative to storage.local_root, got {root!r}."
            )
        robot_id = group.get("robot_id")
        if robot_id is None:
            root_parts = tuple(part for part in root_path.parts if part not in {"", "."})
            robot_id = root_parts[1] if len(root_parts) >= 2 else profile_name
        if not isinstance(robot_id, str) or not robot_id:
            raise ValueError(f"dataset_groups[{index}].robot_id must be a non-empty string when provided.")
        if not isinstance(datasets, list) or not datasets:
            raise ValueError(f"dataset_groups[{index}].datasets must be a non-empty list.")
        children: list[Any] = []
        for child_index, child in enumerate(datasets):
            if isinstance(child, str):
                children.append(child)
                continue
            if not isinstance(child, dict) or not isinstance(child.get("path"), str):
                raise TypeError(f"dataset_groups[{index}].datasets[{child_index}] must be a path string or mapping.")
            translated_child = {"dir": child["path"]}
            key_map = {
                "name": "name",
                "enabled": "enabled",
                "fields": "modalities",
                "mapping_overrides": "canonical_adapter",
                "control_type": "control_schema",
                "format": "format_version",
            }
            for source_key, target_key in key_map.items():
                if source_key in child:
                    translated_child[target_key] = child[source_key]
            children.append(translated_child)
        legacy_sources.append(
            {
                "name": group.get("name", f"group_{index}"),
                "_source_file": group.get("category", group.get("name", f"group_{index}")),
                "enabled": group.get("enabled", True),
                "adapter": profile_name,
                "format_version": group.get("format"),
                "address": root,
                "datasets": children,
                "modalities": group.get("fields"),
                "canonical_adapter": group.get("mapping_overrides"),
                "control_schema": group.get("control_type"),
                "stats_group": group.get("stats_group"),
                "stats_group_template": group.get("stats_group_template"),
                "robot_id": robot_id,
            }
        )

    cache_templates = { 
        "local_meta_dir": storage.get("metadata_cache"),
        "path_index_dir": storage.get("path_index_cache"),
    }
    for key, value in tuple(cache_templates.items()):
        if not isinstance(value, str) or not value:
            cache_templates.pop(key, None)
            continue
        cache_templates[key] = _resolve_local_cache_template(value)

    category_weights = mixture.get("category_weights", {})
    if not isinstance(category_weights, dict):
        raise TypeError("Petrel YAML mixture.category_weights must be a mapping.")
    return {
        "dataset_id": dataset_id,
        "openpi": contract,
        "storage": {  # TODO
            "local_root": storage.get("local_root"),
            "parquet_cache_size": int(storage.get("parquet_cache_size", 8)),
        },
        "sampling": {
            "num_frames": int(sampling.get("state_frames", 1)),
            "action_size": int(sampling.get("action_horizon", contract["action_horizon"])),
            "global_sample_stride": int(sampling.get("stride", 1)),
        },
        "mixture": {
            "training": True,
            "seed": int(mixture.get("seed", 42)),
            "balance_dataset_weights": bool(mixture.get("balance_by_dataset", False)),
            "balance_trajectory_weights": bool(mixture.get("balance_by_trajectory", True)),
            "allow_padding_at_end": bool(sampling.get("allow_end_padding", False)),
            "training_block_size": int(sampling.get("training_block_size", 1)),
            "training_frame_stride": int(sampling.get("training_frame_stride", 1)),
        },
        "cache_templates": cache_templates,
        "dataset_defaults": {
            "dataset_weight": 1.0,
            "distribute_weights": bool(mixture.get("distribute_category_weight_by_valid_windows", True)),
            "dataset_stats_group": norm_grouping.get("dataset_group"),
            "dataset_stats_group_template": norm_grouping.get("dataset_group_template"),
            "dataset_id": dataset_id,
        },
        "dataset_weights": {"source_files": category_weights, "sources": {}, "datasets": {}, "embodiments": {}},
        "canonical_action_space": control_space,
        "source_control_space": control_space,
        "embodiments": legacy_profiles,  # number = embodimets numbers
        "sources": legacy_sources,   # number = sub_datasets numbsers  xxx-dataset/xxx-tasks number
        "_config_dir": str(config_dir),
    }


def _validate_openpi_petrel_metadata(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Validate and return only the global fields needed while creating DataConfig."""
    if payload.get("schema") != "openpi_petrel_v1":
        raise ValueError("Petrel YAML must declare schema: openpi_petrel_v1")
    dataset_id = payload.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("Petrel YAML dataset_id must be a non-empty string.")
    contract = payload.get("openpi")
    if not isinstance(contract, dict):
        raise TypeError("Petrel YAML openpi must be a mapping.")
    for key in ("state_dim", "action_dim", "action_horizon"):
        if int(contract.get(key, 0)) <= 0:
            raise ValueError(f"Petrel YAML openpi.{key} must be positive.")
    image_config = contract.get("images")
    if not isinstance(image_config, dict):
        raise TypeError("Petrel YAML openpi.images must be a mapping.")
    if image_config.get("slots") != ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
        raise ValueError("Petrel YAML openpi.images.slots must match the three pi0 camera slots.")
    if image_config.get("size") != [224, 224]:
        raise ValueError("Petrel YAML openpi.images.size must be [224, 224].")
    key_map = image_config.get("key_map", {})
    if not isinstance(key_map, dict):
        raise TypeError("Petrel YAML openpi.images.key_map must be a mapping.")
    invalid_targets = sorted({str(value) for value in key_map.values()} - set(image_config["slots"]))
    if invalid_targets:
        raise ValueError(f"Petrel YAML openpi.images.key_map contains invalid target slots: {invalid_targets}.")
    if any(not isinstance(key, str) or not key for key in key_map):
        raise ValueError("Petrel YAML openpi.images.key_map source keys must be non-empty strings.")
    if contract.get("normalization") not in {"quantile", "zscore"}:
        raise ValueError("Petrel YAML openpi.normalization must be quantile or zscore.")
    if contract.get("control_projection") != "mapped":
        raise ValueError("openpi_petrel_v1 control_projection must be 'mapped'.")
    return dataset_id, contract

def _resolve_openpi_petrel_path(config_path: str | Path) -> Path:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = OPENPI_REPO_ROOT / path
    return path.resolve()

def read_openpi_petrel_metadata(config_path: str | Path) -> tuple[str, dict[str, Any]]:
    """Read the global dataset id and OpenPI contract without scanning dataset-group YAMLs."""
    payload = _read_yaml_or_json(_resolve_openpi_petrel_path(config_path))
    return _validate_openpi_petrel_metadata(payload)

def read_openpi_petrel_config(config_path: str | Path) -> dict[str, Any]:
    """Read and validate an openpi_petrel_v1 YAML without accessing remote data."""
    path = _resolve_openpi_petrel_path(config_path)
    payload = _read_yaml_or_json(path)
    return _translate_openpi_petrel_config(payload, path.parent)


def load_dataset_specs(
    config_path: str | Path,
    *,
    local_root: str | None = None,
) -> tuple[dict[str, Any], list[DatasetSpec]]:
    """Validate one OpenPI YAML and expand it into local filesystem dataset specs."""
    config = read_openpi_petrel_config(config_path)
    specs = _flatten_source_specs(config)
    if not specs:
        raise ValueError(f"Dataset config expanded to no dataset specs: {config_path}")
    storage = config["storage"]
    configured_root = local_root or storage.get("local_root")
    if not configured_root:
        raise ValueError("Local storage requires local_root in YAML or the command-line override.")
    resolved_root = Path(str(configured_root)).expanduser().resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"Local dataset root does not exist or is not a directory: {resolved_root}")

    local_specs: list[DatasetSpec] = []
    missing_local_roots: list[Path] = []
    local_children: dict[Path, set[str]] = {}
    for spec in specs:
        relative_root = Path(spec.remote_root)
        if relative_root.is_absolute() or ".." in relative_root.parts:
            raise ValueError(
                f"Dataset root must be relative to storage.local_root, got {spec.remote_root!r}."
            )
        local_dataset_root = (resolved_root / relative_root).resolve()
        if not local_dataset_root.is_relative_to(resolved_root):
            raise ValueError(f"Dataset root escapes storage.local_root: {local_dataset_root}")
        parent = local_dataset_root.parent
        if parent not in local_children:
            try:
                local_children[parent] = {entry.name for entry in os.scandir(parent) if entry.is_dir()}
            except OSError:
                local_children[parent] = set()
        if local_dataset_root.name not in local_children[parent]:
            missing_local_roots.append(local_dataset_root)
            continue
        local_specs.append(
            replace(
                spec,
                remote_root=str(local_dataset_root),
                local_root=str(resolved_root),
                # Detect v2/v3 from each local dataset's own meta/info.json.
                format_version=None,
            )
        )
    if missing_local_roots:
        preview = ", ".join(str(path) for path in missing_local_roots[:5])
        raise FileNotFoundError(
            f"{len(missing_local_roots)}/{len(specs)} configured dataset directories are missing under "
            f"{resolved_root}; first missing paths: {preview}"
        )
    storage["local_root"] = str(resolved_root)
    return config, local_specs


def _get_nested(config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    value = config.get(section, {})
    if isinstance(value, dict) and key in value:
        return value[key]
    return default


def _resolved_arg(args_value: Any, config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return args_value if args_value is not None else _get_nested(config, section, key, default)


def _optional_str(value: Any, default: str | None = None) -> str | None:
    if value is None or value == "":
        return default
    return str(value)


def _parse_format_major(value: Any) -> int | None:
    if value in (None, ""):
        return None
    text = str(value).strip().lower().lstrip("v")
    if text.startswith("lerobot"):
        text = text.replace("lerobot", "").strip("_-v ")
    try:
        return int(text.split(".", 1)[0])
    except ValueError as exc:
        raise ValueError(f"Unsupported format_version={value!r}; use v2 or v3") from exc


def _path_segments_for_group(value: str) -> list[str]:
    text = str(value).replace("://", "/")
    return [part for part in text.replace("\\", "/").split("/") if part]


def _clean_oxe_group_name(name: str) -> str:
    text = str(name)
    for suffix in ("_converted_externally_to_rlds", "_lerobot"):
        text = text.replace(suffix, "")
    for version in ("_0.1.0", "_1.0.0", "_0.0.1"):
        text = text.replace(version, "")
    return _safe_name(text)


def _infer_stats_group(remote_root: str, dataset_name: str, embodiment: str, control_schema: str | None) -> str:
    segments = _path_segments_for_group(remote_root)
    lowered = [part.lower() for part in segments]
    remote_text = "/".join(lowered)

    if "egodex_sharpa_flow" in remote_text or "xperience_sharpa_flow" in remote_text:
        return "sharpa"

    for pos, part in enumerate(lowered):
        if "interndata-a1" in part:
            if pos + 2 < len(segments):
                return f"a1/{_safe_name(segments[pos + 2])}"
            return "a1/unknown"

    if any("interndata-m1" in part for part in lowered):
        for marker in ("simulated", "real"):
            if marker in lowered:
                pos = lowered.index(marker)
                if pos + 1 < len(segments):
                    return f"m1/{_safe_name(segments[pos + 1])}"
        return "m1"

    if "interndata-bridgev2" in lowered or "interndata-bridgev2" in remote_text:
        return "bridge/widowx"

    if "agibotworld" in lowered:
        return "agibot/a2d"

    if "galaxea" in lowered:
        return "galaxea/r1lite"

    if "oxe" in lowered:
        pos = lowered.index("oxe")
        if pos + 1 < len(segments):
            return f"oxe/{_clean_oxe_group_name(segments[pos + 1])}"
        return f"oxe/{_clean_oxe_group_name(dataset_name)}"

    emb = _safe_name(embodiment or "unknown")
    schema = _safe_name(control_schema or "")
    return f"{emb}/{schema}" if schema else emb


def _make_spec(
    item: dict[str, Any] | str,
    *,
    idx: int,
    group_id: int,
    inherited: dict[str, Any] | None = None,
) -> DatasetSpec:
    inherited = inherited or {}

    # item {"name": petrel/dataset_groups/xxx-dataset/xxx-task.yaml-datasets-name,
    #       "dir": petrel/dataset_groups/xxx-dataset/xxx-task.yaml-datasets-path}
    if isinstance(item, str):
        item = {"dir": item}
    
    remotes = inherited.get("_remotes", {}) if isinstance(inherited.get("_remotes"), dict) else {}  # {}
    remote_name = str(item.get("remote") or inherited.get("remote") or "")  # {}
    remote_base = item.get("remote_base", inherited.get("remote_base"))   # dataset_groups/xxx-dataset/xxx-task.yaml-root
    if not remote_base and remote_name:
        if remote_name not in remotes:
            raise KeyError(f"Unknown remote alias {remote_name!r}; available={sorted(remotes)}")
        remote_base = remotes[remote_name]

    base_path = str(item.get("base_path", inherited.get("base_path", ""))).strip("/")  # ""
    rel_path = item.get("dir", item.get("path", item.get("relative_path", "")))
    rel_path = str(rel_path).strip("/") if rel_path not in (None, "") else ""

    remote_root = item.get("remote_root", item.get("address", item.get("root")))  # {}
    if not remote_root and remote_base:
        parts = [str(remote_base)]
        if base_path:
            parts.append(base_path)
        if rel_path:
            parts.append(rel_path)
        remote_root = petrel_join(parts[0], *parts[1:])  # dataset_groups/xxx-dataset/xxx-task.yaml-(root+datasets/path)
    if not remote_root:
        raise KeyError(f"dataset spec {idx} is missing remote_root or remote/base_path/dir")

    dataset_path = "/".join(part for part in (base_path, rel_path) if part)
    dataset_name = str(
        item.get("name")
        or _safe_name(dataset_path or str(remote_root).rstrip("/").rsplit("/", 1)[-1])
        or f"dataset_{idx}"
    )
    embodiment = str(
        item.get("adapter")
        or item.get("embodiment")
        or inherited.get("adapter")
        or inherited.get("embodiment")
        or "default"
    )
    dataset_id = str(inherited.get("dataset_id") or "dataset")
    robot_id = str(item.get("robot_id") or inherited.get("robot_id") or embodiment)

    stats_value = item.get("stats_path", inherited.get("stats_path", "dataset_stats.json"))
    local_stats_value = item.get("local_stats_path", inherited.get("local_stats_path"))
    local_meta_value = item.get("local_meta_dir", inherited.get("local_meta_dir"))
    path_index_value = item.get("path_index_dir", inherited.get("path_index_dir"))
    local_text_cache_value = item.get(
        "local_text_embedding_cache_dir",
        item.get(
            "text_embedding_cache_dir",
            inherited.get("local_text_embedding_cache_dir", inherited.get("text_embedding_cache_dir")),
        ),
    )

    source_name = inherited.get("source_name") or inherited.get("_source_file") or remote_name
    relative_remote = _relative_dataset_cache_path(str(remote_root))
    local_meta_value = local_meta_value or _format_template(
        inherited.get("local_meta_dir_template"),
        name=dataset_name,
        path=dataset_path,
        remote=str(remote_root),
        source=source_name,
        relative_remote=relative_remote,
        embodiment=embodiment,
        dataset_id=dataset_id,
        robot_id=robot_id,
    )  # 
    local_stats_value = local_stats_value or _format_template(
        inherited.get("local_stats_path_template"),
        name=dataset_name,
        path=dataset_path,
        remote=str(remote_root),
        source=source_name,
        relative_remote=relative_remote,
        embodiment=embodiment,
        dataset_id=dataset_id,
        robot_id=robot_id,
    )  # None
    path_index_value = path_index_value or _format_template(
        inherited.get("path_index_dir_template"),  
        name=dataset_name,
        path=dataset_path,
        remote=str(remote_root),
        source=source_name,
        relative_remote=relative_remote,
        embodiment=embodiment,
        dataset_id=dataset_id,
        robot_id=robot_id,
    )  # data/petrel_cache/path_index/{dataset_id}/{remote}
    local_text_cache_value = local_text_cache_value or _format_template(
        inherited.get("local_text_embedding_cache_dir_template"),
        name=dataset_name,
        path=dataset_path,
        remote=str(remote_root),
        source=source_name,
        relative_remote=relative_remote,
        embodiment=embodiment,
        dataset_id=dataset_id,
        robot_id=robot_id,
    )  # None

    dataset_weight = float(item.get("dataset_weight", inherited.get("dataset_weight", 1.0)))
    distribute_weights = bool(item.get("distribute_weights", inherited.get("distribute_weights", False)))
    dataset_override = _lookup_weight_override(
        inherited.get("dataset_weight_overrides"),
        [dataset_name, dataset_path, rel_path, remote_root],
    )
    dataset_weight, distribute_weights = _apply_weight_override(dataset_weight, distribute_weights, dataset_override)

    modalities = _merge_mapping(inherited.get("modalities"), item.get("modalities"))
    control_schema = _optional_str(item.get("control_schema", inherited.get("control_schema")))
    stats_group_template = item.get("stats_group_template", inherited.get("stats_group_template"))
    stats_group = item.get("stats_group", inherited.get("stats_group"))
    if not stats_group and stats_group_template:
        body = _infer_stats_group(str(remote_root), dataset_name, embodiment, control_schema).split("/", 1)[-1]
        stats_group = _format_template(
            stats_group_template,
            name=dataset_name,
            path=dataset_path,
            remote=str(remote_root),
            source=source_name,
            relative_remote=relative_remote,
            embodiment=embodiment,
            body=body,
            control_schema=control_schema or "",
            dataset_id=dataset_id,
            robot_id=robot_id,
        )
    stats_group_fallbacks: list[str] = []
    for fallback in inherited.get("stats_group_fallbacks", ()):
        if fallback and str(fallback) != str(stats_group) and str(fallback) not in stats_group_fallbacks:
            stats_group_fallbacks.append(str(fallback))
    for fallback_template in inherited.get("stats_group_fallback_templates", ()):
        fallback = _format_template(
            fallback_template,
            name=dataset_name,
            path=dataset_path,
            remote=str(remote_root),
            source=source_name,
            relative_remote=relative_remote,
            embodiment=embodiment,
            control_schema=control_schema or "",
            dataset_id=dataset_id,
            robot_id=robot_id,
        )
        if fallback and fallback != stats_group and fallback not in stats_group_fallbacks:
            stats_group_fallbacks.append(fallback)
    if not stats_group and stats_group_fallbacks:
        stats_group = stats_group_fallbacks.pop(0)
    if not stats_group:
        stats_group = _infer_stats_group(str(remote_root), dataset_name, embodiment, control_schema)

    return DatasetSpec(
        name=dataset_name,  # petrel/dataset_groups/xxx-dataset/xxx-task.yaml-datasets-name
        remote_root=str(remote_root), # dataset_groups/xxx-dataset/xxx-task.yaml-(root+datasets/name)
        stats_path=_optional_str(stats_value),  # dataset_stats.json
        local_stats_path=_optional_str(local_stats_value),  # None
        local_meta_dir=_optional_str(local_meta_value),  # data/petrel_cache/meta/{dataset_id}/{remote}
        path_index_dir=_optional_str(path_index_value),  # data/petrel_cache/path_index/{dataset_id}/{remote}
        local_text_embedding_cache_dir=_optional_str(local_text_cache_value),  # None
        context_len=int(item.get("context_len", inherited.get("context_len", 128))), # 128
        text_encoder_id=None, # str(item.get("text_encoder_id", inherited.get("text_encoder_id", "wan22ti2v5b"))),
        # /petrel/(dataset_groups/xxx-dataset/xxx.yaml + control_profiles/xxx.yaml)-(mapping+mapping_overrides)
        modalities=modalities, 
        # /petrel/(dataset_groups/xxx-dataset/xxx.yaml + control_profiles/xxx.yaml)-(control_profiles/fields+fields)
        canonical_adapter=_merge_mapping(inherited.get("canonical_adapter"), item.get("canonical_adapter")),
        format_version=_parse_format_major(
            item.get("format_version", item.get("format", inherited.get("format_version", inherited.get("format"))))
        ), 
        embodiment=embodiment, # #  dataset_groups/xxx-dataset/xxx-task.yaml-profile
        control_schema=control_schema,  # control_type
        stats_group=_optional_str(stats_group),  # dataset_groups/xxx-dataset/xxx-task.yaml-stats_group or stats_group_template
        stats_group_fallbacks=tuple(stats_group_fallbacks),  # control_profiles/xxx.yaml or pi05_xxx.yaml-stats_group or stats_group_template
        source_name=_optional_str(source_name),  #  dataset_groups/xxx-dataset/xxx-task.yaml-name
        dataset_weight=dataset_weight,
        distribute_weights=distribute_weights, # True
        action_target_dim=item.get("action_target_dim", inherited.get("action_target_dim")), # None
        state_target_dim=item.get("state_target_dim", inherited.get("state_target_dim")),  # None
        group_id=int(group_id),
    )


def _resolve_config_path(path: str | Path, config_dir: str | None) -> Path:
    out = Path(path).expanduser()
    if not out.is_absolute() and config_dir:
        out = Path(config_dir) / out
    return out


def _read_dataset_entries_file(path: str | Path, config_dir: str | None) -> list[dict[str, Any] | str]:
    file_path = _resolve_config_path(path, config_dir)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    suffix = file_path.suffix.lower()
    if suffix in {".yaml", ".yml", ".json"}:
        if suffix == ".json":
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        else:
            import yaml

            payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("datasets", payload.get("paths", payload.get("items")))
        if not isinstance(payload, list):
            raise TypeError(f"Dataset list file must contain a list: {file_path}")
        return payload

    entries: list[dict[str, Any] | str] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def _dataset_entries_from_group(group: dict[str, Any], config_dir: str | None) -> list[dict[str, Any] | str]:
    entries: list[dict[str, Any] | str] = []
    files = group.get("datasets_files", group.get("datasets_file"))
    if files is not None:
        if isinstance(files, (str, Path)):
            files = [files]
        if not isinstance(files, list):
            raise TypeError("datasets_file/datasets_files must be a path or list of paths")
        for file_path in files:
            entries.extend(_read_dataset_entries_file(file_path, config_dir))

    inline = group.get("datasets")
    if inline is not None:
        if not isinstance(inline, list):
            raise TypeError("datasets must be a list")
        entries.extend(inline)

    return entries or [group]


def _flatten_source_specs(config: dict[str, Any]) -> list[DatasetSpec] | None:
    sources = config.get("sources")
    if not isinstance(sources, list):
        return None

    cache_templates = config.get("cache_templates", {}) if isinstance(config.get("cache_templates"), dict) else {}
    dataset_defaults = config.get("dataset_defaults", {}) if isinstance(config.get("dataset_defaults"), dict) else {}
    weight_config = config.get("dataset_weights", {}) if isinstance(config.get("dataset_weights"), dict) else {}
    default_dataset_weight = float(dataset_defaults.get("dataset_weight", config.get("dataset_weight", 1.0)))
    default_distribute_weights = bool(
        dataset_defaults.get("distribute_weights", config.get("distribute_weights", False))
    )
    source_weight_overrides = weight_config.get("sources", {}) if isinstance(weight_config, dict) else {}
    source_file_weight_overrides = weight_config.get("source_files", {}) if isinstance(weight_config, dict) else {}
    embodiment_weight_overrides = weight_config.get("embodiments", {}) if isinstance(weight_config, dict) else {}
    dataset_weight_overrides = weight_config.get("datasets", {}) if isinstance(weight_config, dict) else {}
    embodiment_defaults = config.get("embodiments", {}) if isinstance(config.get("embodiments"), dict) else {}
    config_dir = config.get("_config_dir")
    specs: list[DatasetSpec] = []
    source_file_group_ids: dict[str, int] = {}
    next_group_id = 0

    for source_idx, source in enumerate(sources):
        if not isinstance(source, dict):
            raise TypeError(f"sources[{source_idx}] must be a mapping")
        if source.get("enabled", True) is False:
            continue
        source_file = str(source.get("_source_file") or source.get("name") or f"source_{source_idx}")
        source_file_override = _lookup_weight_override(source_file_weight_overrides, [source_file])
        if source_file_override is not None:
            if source_file not in source_file_group_ids:
                source_file_group_ids[source_file] = next_group_id
                next_group_id += 1
            group_id = source_file_group_ids[source_file]
        else:
            group_id = next_group_id
            next_group_id += 1
        embodiment = str(source.get("adapter", source.get("embodiment", "default")))
        emb_cfg = embodiment_defaults.get(embodiment, {})
        if emb_cfg is None:
            emb_cfg = {}
        if not isinstance(emb_cfg, dict):
            raise TypeError(f"embodiments.{embodiment} must be a mapping")
        address = source.get("address", source.get("root", source.get("remote_root")))
        source_name = str(source.get("name") or source.get("_source_file") or "")
        base_source_weight = float(source.get("dataset_weight", emb_cfg.get("dataset_weight", default_dataset_weight)))
        base_source_distribute = bool(
            source.get("distribute_weights", emb_cfg.get("distribute_weights", default_distribute_weights))
        )
        if source_file_override is not None:
            source_weight, _ = _apply_weight_override(
                base_source_weight,
                base_source_distribute,
                source_file_override,
            )
            source_distribute = True
        else:
            source_weight, source_distribute = _apply_weight_override(
                base_source_weight,
                base_source_distribute,
                _lookup_weight_override(
                    source_weight_overrides, [source_name, source.get("_source_file"), address, embodiment]
                ),
            )
            source_weight, source_distribute = _apply_weight_override(
                source_weight,
                source_distribute,
                _lookup_weight_override(embodiment_weight_overrides, [embodiment]),
            )
        source_modalities = _merge_mapping(emb_cfg.get("modalities"), source.get("modalities"))
        source_canonical_adapter = _merge_mapping(emb_cfg.get("canonical_adapter"), source.get("canonical_adapter"))
        inherited = {
            "_source_file": source.get("_source_file"),  # category
            "source_name": source_name,  #  dataset_groups/xxx-dataset/xxx-task.yaml-name
            "remote": source.get("remote", emb_cfg.get("remote")),  #  {}
            #  dataset_groups/xxx-dataset/xxx-task.yaml-root
            "remote_base": source.get("remote_base", address if address is not None else emb_cfg.get("remote_base")),
            "base_path": source.get("base_path", emb_cfg.get("base_path", "")),  # ""
            "stats_path": source.get("stats_path", emb_cfg.get("stats_path", "dataset_stats.json")),  # "dataset_stats.json"
            "local_text_embedding_cache_dir": source.get(
                "local_text_embedding_cache_dir",
                source.get(
                    "text_embedding_cache_dir",
                    emb_cfg.get("local_text_embedding_cache_dir", emb_cfg.get("text_embedding_cache_dir")),
                ),
            ), # None
            "local_meta_dir_template": source.get("local_meta_dir_template")
            or cache_templates.get("local_meta_dir"),  # openpi/configs/petrel/pi05_xxx.yaml-storage-metadata_cache
            "path_index_dir_template": source.get("path_index_dir_template")
            or cache_templates.get("path_index_dir"),  # openpi/configs/petrel/pi05_xxx.yaml-storage-path_index_cache
            "local_text_embedding_cache_dir_template": source.get(
                "local_text_embedding_cache_dir_template",
                emb_cfg.get(
                    "local_text_embedding_cache_dir_template", cache_templates.get("local_text_embedding_cache_dir")
                ),
            ),  # None
            "context_len": source.get("context_len", emb_cfg.get("context_len", 128)),  # 128
            "modalities": source_modalities,   # /petrel/(dataset_groups/xxx-dataset/xxx.yaml + control_profiles/xxx.yaml)-(mapping+mapping_overrides)
            "canonical_adapter": source_canonical_adapter,  # /petrel/(dataset_groups/xxx-dataset/xxx.yaml + control_profiles/xxx.yaml)-(fields+fields)
            "format_version": source.get(
                "format_version", source.get("format", emb_cfg.get("format_version", emb_cfg.get("format")))
            ),  # dataset_groups/xxx-dataset/xxx-task.yaml-format
            "embodiment": embodiment,  #  dataset_groups/xxx-dataset/xxx-task.yaml-profile
            "control_schema": source.get("control_schema") or emb_cfg.get("control_schema"),
            "dataset_id": dataset_defaults.get("dataset_id", config.get("dataset_id")),
            "robot_id": source.get("robot_id"),
            "stats_group": source.get("stats_group"),  # dataset_groups/xxx-dataset/xxx-task.yaml-stats_group
            "stats_group_template": source.get("stats_group_template"),  # dataset group task-level norm
            "stats_group_fallbacks": tuple(
                value
                for value in (
                    emb_cfg.get("stats_group"),
                    dataset_defaults.get("dataset_stats_group"),
                )
                if value
            ),
            "stats_group_fallback_templates": tuple(
                value
                for value in (
                    emb_cfg.get("stats_group_template"),
                    dataset_defaults.get("dataset_stats_group_template"),
                )
                if value
            ),
            "dataset_weight": source_weight,  # dataset weight
            "distribute_weights": source_distribute,  
            "dataset_weight_overrides": dataset_weight_overrides,  # {}
            "action_target_dim": source.get("action_target_dim", emb_cfg.get("action_target_dim")),  # None
            "state_target_dim": source.get("state_target_dim", emb_cfg.get("state_target_dim")),  # None
        }
        children = _dataset_entries_from_group(source, config_dir)
        for child in children:
            if not isinstance(child, (dict, str)):
                raise TypeError(f"sources[{source_idx}] dataset entry must be a mapping or string")
            if isinstance(child, dict) and child.get("enabled", True) is False:
                continue
            specs.append(_make_spec(child, idx=len(specs), group_id=group_id, inherited=inherited))
    return specs


def distributed_dataset_weights(
    specs: list[DatasetSpec], datasets: list[PetrelLeRobotDataset], *, allow_padding_at_end: bool
) -> list[float]:
    """Compute mixture weights, distributing group weights by valid window count."""
    return _distributed_dataset_weights(specs, datasets, allow_padding_at_end=allow_padding_at_end)


def decode_mp4_bytes(raw: bytes, timestamps: list[float]) -> torch.Tensor:
    """Decode selected timestamps from remote MP4 bytes."""
    return _decode_mp4_bytes_lerobot(raw, timestamps)


def _summarize_item(item: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            summary[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        elif isinstance(value, str):
            summary[key] = value[:200]
        else:
            summary[key] = str(type(value).__name__)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-config", type=Path, required=True, help="YAML config describing local LeRobot datasets."
    )
    parser.add_argument("--local-root", default=None, help="Override storage.local_root from the YAML.")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--action-size", type=int, default=None)
    parser.add_argument("--global-sample-stride", type=int, default=None)
    parser.add_argument("--parquet-cache-size", type=int, default=None)
    parser.add_argument("--with-video", action="store_true")
    parser.add_argument(
        "--load-stats",
        action="store_true",
        help="Also read each configured dataset_stats.json and print its top-level keys.",
    )
    parser.add_argument(
        "--epoch", type=int, default=0, help="Epoch used by DreamZero-style deterministic training sampling."
    )
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        help="Force deterministic eval subset mode instead of training random sampling.",
    )
    parser.add_argument("--eval-episodes-per-dataset", type=int, default=None)
    parser.add_argument("--eval-frames-per-episode", type=int, default=None)
    parser.add_argument("--eval-frame-policy", choices=["first", "middle", "last", "uniform", "random"], default=None)
    parser.add_argument(
        "--eval-scope", choices=["dataset", "source", "source_type", "type", "source_entry"], default=None
    )
    parser.add_argument("--eval-segments-per-source", type=int, default=None)
    parser.add_argument("--eval-total-samples", type=int, default=None)
    parser.add_argument("--eval-fixed-samples-json", default=None, help="JSON string/path for fixed eval samples.")
    parser.add_argument(
        "--eval-fixed-per-dataset", action="store_true", help="Use one fixed eval sample from every expanded dataset."
    )
    parser.add_argument("--eval-exclude-from-training", choices=["none", "sample", "episode"], default=None)
    parser.add_argument(
        "--profile", action="store_true", help="Print coarse timing breakdown for local loading."
    )
    parser.add_argument("--save-summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    global _PROFILE_ENABLED
    args = parse_args()
    _PROFILE_ENABLED = bool(args.profile)
    t_main = time.perf_counter()
    config, specs = load_dataset_specs(args.dataset_config, local_root=args.local_root)
    num_frames = int(_resolved_arg(args.num_frames, config, "sampling", "num_frames", 33))
    action_size = _resolved_arg(args.action_size, config, "sampling", "action_size", None)
    if action_size is not None:
        action_size = int(action_size)
    global_sample_stride = int(_resolved_arg(args.global_sample_stride, config, "sampling", "global_sample_stride", 1))
    parquet_cache_size = int(_resolved_arg(args.parquet_cache_size, config, "storage", "parquet_cache_size", 8))

    t0 = time.perf_counter()
    datasets = [
        PetrelLeRobotDataset(
            spec,
            num_frames=num_frames,
            action_size=action_size,
            global_sample_stride=global_sample_stride,
            parquet_cache_size=parquet_cache_size,
        )
        for spec in specs
    ]
    _profile_record("main.build_datasets", time.perf_counter() - t0, num_datasets=len(datasets))
    if args.load_stats:
        for ds in datasets:
            t0 = time.perf_counter()
            stats = ds.load_dataset_stats()
            _profile_record("main.load_stats", time.perf_counter() - t0, dataset=ds.name)
            print(f"[stats] dataset={ds.name} stats_path={ds.spec.stats_path} keys={list(stats.keys())}", flush=True)

    mixture_cfg = config.get("mixture", {}) if isinstance(config.get("mixture"), dict) else {}
    eval_cfg = mixture_cfg.get("eval_subset", mixture_cfg.get("eval", {}))
    if not isinstance(eval_cfg, dict):
        eval_cfg = {}
    allow_padding_at_end = bool(mixture_cfg.get("allow_padding_at_end", False))
    dataset_weights = _distributed_dataset_weights(specs, datasets, allow_padding_at_end=allow_padding_at_end)
    epoch_length = mixture_cfg.get("epoch_length")
    training = bool(mixture_cfg.get("training", True)) and not bool(args.eval_mode)
    eval_episodes_per_dataset = args.eval_episodes_per_dataset
    if eval_episodes_per_dataset is None:
        eval_episodes_per_dataset = eval_cfg.get("episodes_per_dataset")
    eval_frames_per_episode = args.eval_frames_per_episode
    if eval_frames_per_episode is None:
        eval_frames_per_episode = eval_cfg.get("frames_per_episode")
    eval_frame_policy = args.eval_frame_policy or str(eval_cfg.get("frame_policy", "first"))
    eval_scope = args.eval_scope or str(eval_cfg.get("scope", "dataset"))
    eval_fixed_samples = eval_cfg.get("fixed_samples")
    if args.eval_fixed_samples_json:
        fixed_text = args.eval_fixed_samples_json
        fixed_path = Path(fixed_text)
        if fixed_path.exists():
            fixed_text = fixed_path.read_text(encoding="utf-8")
        eval_fixed_samples = json.loads(fixed_text)
    eval_fixed_per_dataset = bool(args.eval_fixed_per_dataset or eval_cfg.get("fixed_per_dataset", False))
    eval_fixed_episode_pos = eval_cfg.get("episode_pos", 0)
    eval_fixed_frame_index = int(eval_cfg.get("frame_index", 0))
    eval_exclude_from_training = args.eval_exclude_from_training
    if eval_exclude_from_training is None:
        eval_exclude_from_training = eval_cfg.get("exclude_from_training")
    eval_segments_per_source = args.eval_segments_per_source
    if eval_segments_per_source is None:
        eval_segments_per_source = eval_cfg.get("segments_per_source")
    eval_total_samples = args.eval_total_samples
    if eval_total_samples is None:
        eval_total_samples = eval_cfg.get("total_samples", eval_cfg.get("eval_total_samples"))
    dataset = MixedPetrelLeRobotDataset(
        datasets,
        dataset_weights=dataset_weights,
        training=training,
        balance_dataset_weights=bool(mixture_cfg.get("balance_dataset_weights", False)),
        balance_trajectory_weights=bool(mixture_cfg.get("balance_trajectory_weights", True)),
        seed=int(mixture_cfg.get("seed", 42)),
        allow_padding_at_end=allow_padding_at_end,
        epoch_length=None if epoch_length in (None, "") else int(epoch_length),
        eval_episodes_per_dataset=None if eval_episodes_per_dataset in (None, "") else int(eval_episodes_per_dataset),
        eval_frames_per_episode=None if eval_frames_per_episode in (None, "") else int(eval_frames_per_episode),
        eval_frame_policy=eval_frame_policy,
        eval_scope=eval_scope,
        eval_segments_per_source=None if eval_segments_per_source in (None, "") else int(eval_segments_per_source),
        eval_total_samples=None if eval_total_samples in (None, "") else int(eval_total_samples),
        eval_fixed_samples=eval_fixed_samples if isinstance(eval_fixed_samples, list) else None,
        eval_fixed_per_dataset=eval_fixed_per_dataset,
        eval_fixed_episode_pos=None if eval_fixed_episode_pos in (None, "") else int(eval_fixed_episode_pos),
        eval_fixed_frame_index=eval_fixed_frame_index,
        eval_exclude_from_training=eval_exclude_from_training,
    )
    dataset.set_epoch(args.epoch)
    t0 = time.perf_counter()
    item = dataset[args.index]
    _profile_record("main.get_item", time.perf_counter() - t0, index=args.index)
    summary = _summarize_item(item)
    payload = {
        "num_datasets": len(datasets),
        "length": len(dataset),
        "dataset_sampling_weights": dataset.dataset_sampling_weights.tolist(),
        "item": summary,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)

    if args.with_video:
        t0 = time.perf_counter()
        video = dataset.get_video_for_item(item)
        _profile_record("main.get_video", time.perf_counter() - t0)
        print(
            f"[video] shape={tuple(video.shape)} dtype={video.dtype} range=({video.min().item():.4f},{video.max().item():.4f})",
            flush=True,
        )
        summary["video"] = {"shape": list(video.shape), "dtype": str(video.dtype)}

    if args.save_summary is not None:
        args.save_summary.parent.mkdir(parents=True, exist_ok=True)
        args.save_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] wrote {args.save_summary}", flush=True)

    _profile_record("main.total", time.perf_counter() - t_main)
    _profile_print()


if __name__ == "__main__":
    main()
