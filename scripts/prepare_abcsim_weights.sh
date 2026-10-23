#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Cache pi05 base weights locally. Override OPENPI_BASE_WEIGHT_SOURCE or OPENPI_WEIGHT_PATH as needed."
    exit 0
fi
export OPENPI_WORKDIR="${OPENPI_WORKDIR:-/sunyiming/openpi}"
export OPENPI_BASE_WEIGHT_SOURCE="${OPENPI_BASE_WEIGHT_SOURCE:-/dataset/public/model/pi05_base_pytorch_bf16}"
export OPENPI_WEIGHT_PATH="${OPENPI_WEIGHT_PATH:-${OPENPI_WORKDIR}/data/models/pi05_base_pytorch_bf16}"
"${OPENPI_WORKDIR}/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path
import shutil

source = Path(os.environ["OPENPI_BASE_WEIGHT_SOURCE"])
target = Path(os.environ["OPENPI_WEIGHT_PATH"])
if source.resolve() == target.resolve():
    raise ValueError("Source and cache directories must be different.")
target.mkdir(parents=True, exist_ok=True)
for name in ("model.safetensors", "config.json"):
    src, dst = source / name, target / name
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.is_file():
        if dst.stat().st_size != src.stat().st_size:
            raise ValueError(f"Existing cache size differs from source: {dst}")
        print(f"[weights] cache exists: {dst}", flush=True)
        continue
    temporary = dst.with_name(dst.name + ".partial")
    print(f"[weights] copying {src} -> {dst}", flush=True)
    with src.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
    if temporary.stat().st_size != src.stat().st_size:
        raise ValueError(f"Incomplete weight copy: {temporary}")
    temporary.replace(dst)
(target / "source.json").write_text(json.dumps({"source": str(source), "weights_bytes": (target / "model.safetensors").stat().st_size}, indent=2) + "\n")
print(f"[weights] ready: {target}", flush=True)
PY
