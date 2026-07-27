#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENPI_WORKDIR="${OPENPI_WORKDIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
export OPENPI_RUN_ROOT="${OPENPI_RUN_ROOT:-${OPENPI_WORKDIR}/runs}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENPI_PROFILE=1
export OPENPI_RUN_ID=profile_w0
export OPENPI_NUM_WORKERS=0
export OPENPI_TRAIN_STEPS=120
# Profiling is opt-in. Example:
# OPENPI_PROFILE=1 OPENPI_TORCH_TRACE=1 OPENPI_NUM_WORKERS=0 scripts/train_single.sh

cd "${OPENPI_WORKDIR}"

if [[ -n "${OPENPI_PYTHON:-}" ]]; then
  PYTHON_BIN="${OPENPI_PYTHON}"
elif [[ -x "${OPENPI_WORKDIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${OPENPI_WORKDIR}/.venv/bin/python"
else
  PYTHON_BIN="python"
fi

exec "${PYTHON_BIN}" scripts/train_single.py
