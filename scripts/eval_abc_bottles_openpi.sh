#!/usr/bin/env bash
set -Eeuo pipefail

OPENPI_WORKDIR="${OPENPI_WORKDIR:-/sunyiming/openpi}"
ABC_ROOT="${ABC_ROOT:-/sunyiming/abc}"
cd "${OPENPI_WORKDIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${OPENPI_WORKDIR}/data/openpi_assets}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-14}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-14}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# ABC uses its MuJoCo 3.8 environment. OpenPI remains in its original environment.
test -x "${ABC_ROOT}/.venv/bin/python"
test -x "${OPENPI_WORKDIR}/.venv/bin/python"
exec "${ABC_ROOT}/.venv/bin/python" -u scripts/eval_abc_bottles_openpi.py \
    --abc-root "${ABC_ROOT}" \
    --openpi-python "${OPENPI_WORKDIR}/.venv/bin/python" \
    "$@"
