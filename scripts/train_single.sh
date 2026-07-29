#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENPI_WORKDIR="${OPENPI_WORKDIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
export OPENPI_RUN_ROOT="${OPENPI_RUN_ROOT:-${OPENPI_WORKDIR}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OPENPI_NUM_WORKERS="${OPENPI_NUM_WORKERS:-16}"
export OPENPI_OVERWRITE="${OPENPI_OVERWRITE:-0}"
export OPENPI_LOG_RANK_PHASES="${OPENPI_LOG_RANK_PHASES:-1}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

normalize_bool_env() {
  local variable_name="$1"
  local default_value="$2"
  local value="${!variable_name:-${default_value}}"
  case "${value,,}" in
    1|true|yes|on) printf -v "${variable_name}" '%s' 1 ;;
    0|false|no|off) printf -v "${variable_name}" '%s' 0 ;;
    *)
      echo "${variable_name} must be a boolean value, got: ${value}" >&2
      exit 2
      ;;
  esac
  export "${variable_name}"
}

normalize_bool_env OPENPI_PROFILE 0
normalize_bool_env OPENPI_TORCH_TRACE 0

if [[ "${OPENPI_PROFILE}" == "1" ]]; then
  TRAIN_MODE="profiling"
  export OPENPI_TRAIN_STEPS="${OPENPI_TRAIN_STEPS:-120}"
else
  TRAIN_MODE="training"
  export OPENPI_TRAIN_STEPS="${OPENPI_TRAIN_STEPS:-10000}"
  if [[ "${OPENPI_TORCH_TRACE}" == "1" ]]; then
    echo "OPENPI_TORCH_TRACE requires OPENPI_PROFILE=1." >&2
    exit 2
  fi
fi

if [[ -z "${OPENPI_RUN_ID:-}" ]]; then
  export OPENPI_RUN_ID="single_$(date -u +%Y%m%d_%H%M%S)"
  echo "OPENPI_RUN_ID was not set; using ${OPENPI_RUN_ID}" >&2
fi
if [[ ! "${OPENPI_RUN_ID}" =~ ^[[:alnum:]][[:alnum:]_.-]*$ ]]; then
  echo "OPENPI_RUN_ID must start with an alphanumeric character and contain only letters, numbers, '.', '_' or '-'." >&2
  exit 2
fi
if [[ ! "${OPENPI_TRAIN_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "OPENPI_TRAIN_STEPS must be a positive integer, got: ${OPENPI_TRAIN_STEPS}" >&2
  exit 2
fi
if [[ ! "${OPENPI_NUM_WORKERS}" =~ ^[0-9]+$ ]]; then
  echo "OPENPI_NUM_WORKERS must be a non-negative integer, got: ${OPENPI_NUM_WORKERS}" >&2
  exit 2
fi

IFS=',' read -r -a VISIBLE_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#VISIBLE_DEVICES[@]} != 1 )); then
  echo "Single-GPU training requires exactly one CUDA_VISIBLE_DEVICES entry, got: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

LOG_ROOT="${OPENPI_LOG_ROOT:-${OPENPI_RUN_ROOT}/runs/distributed_logs}"
LOG_DIR="${LOG_ROOT}/${OPENPI_RUN_ID}/node_0"
mkdir -p "${LOG_DIR}"
LAUNCHER_LOG_FILE="${LOG_DIR}/launcher.log"

exec 3>&1 4>&2
exec >>"${LAUNCHER_LOG_FILE}" 2>&1
trap 'rc=$?; message="[launcher][error] timestamp=$(date --iso-8601=seconds) line=${LINENO} exit=${rc} command=${BASH_COMMAND}"; echo "${message}" >&2; echo "${message}" >&4' ERR
echo "[launcher-bootstrap] timestamp=$(date --iso-8601=seconds) host=$(hostname) pid=$$ log=${LAUNCHER_LOG_FILE}" >&3

cd "${OPENPI_WORKDIR}"
if [[ -n "${OPENPI_PYTHON:-}" ]]; then
  PYTHON_BIN="${OPENPI_PYTHON}"
elif [[ -x "${OPENPI_WORKDIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${OPENPI_WORKDIR}/.venv/bin/python"
else
  PYTHON_BIN="python"
fi

echo "[launcher] timestamp=$(date --iso-8601=seconds) host=$(hostname) pid=$$"
echo "[launcher] workdir=${OPENPI_WORKDIR} python=${PYTHON_BIN} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[train] run_id=${OPENPI_RUN_ID} mode=${TRAIN_MODE} steps=${OPENPI_TRAIN_STEPS} workers=${OPENPI_NUM_WORKERS}"
echo "[train] run_dir_template=${OPENPI_RUN_ROOT}/pi05_interna1/${OPENPI_RUN_ID}_<UTC start time>"
echo "[logging] launcher_log=${LAUNCHER_LOG_FILE}"

if [[ "${OPENPI_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
  printf '[launcher] command: %q %q\n' "${PYTHON_BIN}" scripts/train_single.py
  echo "[launcher] dry run complete; training was not started"
  exit 0
fi

exec "${PYTHON_BIN}" scripts/train_single.py
