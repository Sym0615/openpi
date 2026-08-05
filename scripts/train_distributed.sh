#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENPI_WORKDIR="${OPENPI_WORKDIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
export OPENPI_RUN_ROOT="${OPENPI_RUN_ROOT:-${OPENPI_WORKDIR}}"
export PYTHONPATH="${OPENPI_WORKDIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CHECKPINTS_ROOT="${OPENPI_RUN_ROOT}/checkpoints"

# Multi-node SSP jobs inject WORLD_SIZE and rendezvous variables. A single-node
# GPU job may inject none of them, so default that case to one local node.
export WORLD_SIZE="${WORLD_SIZE:-${OPENPI_NUM_NODES:-1}}"
if [[ "${WORLD_SIZE}" == "1" ]]; then
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-29500}"
fi

resolve_node_rank() {
  # SCO_POD_NAME uses role names such as master-0 or worker-6. When it is not
  # injected, SSP HOSTNAME may append a replica suffix, for example worker-6-0.
  local pod_name="${SCO_POD_NAME:-${HOSTNAME:-$(hostname)}}"
  if [[ "${WORLD_SIZE}" == "1" ]]; then
    printf '0'
  elif [[ "${pod_name}" == *master-0* ]]; then
    printf '0'
  elif [[ "${pod_name}" =~ worker-([0-9]+)-[0-9]+$ ]]; then
    printf '%s' "$((BASH_REMATCH[1] + 1))"
  elif [[ "${pod_name}" =~ worker-([0-9]+)$ ]]; then
    printf '%s' "$((BASH_REMATCH[1] + 1))"
  else
    printf 'unknown'
  fi
}

BOOTSTRAP_NODE_RANK="$(resolve_node_rank)"
BOOTSTRAP_RUN_SOURCE="${OPENPI_RUN_ID:-interna1_${MASTER_ADDR:-unknown}_${MASTER_PORT:-unknown}}"
BOOTSTRAP_RUN_ID="$(printf '%s' "${BOOTSTRAP_RUN_SOURCE}" | tr -cs '[:alnum:]_.-' '_')"
LOG_ROOT="${OPENPI_LOG_ROOT:-${OPENPI_RUN_ROOT}/runs/distributed_logs}"
BOOTSTRAP_LOG_DIR="${LOG_ROOT}/${BOOTSTRAP_RUN_ID}/node_${BOOTSTRAP_NODE_RANK}"
mkdir -p "${BOOTSTRAP_LOG_DIR}"
LAUNCHER_LOG_FILE="${BOOTSTRAP_LOG_DIR}/launcher.log"

exec 3>&1 4>&2
exec >>"${LAUNCHER_LOG_FILE}" 2>&1

trap 'rc=$?; message="[launcher][error] timestamp=$(date --iso-8601=seconds) line=${LINENO} exit=${rc} command=${BASH_COMMAND}"; echo "${message}" >&2; echo "${message}" >&4' ERR
echo "[launcher-bootstrap] timestamp=$(date --iso-8601=seconds) host=$(hostname) pid=$$ log=${LAUNCHER_LOG_FILE}"
echo "[launcher-bootstrap] pod=${SCO_POD_NAME:-unset} host=$(hostname) raw_rank=${RANK:-unset} resolved_node_rank=${BOOTSTRAP_NODE_RANK}" >&3
echo "[launcher-bootstrap] topology environment:"
for variable_name in \
  WORLD_SIZE RANK MASTER_ADDR MASTER_PORT SCO_POD_NAME HOSTNAME CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES \
  OPENPI_GPUS_PER_NODE; do
  if [[ -n "${!variable_name:-}" ]]; then
    printf '  %s=%s\n' "${variable_name}" "${!variable_name}"
  fi
done

usage() {
  cat <<'EOF'
Usage (Kubernetes task command):
  # Normal training (default):
  OPENPI_PROFILE=0 OPENPI_RUN_ID=<run_id> bash scripts/train_distributed.sh

  # Select a training config explicitly (hyphen and underscore spellings are accepted):
  OPENPI_RUN_ID=<run_id> bash scripts/train_distributed.sh --config-name=pi05_abc130k

  # Short 16-GPU profiling run:
  OPENPI_PROFILE=1 OPENPI_RUN_ID=<run_id> OPENPI_TRAIN_STEPS=120 bash scripts/train_distributed.sh

  # Profiling with per-rank PyTorch traces (higher overhead and disk usage):
  OPENPI_PROFILE=1 OPENPI_TORCH_TRACE=1 OPENPI_RUN_ID=<run_id> bash scripts/train_distributed.sh

Multi-node Kubernetes topology variables:
  WORLD_SIZE   Total number of nodes before torchrun starts.
  MASTER_ADDR  Address of node rank 0.
  MASTER_PORT  Rendezvous port.

For a single-node job these variables are optional and default to one node at
127.0.0.1:29500. OPENPI_NUM_NODES can explicitly set the node count when
WORLD_SIZE is unavailable.

Node rank resolution order:
  The master/worker role in SCO_POD_NAME. HOSTNAME is used only when
  SCO_POD_NAME is unavailable.

The number of GPUs per node is detected with torch.cuda.device_count(). Set
OPENPI_GPUS_PER_NODE only when an explicit override is required.

The script accepts no positional topology arguments. Configure node count and
GPU count in the Kubernetes task specification. The same command is executed
on every node with node-specific WORLD_SIZE and RANK values.

Optional training variables:
  OPENPI_CONFIG_NAME, OPENPI_RUN_ID, OPENPI_LOCAL_BATCH_SIZE, OPENPI_TRAIN_STEPS, OPENPI_NUM_WORKERS,
  OPENPI_WORKER_BATCH_SIZE, OPENPI_PREFETCH_FACTOR, OPENPI_DATALOADER_IN_ORDER,
  OPENPI_PIN_MEMORY, OPENPI_DATALOADER_PREWARM,
  OPENPI_LOCAL_VIDEO_PATH_DECODE,
  OPENPI_SAVE_INTERVAL, OPENPI_PROFILE, OPENPI_TORCH_TRACE, OPENPI_RESUME,
  OPENPI_OVERWRITE, OPENPI_WANDB, OPENPI_RUN_ROOT, OPENPI_WEIGHT_PATH,
  OPENPI_LOG_ROOT, OPENPI_LOG_ALL_RANKS, OPENPI_LOG_RANK_PHASES,
  OPENPI_GPUS_PER_NODE, OPENPI_NUM_NODES, OPENPI_LAUNCH_DRY_RUN.

Set OPENPI_LAUNCH_DRY_RUN=1 to print the resolved torchrun command without
starting rendezvous or training.

OPENPI_PROFILE accepts 0/1, false/true, no/yes, or off/on. It defaults to 0.
Profiling defaults to 120 steps; normal training defaults to 10000 steps.
OPENPI_TORCH_TRACE is only valid when profiling is enabled.

Checkpoints and profiling outputs share this timestamped run directory:
  <OPENPI_RUN_ROOT>/<OPENPI_CONFIG_NAME>/<OPENPI_RUN_ID>_<UTC start time>/
EOF
}

CLI_CONFIG_NAME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --config-name|--config_name)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "$1 requires a non-empty value." >&2
        exit 2
      fi
      CLI_CONFIG_NAME="$2"
      shift 2
      ;;
    --config-name=*|--config_name=*)
      CLI_CONFIG_NAME="${1#*=}"
      if [[ -z "${CLI_CONFIG_NAME}" ]]; then
        echo "${1%%=*} requires a non-empty value." >&2
        exit 2
      fi
      shift
      ;;
    *)
      echo "Unsupported launcher argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

: "${WORLD_SIZE:?Kubernetes did not inject node count in WORLD_SIZE}"
: "${MASTER_ADDR:?Kubernetes did not inject MASTER_ADDR}"
: "${MASTER_PORT:?Kubernetes did not inject MASTER_PORT}"

NUM_NODES="${WORLD_SIZE}"
NODE_RANK="${BOOTSTRAP_NODE_RANK}"
GPUS_PER_NODE="${OPENPI_GPUS_PER_NODE:-8}"
LOCAL_BATCH_SIZE="${OPENPI_LOCAL_BATCH_SIZE:-64}"
export OPENPI_CONFIG_NAME="${CLI_CONFIG_NAME:-${OPENPI_CONFIG_NAME:-pi05_abc130k}}"

cd "${OPENPI_WORKDIR}"
if [[ -n "${OPENPI_PYTHON:-}" ]]; then
  PYTHON_BIN="${OPENPI_PYTHON}"
elif [[ -x "${OPENPI_WORKDIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${OPENPI_WORKDIR}/.venv/bin/python"
else
  PYTHON_BIN="python"
fi

for value_name in NUM_NODES GPUS_PER_NODE; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${value_name} must be a positive integer, got: ${value}" >&2
    exit 2
  fi
done
if [[ ! "${NODE_RANK}" =~ ^[0-9]+$ ]]; then
  echo "NODE_RANK must be a non-negative integer, got: ${NODE_RANK}" >&2
  exit 2
fi
if (( NODE_RANK >= NUM_NODES )); then
  echo "NODE_RANK (${NODE_RANK}) must be smaller than NUM_NODES (${NUM_NODES})" >&2
  exit 2
fi
if [[ ! "${MASTER_PORT}" =~ ^[0-9]+$ ]] || (( MASTER_PORT < 1 || MASTER_PORT > 65535 )); then
  echo "MASTER_PORT must be an integer in 1..65535, got: ${MASTER_PORT}" >&2
  exit 2
fi

GLOBAL_WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
GLOBAL_BATCH_SIZE=$((GLOBAL_WORLD_SIZE * LOCAL_BATCH_SIZE))

export OPENPI_BATCH_SIZE="${GLOBAL_BATCH_SIZE}"
export OPENPI_NUM_WORKERS="${OPENPI_NUM_WORKERS:-16}"
export OPENPI_WORKER_BATCH_SIZE="${OPENPI_WORKER_BATCH_SIZE:-16}"
export OPENPI_PREFETCH_FACTOR="${OPENPI_PREFETCH_FACTOR:-2}"
export OPENPI_OVERWRITE="${OPENPI_OVERWRITE:-0}"
export OPENPI_DISTRIBUTED=1
export OPENPI_LOG_ALL_RANKS="${OPENPI_LOG_ALL_RANKS:-0}"
export OPENPI_LOG_RANK_PHASES="${OPENPI_LOG_RANK_PHASES:-1}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-INFO}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,ENV,NET}"
export OPENPI_WANDB=1 
export WANDB_API_KEY="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"
export WANDB_MODE=online 

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
normalize_bool_env OPENPI_DATALOADER_IN_ORDER 0
normalize_bool_env OPENPI_PIN_MEMORY 0
normalize_bool_env OPENPI_DATALOADER_PREWARM 1
normalize_bool_env OPENPI_LOCAL_VIDEO_PATH_DECODE 1

if [[ "${OPENPI_PROFILE}" == "1" ]]; then
  TRAIN_MODE="profiling"
  export OPENPI_TRAIN_STEPS="${OPENPI_TRAIN_STEPS:-1000}"
else
  TRAIN_MODE="training"
  export OPENPI_TRAIN_STEPS="${OPENPI_TRAIN_STEPS:-10000}"
  if [[ "${OPENPI_TORCH_TRACE}" == "1" ]]; then
    echo "OPENPI_TORCH_TRACE requires OPENPI_PROFILE=1." >&2
    exit 2
  fi
fi

if [[ ! "${OPENPI_TRAIN_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "OPENPI_TRAIN_STEPS must be a positive integer, got: ${OPENPI_TRAIN_STEPS}" >&2
  exit 2
fi

if [[ ! "${OPENPI_NUM_WORKERS}" =~ ^[0-9]+$ ]]; then
  echo "OPENPI_NUM_WORKERS must be a non-negative integer, got: ${OPENPI_NUM_WORKERS}" >&2
  exit 2
fi
if [[ ! "${OPENPI_WORKER_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "OPENPI_WORKER_BATCH_SIZE must be a positive integer, got: ${OPENPI_WORKER_BATCH_SIZE}" >&2
  exit 2
fi
if [[ ! "${OPENPI_PREFETCH_FACTOR}" =~ ^[1-9][0-9]*$ ]]; then
  echo "OPENPI_PREFETCH_FACTOR must be a positive integer, got: ${OPENPI_PREFETCH_FACTOR}" >&2
  exit 2
fi
if (( LOCAL_BATCH_SIZE % OPENPI_WORKER_BATCH_SIZE != 0 )); then
  echo "OPENPI_LOCAL_BATCH_SIZE (${LOCAL_BATCH_SIZE}) must be divisible by OPENPI_WORKER_BATCH_SIZE (${OPENPI_WORKER_BATCH_SIZE})." >&2
  exit 2
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((GPUS_PER_NODE - 1))")"
  export CUDA_VISIBLE_DEVICES
else
  IFS=',' read -r -a VISIBLE_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
  if (( ${#VISIBLE_DEVICES[@]} < GPUS_PER_NODE )); then
    echo "CUDA_VISIBLE_DEVICES exposes ${#VISIBLE_DEVICES[@]} device(s), but ${GPUS_PER_NODE} GPU process(es) were requested." >&2
    exit 2
  fi
fi

TORCHRUN_LOG_DIR="${LOG_ROOT}/${OPENPI_RUN_ID}/node_${NODE_RANK}"
mkdir -p "${TORCHRUN_LOG_DIR}"

LAUNCH_COMMAND=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nnodes="${NUM_NODES}"
  --nproc-per-node="${GPUS_PER_NODE}"
  --node-rank="${NODE_RANK}"
  --master-addr="${MASTER_ADDR}"
  --master-port="${MASTER_PORT}"
  --log-dir="${TORCHRUN_LOG_DIR}"
  --tee=3
  scripts/train_distributed.py
)

echo "[launcher] timestamp=$(date --iso-8601=seconds) host=$(hostname) pid=$$ user=$(id -u):$(id -g)"
echo "[launcher] workdir=${OPENPI_WORKDIR} python=${PYTHON_BIN}"
echo "[topology] nnodes=${NUM_NODES} gpus_per_node=${GPUS_PER_NODE} node_rank=${NODE_RANK}"
echo "[topology] master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[topology] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[train] global_world_size=${GLOBAL_WORLD_SIZE} global_batch_size=${GLOBAL_BATCH_SIZE} local_batch_size=${LOCAL_BATCH_SIZE}"
echo "[train] config_name=${OPENPI_CONFIG_NAME} run_id=${OPENPI_RUN_ID} workers_per_rank=${OPENPI_NUM_WORKERS} checkpoint_root=${OPENPI_RUN_ROOT}"
echo "[train] run_dir_template=${OPENPI_RUN_ROOT}/${OPENPI_CONFIG_NAME}/${OPENPI_RUN_ID}_<UTC start time>"
echo "[train] mode=${TRAIN_MODE} steps=${OPENPI_TRAIN_STEPS} profiling=${OPENPI_PROFILE} torch_trace=${OPENPI_TORCH_TRACE}"
echo "[logging] torchrun_log_dir=${TORCHRUN_LOG_DIR} tee=stdout+stderr all_ranks"
echo "[logging] TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG} NCCL_DEBUG=${NCCL_DEBUG} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[launcher] visible GPUs on node ${NODE_RANK}:"
  nvidia-smi -L || echo "[launcher][warning] nvidia-smi failed with exit code $?" >&2
else
  echo "[launcher][warning] nvidia-smi is not available" >&2
fi

echo "[launcher] Python/PyTorch preflight:"
"${PYTHON_BIN}" -c 'import os, platform, sys, torch; print("python={} torch={} host={} cuda_available={} cuda_count={} CUDA_VISIBLE_DEVICES={}".format(sys.version.split()[0], torch.__version__, platform.node(), torch.cuda.is_available(), torch.cuda.device_count(), os.environ.get("CUDA_VISIBLE_DEVICES")), flush=True)'

if [[ "${OPENPI_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
  printf '[launcher] command:'
  printf ' %q' "${LAUNCH_COMMAND[@]}"
  printf '\n[launcher] dry run complete; torchrun was not started\n'
  exit 0
fi

unset WORLD_SIZE RANK LOCAL_RANK GROUP_RANK ROLE_RANK
exec "${LAUNCH_COMMAND[@]}"
