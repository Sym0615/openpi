#!/usr/bin/env bash
set -Eeuo pipefail

# Clusterx starts the container as root. Keep root only long enough to restore
# the normal runtime identity, while preserving the injected DDP topology env.
if [[ "$(id -u)" -eq 0 ]]; then
    if ! id zbzhu >/dev/null 2>&1; then
        echo "cluster runtime user is missing" >&2
        exit 70
    fi
    exec /usr/sbin/runuser -u zbzhu -- \
        env HOME=/home/zbzhu USER=zbzhu LOGNAME=zbzhu \
        /usr/bin/zsh -c 'source "$HOME/.zshrc"; exec bash "$@"' \
        openpi-clusterx "$0" "$@"
fi

umask 002

export HOME=/home/zbzhu
export OPENPI_WORKDIR=/home/zbzhu/Code/openpi
export OPENPI_RUN_ROOT="${OPENPI_RUN_ROOT:-/data/zbzhu/openpi/pi05_abc130k}"
export OPENPI_LOG_ROOT="${OPENPI_RUN_ROOT}/distributed_logs"
export OPENPI_DATA_HOME=/home/zbzhu/.cache/openpi
export OPENPI_SHARED_CACHE=/data/zbzhu/openpi/cache
export HF_HOME="${OPENPI_SHARED_CACHE}/huggingface"
export HF_LEROBOT_HOME="${OPENPI_SHARED_CACHE}/lerobot"
export TMPDIR=/data/zbzhu/openpi/tmp
export OPENPI_WEIGHT_PATH=/oss/datahub/public/model/pi05_base_pytorch_bf16

: "${OPENPI_RUN_ID:?OPENPI_RUN_ID must be set}"

# Stable 64-GPU profile (2026-08-05): DDP + ZeRO-1, B13 x 4 gradient
# accumulation. This yields an effective global batch of 13 * 4 * 64 = 3328.
export OPENPI_LOCAL_BATCH_SIZE="${OPENPI_LOCAL_BATCH_SIZE:-13}"
export OPENPI_NUM_WORKERS="${OPENPI_NUM_WORKERS:-16}"
export OPENPI_WORKER_BATCH_SIZE="${OPENPI_WORKER_BATCH_SIZE:-13}"
export OPENPI_PREFETCH_FACTOR="${OPENPI_PREFETCH_FACTOR:-2}"
export OPENPI_TRAIN_STEPS="${OPENPI_TRAIN_STEPS:-50000}"
export OPENPI_GRADIENT_ACCUMULATION_STEPS="${OPENPI_GRADIENT_ACCUMULATION_STEPS:-4}"
export OPENPI_STREAMING_ACCUMULATION="${OPENPI_STREAMING_ACCUMULATION:-1}"
export OPENPI_ASYNC_QUEUE_SIZE="${OPENPI_ASYNC_QUEUE_SIZE:-16}"
# The ABC-130K parquet working set is about 202 GiB. ClusterX submissions using
# this optimized default must request at least 256 GiB of /dev/shm per node.
export OPENPI_STAGE_PARQUETS="${OPENPI_STAGE_PARQUETS:-1}"
export OPENPI_PARQUET_STAGE_WORKERS="${OPENPI_PARQUET_STAGE_WORKERS:-32}"
export OPENPI_SAVE_INTERVAL="${OPENPI_SAVE_INTERVAL:-1000}"
export OPENPI_LOG_INTERVAL="${OPENPI_LOG_INTERVAL:-1}"
export OPENPI_PROFILE="${OPENPI_PROFILE:-0}"
export OPENPI_TORCH_TRACE="${OPENPI_TORCH_TRACE:-0}"
export OPENPI_RESUME="${OPENPI_RESUME:-0}"
export OPENPI_OVERWRITE="${OPENPI_OVERWRITE:-0}"
export OPENPI_WANDB="${OPENPI_WANDB:-1}"
export OPENPI_ZERO_OPTIMIZER="${OPENPI_ZERO_OPTIMIZER:-1}"
export OPENPI_BF16_AUTOCAST="${OPENPI_BF16_AUTOCAST:-1}"
export OPENPI_COMPILE_TRANSFORMER_LAYERS="${OPENPI_COMPILE_TRANSFORMER_LAYERS:-1}"
export OPENPI_DATALOADER_IN_ORDER=0
export OPENPI_PIN_MEMORY=0
export OPENPI_DATALOADER_PREWARM=1
export OPENPI_LOCAL_VIDEO_PATH_DECODE=1
export OPENPI_GPUS_PER_NODE=8

export WANDB_MODE=online
export WANDB_DIR=/data/zbzhu/openpi/wandb
export JAX_PLATFORMS=cpu
export OPENPI_PETREL_BACKEND=petrel
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

test -x "${OPENPI_WORKDIR}/.venv/bin/python"
test -s "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"
test -s "${OPENPI_WEIGHT_PATH}/model.safetensors"
test -s "${OPENPI_WORKDIR}/assets/pi05_abc130k_debug/abc130k/norm_stats.json"
if [[ "${OPENPI_WANDB}" == "1" ]]; then
    "${OPENPI_WORKDIR}/.venv/bin/python" - <<'PY'
import netrc

credentials = netrc.netrc().authenticators("api.wandb.ai")
if not credentials or not credentials[2]:
    raise RuntimeError("W&B credentials are missing from ~/.netrc")
PY
fi
mkdir -p "${OPENPI_RUN_ROOT}" "${OPENPI_LOG_ROOT}" "${OPENPI_DATA_HOME}" "${TMPDIR}" "${WANDB_DIR}"

cd "${OPENPI_WORKDIR}"
if [[ "${OPENPI_STAGE_PARQUETS:-0}" == "1" ]]; then
    : "${OPENPI_PARQUET_CACHE_DIR:=/dev/shm/openpi-abc130k-parquets}"
    export OPENPI_PARQUET_CACHE_DIR
    stage_workers="${OPENPI_PARQUET_STAGE_WORKERS}"
    echo "[train] staging parquet cache=${OPENPI_PARQUET_CACHE_DIR} workers=${stage_workers}"
    .venv/bin/python scripts/stage_petrel_parquets.py \
        --path-index-root data/petrel_cache/path_index/abc130k \
        --cache-dir "${OPENPI_PARQUET_CACHE_DIR}" \
        --workers "${stage_workers}"
fi
exec bash scripts/train_distributed.sh --config-name=pi05_abc130k_debug
