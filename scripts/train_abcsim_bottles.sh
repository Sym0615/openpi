#!/usr/bin/env bash
set -Eeuo pipefail

# Follow the ClusterX wrapper layout; this development container runs as root.
# Keep the current runtime identity and the injected DDP topology environment.
umask 002

export OPENPI_WORKDIR="${OPENPI_WORKDIR:-/sunyiming/openpi}"
# Use the existing /data mount for full checkpoints; keep launcher logs in the workspace.
export OPENPI_RUN_ROOT="${OPENPI_RUN_ROOT:-/data/sunyiming/openpi/abc_sim_bottles}"
export OPENPI_LOG_ROOT="${OPENPI_LOG_ROOT:-${OPENPI_WORKDIR}/runs/distributed_logs}"
export OPENPI_RUN_ID="${OPENPI_RUN_ID:-abc_sim_bottles_finetune}"

# Keep startup diagnostics on shared storage even if the platform deletes the pod.
preflight_run_id="$(printf '%s' "${OPENPI_RUN_ID}" | tr -cs '[:alnum:]_.-' '_')"
preflight_log_dir="${OPENPI_LOG_ROOT}/${preflight_run_id}"
mkdir -p "${preflight_log_dir}"
exec > >(tee -a "${preflight_log_dir}/preflight.log") 2>&1
trap 'rc=$?; echo "[preflight][error] line=${LINENO} exit=${rc}" >&2' ERR

# Use the same proxy_on settings as the development shell, from shared storage.
PROXY_ENV_FILE="${OPENPI_PROXY_ENV_FILE:-${OPENPI_WORKDIR}/data/network/proxy.env}"
if [[ ! -r "${PROXY_ENV_FILE}" ]]; then
    echo "Proxy configuration is not readable: ${PROXY_ENV_FILE}" >&2
    exit 1
fi
source "${PROXY_ENV_FILE}"
proxy_on

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${OPENPI_WORKDIR}/data/openpi_assets}"
export OPENPI_SHARED_CACHE="${OPENPI_SHARED_CACHE:-${OPENPI_WORKDIR}/data/cache}"
export HF_HOME="${OPENPI_SHARED_CACHE}/huggingface"
export HF_LEROBOT_HOME="${OPENPI_SHARED_CACHE}/lerobot"
export TMPDIR="${TMPDIR:-${OPENPI_SHARED_CACHE}/tmp}"
export OPENPI_WEIGHT_PATH="${OPENPI_WEIGHT_PATH:-${OPENPI_WORKDIR}/data/models/pi05_base_hf}"
export OPENPI_CHECKPOINT_STORAGE="${OPENPI_CHECKPOINT_STORAGE:-local}"

export OPENPI_CONFIG_NAME=pi05_abc_sim_bottles

# ABC-Sim bottles fine-tuning: 8 x A100 80GB, DDP + ZeRO-1.
# Per-GPU batch 8 x 2 accumulation x 8 GPUs gives effective global batch 128.
# For a smoke run, override OPENPI_TRAIN_STEPS=100 and OPENPI_SAVE_INTERVAL=100.
export OPENPI_LOCAL_BATCH_SIZE="${OPENPI_LOCAL_BATCH_SIZE:-8}"
export OPENPI_NUM_WORKERS="${OPENPI_NUM_WORKERS:-4}"
export OPENPI_WORKER_BATCH_SIZE="${OPENPI_WORKER_BATCH_SIZE:-8}"
export OPENPI_PREFETCH_FACTOR="${OPENPI_PREFETCH_FACTOR:-2}"
export OPENPI_TRAIN_STEPS="${OPENPI_TRAIN_STEPS:-30000}"
export OPENPI_GRADIENT_ACCUMULATION_STEPS="${OPENPI_GRADIENT_ACCUMULATION_STEPS:-2}"
export OPENPI_STREAMING_ACCUMULATION="${OPENPI_STREAMING_ACCUMULATION:-0}"
export OPENPI_ASYNC_QUEUE_SIZE="${OPENPI_ASYNC_QUEUE_SIZE:-0}"
# Keep the validated data-loading settings; the optional staging block uses
# the ABC-Sim train path index and the node-local cache.
export OPENPI_STAGE_PARQUETS="${OPENPI_STAGE_PARQUETS:-0}"
export OPENPI_PARQUET_STAGE_WORKERS="${OPENPI_PARQUET_STAGE_WORKERS:-32}"
export OPENPI_SAVE_INTERVAL="${OPENPI_SAVE_INTERVAL:-5000}"
export OPENPI_LOG_INTERVAL="${OPENPI_LOG_INTERVAL:-10}"
export OPENPI_PROFILE="${OPENPI_PROFILE:-0}"
export OPENPI_TORCH_TRACE="${OPENPI_TORCH_TRACE:-0}"
export OPENPI_RESUME="${OPENPI_RESUME:-0}"
export OPENPI_OVERWRITE="${OPENPI_OVERWRITE:-0}"
export OPENPI_WANDB="${OPENPI_WANDB:-1}"
export OPENPI_ZERO_OPTIMIZER="${OPENPI_ZERO_OPTIMIZER:-1}"
export OPENPI_BF16_AUTOCAST="${OPENPI_BF16_AUTOCAST:-1}"
export OPENPI_GRADIENT_CHECKPOINTING="${OPENPI_GRADIENT_CHECKPOINTING:-1}"
export OPENPI_COMPILE_TRANSFORMER_LAYERS="${OPENPI_COMPILE_TRANSFORMER_LAYERS:-0}"
export OPENPI_DATALOADER_IN_ORDER="${OPENPI_DATALOADER_IN_ORDER:-1}"
export OPENPI_PIN_MEMORY=0
export OPENPI_DATALOADER_PREWARM="${OPENPI_DATALOADER_PREWARM:-0}"
export OPENPI_LOCAL_VIDEO_PATH_DECODE=1
export OPENPI_GPUS_PER_NODE=8

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-sym0615-tongji-university}"
# Resolve credentials from shared storage instead of the container-local ~/.netrc.
export NETRC="${NETRC:-${OPENPI_WORKDIR}/data/network/wandb.netrc}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
# Disable Sentry error reporting; W&B training metrics remain enabled.
export WANDB_ERROR_REPORTING=false
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-60}"
export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-20}"
export WANDB_DIR="${WANDB_DIR:-${OPENPI_RUN_ROOT}/wandb}"
export JAX_PLATFORMS=cpu
export OPENPI_PETREL_BACKEND=petrel
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${OPENPI_RUN_ROOT}" "${OPENPI_LOG_ROOT}" "${OPENPI_DATA_HOME}" "${TMPDIR}" "${WANDB_DIR}"

test -x "${OPENPI_WORKDIR}/.venv/bin/python"
test -s "${OPENPI_DATA_HOME}/big_vision/paligemma_tokenizer.model"
test -s "${OPENPI_WEIGHT_PATH}/model.safetensors"
test -s "${OPENPI_WORKDIR}/assets/pi05_abc_sim_bottles/abc_sim_bottles/norm_stats.json"
test -s "${OPENPI_WORKDIR}/data/petrel_cache/filter/abc_sim_bottles/train/frame_filter_manifest.json"
if [[ "${OPENPI_WANDB}" == "1" ]]; then
    "${OPENPI_WORKDIR}/.venv/bin/python" - <<'PY'
import netrc
import os
import re
import time
from urllib.parse import unquote, urlparse

import requests

key = os.environ.get("WANDB_API_KEY")
if not key:
    path = os.environ["NETRC"]
    try:
        credentials = netrc.netrc(path).authenticators("api.wandb.ai")
    except (OSError, netrc.NetrcParseError) as exc:
        raise RuntimeError(f"Cannot read W&B login file: {path} ({type(exc).__name__})") from None
    if not credentials or not credentials[2]:
        raise RuntimeError(f"W&B credentials are missing from {path}")
    key = credentials[2]

# Keep exception details useful without exposing proxy or W&B credentials.
secrets = {key, os.environ.get("PROXY_PASSWORD", ""), os.environ.get("PROXY_USER", "")}
for variable in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    proxy = urlparse(os.environ.get(variable, ""))
    for value in (proxy.username, proxy.password):
        if value:
            secrets.update((value, unquote(value)))


def safe_error(exc):
    detail = str(exc)
    for secret in sorted(filter(None, secrets), key=len, reverse=True):
        detail = detail.replace(secret, "<REDACTED>")
    return re.sub(r"(https?://)[^/\s@]+@", r"\1<REDACTED>@", detail)


# Check the actual training container's proxy and login before starting 8 ranks.
# Retry brief network failures while preserving online W&B and failing on bad login.
query = "query { viewer { username entity teams { edges { node { name } } } } }"
for attempt in range(1, 4):
    try:
        response = requests.post(
            "https://api.wandb.ai/graphql",
            json={"query": query},
            auth=("api", key),
            timeout=(10, 20),
        )
        break
    except requests.RequestException as exc:
        detail = f"{type(exc).__name__}: {safe_error(exc)}"
        print(f"[wandb-preflight] attempt={attempt}/3 network error: {detail}", flush=True)
        if attempt == 3 or "407" in str(exc):
            raise RuntimeError(f"W&B network check failed: {detail}; check the proxy configuration") from None
        time.sleep(5)
if response.status_code != 200:
    raise RuntimeError(f"W&B login check failed: HTTP {response.status_code}")
try:
    viewer = (response.json().get("data") or {}).get("viewer")
except ValueError:
    viewer = None
if not viewer:
    raise RuntimeError("W&B login check returned no account")
entities = {viewer["username"], viewer.get("entity")}
entities.update(edge["node"]["name"] for edge in viewer.get("teams", {}).get("edges", []))
entity = os.environ["WANDB_ENTITY"]
if entity not in entities:
    raise RuntimeError(f"W&B account {viewer['username']} does not belong to entity {entity}")
print(f"[wandb-preflight] account={viewer['username']} entity={entity} network=OK", flush=True)
PY
fi

cd "${OPENPI_WORKDIR}"
if [[ "${OPENPI_STAGE_PARQUETS:-0}" == "1" ]]; then
    : "${OPENPI_PARQUET_CACHE_DIR:=/dev/shm/openpi-abc-sim-bottles-parquets}"
    export OPENPI_PARQUET_CACHE_DIR
    stage_workers="${OPENPI_PARQUET_STAGE_WORKERS}"
    echo "[train] staging parquet cache=${OPENPI_PARQUET_CACHE_DIR} workers=${stage_workers}"
    .venv/bin/python scripts/stage_petrel_parquets.py \
        --path-index-root data/petrel_cache/path_index/abc_sim_bottles/train \
        --cache-dir "${OPENPI_PARQUET_CACHE_DIR}" \
        --workers "${stage_workers}"
    # Staging was handled for ABC-Sim; skip the generic launcher's ABC-130K staging.
    export OPENPI_STAGE_PARQUETS=0
fi
exec bash scripts/train_distributed.sh --config-name=pi05_abc_sim_bottles "$@"
