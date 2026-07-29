#!/usr/bin/env bash
# Launch Cosmos3-Edge LIBERO-all post-training on SSP PyTorch DDP (2 nodes x 8 GPUs).

set -euo pipefail

REPO_ROOT="/liuyu2/project/cosmos-framework"
CONDA_ROOT="/root/miniconda3"
SHARED_CONDA_ENV="/liuyu2/project/envs/cosmos"
ROOT_CONDA_ENV="$CONDA_ROOT/envs/cosmos"

# if [[ -x "$SHARED_CONDA_ENV/bin/python" ]]; then
#     CONDA_ENV_PREFIX="$SHARED_CONDA_ENV"
# else
#     CONDA_ENV_PREFIX="$ROOT_CONDA_ENV"
# fi

CONDA_ENV_PREFIX="$ROOT_CONDA_ENV"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PREFIX"

export HF_HOME="/data/wammodel/share/huggingface"
export LD_LIBRARY_PATH=""
export LIBERO_ROOT="/data/wammodel/share/huggingface/hub/datasets--nvidia--LIBERO_LeRobot_v3/snapshots/e5907374380b8f96511957e6ba5582be52a1e179"
export WAN_VAE_PATH="/data/wammodel/share/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/Wan2.2_VAE.pth"
export BASE_CHECKPOINT_PATH="$REPO_ROOT/examples/checkpoints/Cosmos3-Edge"
export OUTPUT_ROOT="/data/wamdata/liuyu/cosmos3-edge/outputs/libero_edge_16k_32worker_8pre"
export IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT"

# SSP variable names differ between platform versions. Prefer the documented
# PyTorch variables, then TorchElastic/PET variables, and finally derive the
# fixed two-node rank from the master-0 / worker-0 Pod role.
export NNODES="${SENSECORE_PYTORCH_NNODES:-${PET_NNODES:-2}}"
POD_NAME="${SCO_POD_NAME:-${HOSTNAME:-}}"
if [[ -n "${SENSECORE_PYTORCH_NODE_RANK:-}" ]]; then
    NODE_RANK="$SENSECORE_PYTORCH_NODE_RANK"
elif [[ -n "${PET_NODE_RANK:-}" ]]; then
    NODE_RANK="$PET_NODE_RANK"
elif [[ "$POD_NAME" == *master-0* ]]; then
    NODE_RANK=0
elif [[ "$POD_NAME" =~ worker-([0-9]+)$ ]]; then
    NODE_RANK=$((BASH_REMATCH[1] + 1))
else
    echo "ERROR: cannot derive node rank from SSP Pod name: $POD_NAME" >&2
    exit 1
fi
export NODE_RANK

export MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-${LWS_LEADER_ADDRESS:-}}}"
if [[ -z "$MASTER_ADDR" ]]; then
    echo "ERROR: SSP did not inject MASTER_ADDR, PET_MASTER_ADDR, or LWS_LEADER_ADDRESS" >&2
    exit 1
fi
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-50012}"
export NPROC_PER_NODE="${SENSECORE_ACCELERATE_DEVICE_COUNT:-${SCO_NUM_ACCELERATORS_PER_POD:-8}}"
export LOG_FILENAME="action_policy_libero_all_edge_rank${NODE_RANK}.log"

if [[ "$NNODES" != "2" ]]; then
    echo "ERROR: this recipe requires 2 nodes, but SSP supplied NNODES=$NNODES" >&2
    exit 1
fi
if [[ "$NPROC_PER_NODE" != "8" ]]; then
    echo "ERROR: this recipe requires 8 GPUs per node, but SSP supplied NPROC_PER_NODE=$NPROC_PER_NODE" >&2
    exit 1
fi
if [[ ! -d "$BASE_CHECKPOINT_PATH/model" ]]; then
    echo "ERROR: converted Edge DCP is missing: $BASE_CHECKPOINT_PATH/model" >&2
    exit 1
fi
for suite in libero_spatial libero_object libero_goal libero_10; do
    if [[ ! -f "$LIBERO_ROOT/$suite/meta/info.json" ]]; then
        echo "ERROR: missing LIBERO suite metadata: $LIBERO_ROOT/$suite/meta/info.json" >&2
        exit 1
    fi
done
if [[ ! -f "$WAN_VAE_PATH" ]]; then
    echo "ERROR: Wan2.2 VAE is missing: $WAN_VAE_PATH" >&2
    exit 1
fi

echo "SSP launch: nnodes=$NNODES node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT gpus/node=$NPROC_PER_NODE"
echo "Conda environment: $CONDA_ENV_PREFIX"
echo "Output root: $OUTPUT_ROOT"

cd "$REPO_ROOT"
exec bash examples/launch_sft_action_policy_libero_all_edge.sh