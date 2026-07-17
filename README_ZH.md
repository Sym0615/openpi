# openpi + petreloss.conf training

openpi环境配置 checkpoints 准备 参考 飞书文档 

单卡训练sbatch脚本  根据自己路径修改配置 本地需准备sdk petreloss.conf

#!/bin/bash
#SBATCH --job-name=pi05-interna1-pt-smoke
#SBATCH --partition=wam_data
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --quotatype=auto
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/petrelfs/xueyufei/project/code/openpi-workspace/logs/%x-%j.out
#SBATCH --error=/mnt/petrelfs/xueyufei/project/code/openpi-workspace/logs/%x-%j.err

set -euo pipefail
unset LC_ALL

OPENPI_WORKDIR=/mnt/petrelfs/xueyufei/project/code/openpi-workspace
OPENPI_REPO="$OPENPI_WORKDIR/openpi"
OPENPI_ENV="$OPENPI_REPO/.venv-ubuntu22"
SIF="$OPENPI_WORKDIR/ubuntu22-cuda124.sif"
RUN_ROOT="$OPENPI_WORKDIR/runs"
BASE_MODEL_DIR="$OPENPI_WORKDIR/checkpoints/pi05_base_pytorch_bf16"
OPENPI_APT_LIB="$OPENPI_WORKDIR/cache/apt-git/extract/usr/lib/x86_64-linux-gnu"
TOKENIZER="$OPENPI_WORKDIR/cache/openpi/big_vision/paligemma_tokenizer.model"
NORM_STATS="$OPENPI_REPO/assets/pi05_interna1/intern_a1/norm_stats.json"
EXP_NAME="interna1_pytorch_smoke_${SLURM_JOB_ID}"

test -x "$OPENPI_ENV/bin/python"
test -x "$OPENPI_ENV/bin/torchrun"
test -f "$SIF"
test -f "$BASE_MODEL_DIR/model.safetensors"
test -f "$TOKENIZER"
test -f "$NORM_STATS"
test -f "$HOME/petreloss.conf"
test -f "$HOME/.netrc"

mkdir -p \
    "$OPENPI_WORKDIR/logs" \
    "$RUN_ROOT/checkpoints_pytorch" \
    "$RUN_ROOT/wandb" \
    "$OPENPI_WORKDIR/cache/openpi" \
    "$OPENPI_WORKDIR/cache/huggingface" \
    "$OPENPI_WORKDIR/cache/huggingface/lerobot"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

apptainer exec --nv --bind /mnt:/mnt \
    --env OPENPI_DATA_HOME="$OPENPI_WORKDIR/cache/openpi" \
    --env HF_HOME="$OPENPI_WORKDIR/cache/huggingface" \
    --env HF_LEROBOT_HOME="$OPENPI_WORKDIR/cache/huggingface/lerobot" \
    --env WANDB_MODE=online \
    --env WANDB_DIR="$RUN_ROOT/wandb" \
    --env WANDB_PROJECT=openpi \
    --env WANDB_NAME="$EXP_NAME" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    --env NCCL_DEBUG=WARN \
    --env JAX_PLATFORMS=cpu \
    --env OPENPI_PETREL_BACKEND=petrel \
    --env OPENPI_APT_LIB="$OPENPI_APT_LIB" \
    --env PYTHONUNBUFFERED=1 \
    "$SIF" bash -lc "
        set -euo pipefail
        export LD_LIBRARY_PATH='$OPENPI_APT_LIB':\${LD_LIBRARY_PATH:-}
        cd '$OPENPI_REPO'
        exec '$OPENPI_ENV/bin/torchrun' \
            --standalone \
            --nnodes=1 \
            --nproc-per-node=8 \
            scripts/train_pytorch.py \
            pi05_interna1 \
            --exp_name '$EXP_NAME' \
            --pytorch_weight_path '$BASE_MODEL_DIR' \
            --checkpoint_base_dir '$RUN_ROOT/checkpoints_pytorch' \
            --batch_size 8 \
            --num_train_steps 20000 \
            --log_interval 1 \
            --save_interval 1000 \
            --overwrite
    "

CHECKPOINT_DIR="$RUN_ROOT/checkpoints_pytorch/pi05_interna1/$EXP_NAME"
test -d "$CHECKPOINT_DIR"
find "$CHECKPOINT_DIR" -maxdepth 4 -type f | sort | head -100
du -sh "$CHECKPOINT_DIR"


# TODO
1. DATALOADER CACHE
2. 分批数据归一化
