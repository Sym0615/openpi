"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import datetime
import gc
import itertools
import json
import logging
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
from openpi.training import profiling as _profiling
import openpi.training.config as _config
import openpi.training.data_loader as _data


def log_rank_phase(message: str) -> None:
    """Emit unbuffered phase markers from every rank for distributed failure diagnosis."""
    if os.environ.get("OPENPI_LOG_RANK_PHASES", "1") == "0":
        return
    print(
        f"[rank-phase] host={platform.node()} pid={os.getpid()} rank={os.environ.get('RANK', '?')} "
        f"local_rank={os.environ.get('LOCAL_RANK', '?')} {message}",
        file=sys.stderr,
        flush=True,
    )


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            record.rank = os.environ.get("RANK", "0")
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s][rank=%(rank)s] "
        "%(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    rank = int(os.environ.get("RANK", "0"))
    log_all_ranks = os.environ.get("OPENPI_LOG_ALL_RANKS", "0") == "1"
    logger.setLevel(logging.INFO if rank == 0 or log_all_ranks else logging.WARNING)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "INFO")
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        timeout_s = int(os.environ.get("OPENPI_DDP_TIMEOUT_SECONDS", "1800"))
        log_rank_phase(
            f"init_process_group begin backend={backend} world_size={world_size} "
            f"master={os.environ.get('MASTER_ADDR', '?')}:{os.environ.get('MASTER_PORT', '?')} timeout_s={timeout_s}"
        )
        torch.distributed.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=datetime.timedelta(seconds=timeout_s),
        )
        log_rank_phase("init_process_group complete")

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if os.environ.get("OPENPI_DISTRIBUTED") == "1" and not torch.cuda.is_available():
        raise RuntimeError("Distributed InternA1 training requires CUDA, but torch.cuda.is_available() is false.")
    if torch.cuda.is_available() and local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is not available; this process sees {torch.cuda.device_count()} CUDA device(s)."
        )
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    log_rank_phase(
        f"device setup complete device={device} cuda_available={torch.cuda.is_available()} "
        f"visible_cuda_devices={torch.cuda.device_count()}"
    )
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def distributed_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def distributed_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def node_rank() -> int:
    return int(os.environ.get("GROUP_RANK", os.environ.get("NODE_RANK", "0")))


def visible_gpu_for_local_rank(local_rank: int) -> str | int:
    visible_devices = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
    visible_devices = [value for value in visible_devices if value]
    if local_rank < len(visible_devices):
        return visible_devices[local_rank]
    return local_rank


def synchronize_rank0_error(error: Exception | None, *, phase: str, use_ddp: bool) -> None:
    """Broadcast a rank-0 setup failure so every rank exits instead of hanging at a barrier."""
    if not use_ddp:
        if error is not None:
            raise error
        return
    payload = [None if error is None else f"{type(error).__name__}: {error}"]
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise RuntimeError(f"Rank 0 failed during {phase}: {payload[0]}")


def checkpoint_due(global_step: int, config: _config.TrainConfig) -> bool:
    return (global_step > 0 and global_step % config.save_interval == 0) or global_step == config.num_train_steps


def resolve_timestamped_run_config(
    config: _config.TrainConfig, *, use_ddp: bool, rank: int
) -> _config.TrainConfig:
    """Resolve one shared <run_id>_<UTC time> directory before any output is created."""
    if os.environ.get("OPENPI_TIMESTAMPED_RUN_DIR") != "1":
        return config

    run_id = os.environ.get("OPENPI_RUN_ID", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError(
            "OPENPI_RUN_ID must start with an alphanumeric character and contain only "
            f"letters, numbers, '.', '_' or '-', got {run_id!r}"
        )

    run_base_dir = Path(config.checkpoint_base_dir).resolve() / config.name
    payload: list[str | None] = [None]
    resolution_error: Exception | None = None
    if not use_ddp or rank == 0:
        try:
            if config.resume:
                candidates = sorted(
                    (
                        path
                        for path in run_base_dir.glob(f"{run_id}_*")
                        if path.is_dir() and get_latest_checkpoint_step(path) is not None
                    ),
                    key=lambda path: path.name,
                )
                if not candidates:
                    raise FileNotFoundError(
                        f"No timestamped checkpoint directory matching {run_id}_* under {run_base_dir}"
                    )
                payload[0] = candidates[-1].name
            else:
                started_at = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S_%f")
                payload[0] = f"{run_id}_{started_at}"
        except Exception as exc:
            resolution_error = exc

    synchronize_rank0_error(resolution_error, phase="run directory resolution", use_ddp=use_ddp)
    if use_ddp:
        dist.broadcast_object_list(payload, src=0)
    if payload[0] is None:
        raise RuntimeError("Global rank 0 did not provide a resolved run directory name")

    resolved_config = dataclasses.replace(config, exp_name=payload[0])
    if rank == 0:
        logging.info("Resolved run directory: %s", resolved_config.checkpoint_dir)
    return resolved_config


def aggregate_profile_values(
    local_values: dict[str, float], *, local_batch_size: int, device: torch.device, use_ddp: bool
) -> dict[str, float]:
    """Aggregate per-rank profiling windows into global averages, maxima, and throughput."""
    keys = (
        "data_wait_ms",
        "h2d_ms",
        "forward_ms",
        "backward_ms",
        "optimizer_ms",
        "step_ms",
        "samples_per_second",
    )
    values = torch.tensor([float(local_values[key]) for key in keys], dtype=torch.float64, device=device)
    sums = values.clone()
    maxima = values.clone()
    minima = values.clone()
    if use_ddp:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        dist.all_reduce(minima, op=dist.ReduceOp.MIN)
    world_size = distributed_world_size()
    result: dict[str, float] = {}
    for index, key in enumerate(keys[:-1]):
        result[f"{key}_avg"] = sums[index].item() / world_size
        result[f"{key}_max"] = maxima[index].item()
    throughput_index = len(keys) - 1
    result["samples_per_second_effective"] = (
        local_batch_size * world_size / max(result["step_ms_max"] / 1000.0, 1e-9)
    )
    result["samples_per_second_sum_rank_rates"] = sums[throughput_index].item()
    result["samples_per_second_avg_per_rank"] = sums[throughput_index].item() / world_size
    result["samples_per_second_min_per_rank"] = minima[throughput_index].item()
    result["samples_per_second_max_per_rank"] = maxima[throughput_index].item()
    return result


def aggregate_gpu_values(
    local_values: dict[str, float], *, device: torch.device, use_ddp: bool
) -> dict[str, float]:
    """Aggregate one GPU monitor summary from every rank."""
    keys = (
        "gpu_util_avg",
        "gpu_util_max",
        "memory_util_avg",
        "memory_used_mib_max",
        "power_w_avg",
        "temperature_c_max",
    )
    valid = 1.0 if local_values else 0.0
    values = torch.tensor(
        [float(local_values.get(key, 0.0)) for key in keys] + [valid],
        dtype=torch.float64,
        device=device,
    )
    sums = values.clone()
    maxima = values.clone()
    util_min = torch.tensor(
        float(local_values.get("gpu_util_avg", float("inf"))), dtype=torch.float64, device=device
    )
    if use_ddp:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        dist.all_reduce(util_min, op=dist.ReduceOp.MIN)
    valid_count = int(sums[-1].item())
    if valid_count == 0:
        return {}
    return {
        "gpu_count": float(valid_count),
        "gpu_util_avg": sums[0].item() / valid_count,
        "gpu_util_avg_min": util_min.item(),
        "gpu_util_max": maxima[1].item(),
        "memory_util_avg": sums[2].item() / valid_count,
        "memory_used_mib_max": maxima[3].item(),
        "power_w_avg": sums[4].item() / valid_count,
        "temperature_c_max": maxima[5].item(),
    }


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main or not checkpoint_due(global_step, config):
        return

    # Create temporary directory for atomic checkpoint saving
    final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
    tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

    # Remove any existing temp directory and create new one
    if tmp_ckpt_dir.exists():
        shutil.rmtree(tmp_ckpt_dir)
    tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save model state using safetensors (handle shared tensors)
    model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

    # Save optimizer state using PyTorch format
    torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

    # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
    metadata = {
        "global_step": global_step,
        "config": dataclasses.asdict(config),
        "timestamp": time.time(),
    }
    torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

    # save norm stats
    norm_stats = data_config.norm_stats
    if norm_stats is not None and data_config.asset_id is not None:
        _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

    # Atomically move temp directory to final location
    if final_ckpt_dir.exists():
        shutil.rmtree(final_ckpt_dir)
    tmp_ckpt_dir.rename(final_ckpt_dir)

    logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

    # Log checkpoint to wandb
    if config.wandb_enabled:
        wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def start_cuda_phase(
    name: str,
    events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]],
    *,
    profiling_enabled: bool,
    device: torch.device,
) -> float:
    cpu_started = time.perf_counter()
    if profiling_enabled and device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        events[name] = (start_event, end_event)
    return cpu_started


def end_cuda_phase(
    name: str, cpu_started: float, events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]]
) -> float:
    if name in events:
        events[name][1].record()
    return (time.perf_counter() - cpu_started) * 1000.0


def _train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    rank = distributed_rank()
    world_size = distributed_world_size()
    is_main = rank == 0
    set_seed(config.seed, local_rank)
    _profiling.set_enabled(enabled=config.profiling_enabled)
    log_rank_phase(f"train loop initialized global_rank={rank} world_size={world_size}")

    if config.batch_size < world_size or config.batch_size % world_size != 0:
        raise ValueError(
            f"Global batch size ({config.batch_size}) must be positive and divisible by world size ({world_size})."
        )

    config = resolve_timestamped_run_config(config, use_ddp=use_ddp, rank=rank)
    log_rank_phase(f"run directory resolved directory={config.checkpoint_dir}")

    # Initialize checkpoint directory and wandb
    resuming = False
    checkpoint_setup_error: Exception | None = None
    if is_main:
        try:
            if config.resume:
                # Find checkpoint directory based on experiment name
                exp_checkpoint_dir = config.checkpoint_dir
                if exp_checkpoint_dir.exists():
                    # Use validation to find the latest working checkpoint
                    latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
                    if latest_step is not None:
                        resuming = True
                        logging.info(
                            f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} "
                            f"at step {latest_step}"
                        )
                    else:
                        raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
                else:
                    raise FileNotFoundError(
                        f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume"
                    )
            elif config.overwrite and config.checkpoint_dir.exists():
                shutil.rmtree(config.checkpoint_dir)
                logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")
            elif config.checkpoint_dir.exists():
                raise FileExistsError(
                    f"Checkpoint directory {config.checkpoint_dir} already exists. "
                    "Set OPENPI_RESUME=1 to resume or OPENPI_OVERWRITE=1 to replace it."
                )

            if not resuming:
                config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created experiment checkpoint directory: {config.checkpoint_dir}")
            else:
                logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")
        except Exception as exc:  # Propagate the same setup failure to every rank.
            checkpoint_setup_error = exc
    synchronize_rank0_error(checkpoint_setup_error, phase="checkpoint setup", use_ddp=use_ddp)
    resuming = config.resume
    log_rank_phase(f"checkpoint setup complete resume={resuming} directory={config.checkpoint_dir}")

    profile_dir = config.checkpoint_dir / "profiling"
    profile_metrics_path = profile_dir / "step_metrics.jsonl"
    rank_profile_dir = profile_dir / "ranks" / f"node_{node_rank():03d}_rank_{rank:05d}"
    rank_step_metrics_path = rank_profile_dir / "step_metrics.jsonl"
    rank_data_metrics_path = rank_profile_dir / "data_metrics.jsonl"
    gpu_monitor = None
    if config.profiling_enabled:
        if is_main:
            profile_dir.mkdir(parents=True, exist_ok=True)
        if use_ddp:
            dist.barrier()
        rank_profile_dir.mkdir(parents=True, exist_ok=True)
        gpu_index = visible_gpu_for_local_rank(local_rank)
        gpu_monitor = _profiling.GpuMonitor(
            rank_profile_dir / f"gpu_local_{local_rank}_metrics.csv",
            gpu_index,
            config.gpu_monitor_interval_s,
        )
        gpu_monitor.start()
        logging.info(
            "Profiling enabled for rank=%d local_rank=%d gpu=%s; outputs: %s",
            rank,
            local_rank,
            gpu_index,
            rank_profile_dir,
        )
        if config.num_workers > 0:
            logging.info(
                "Detailed DataLoader counters will be collected in workers and merged into each rank's "
                "data_metrics.jsonl. Worker total times overlap and can exceed wall-clock data_wait."
            )

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    log_rank_phase("building distributed dataset and DataLoader")
    loader, data_config = build_datasets(config)
    log_rank_phase("distributed dataset and DataLoader ready")

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    log_rank_phase("constructing model and moving it to the selected device")
    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    log_rank_phase("model construction complete")

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    # Load weights from weight_loader if specified (for fine-tuning)
    # In DDP, rank 0 loads once before wrapping and DDP broadcasts its parameters.
    weight_load_error: Exception | None = None
    log_rank_phase("base weight loading/synchronization begin")
    if config.pytorch_weight_path is not None and (not use_ddp or is_main):
        try:
            logging.info(f"Loading weights from: {config.pytorch_weight_path}")
            model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
            safetensors.torch.load_model(model, model_path)
            logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")
        except Exception as exc:  # Propagate the same load failure to every rank.
            weight_load_error = exc
    synchronize_rank0_error(weight_load_error, phase="base weight loading", use_ddp=use_ddp)
    log_rank_phase("base weight loading/synchronization complete")
    if use_ddp:
        log_rank_phase("DistributedDataParallel wrapping begin")
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            static_graph=world_size >= 8,
        )
        log_rank_phase("DistributedDataParallel wrapping complete")

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    log_rank_phase("optimizer construction complete")

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        log_rank_phase("checkpoint resume loading begin")
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")
        log_rank_phase(f"checkpoint resume loading complete global_step={global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    torch_profiler = None
    if config.profiling_enabled and config.profiling_torch_trace:
        trace_dir = rank_profile_dir / "torch_trace"
        profiler_activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            profiler_activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch_profiler = torch.profiler.profile(
            activities=profiler_activities,
            schedule=torch.profiler.schedule(
                wait=config.profiling_warmup_steps,
                warmup=1,
                active=config.profiling_trace_active_steps,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        torch_profiler.start()
        logging.info("PyTorch trace enabled for rank=%d; trace directory: %s", rank, trace_dir)

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    data_iter = iter(loader)
    if config.dataloader_prewarm:
        prewarm_started = time.perf_counter()
        prewarm_batch = next(data_iter)
        prewarm_elapsed = time.perf_counter() - prewarm_started
        logging.info("DataLoader pre-warm produced one local batch in %.3fs on rank=%d", prewarm_elapsed, rank)
        if use_ddp:
            dist.barrier()
        data_iter = itertools.chain((prewarm_batch,), data_iter)
        log_rank_phase(f"DataLoader pre-warm complete elapsed={prewarm_elapsed:.3f}s")
    log_rank_phase(f"training iteration ready initial_global_step={global_step}")
    logging_interval = config.profiling_log_interval if config.profiling_enabled else config.log_interval
    while global_step < config.num_train_steps:
        step_started = time.perf_counter()
        data_started = time.perf_counter()
        observation, actions = next(data_iter)
        data_wait_ms = (time.perf_counter() - data_started) * 1000.0

        cuda_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}

        # The unified data loader returns (observation, actions) tuple.
        phase_started = start_cuda_phase("h2d", cuda_events, profiling_enabled=config.profiling_enabled, device=device)
        observation = jax.tree.map(lambda x: x.to(device, non_blocking=True), observation)
        actions = actions.to(device=device, dtype=torch.float32, non_blocking=True)
        phase_cpu_ms = {"h2d": end_cuda_phase("h2d", phase_started, cuda_events)}

        for pg in optim.param_groups:
            pg["lr"] = lr_schedule(global_step)

        phase_started = start_cuda_phase(
            "forward", cuda_events, profiling_enabled=config.profiling_enabled, device=device
        )
        losses = model(observation, actions)
        if isinstance(losses, list | tuple):
            losses = torch.stack(losses)
        elif not isinstance(losses, torch.Tensor):
            losses = torch.tensor(losses, device=device, dtype=torch.float32)
        loss = losses.mean()
        phase_cpu_ms["forward"] = end_cuda_phase("forward", phase_started, cuda_events)

        phase_started = start_cuda_phase(
            "backward", cuda_events, profiling_enabled=config.profiling_enabled, device=device
        )
        loss.backward()
        phase_cpu_ms["backward"] = end_cuda_phase("backward", phase_started, cuda_events)

        if global_step < 5 and is_main and torch.cuda.is_available():
            log_memory_usage(device, global_step, "after_backward")

        phase_started = start_cuda_phase(
            "optimizer", cuda_events, profiling_enabled=config.profiling_enabled, device=device
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)
        optim.step()
        optim.zero_grad(set_to_none=True)
        phase_cpu_ms["optimizer"] = end_cuda_phase("optimizer", phase_started, cuda_events)

        for param in model.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad = None

        profile_values: dict[str, float] = {}
        if config.profiling_enabled:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                profile_values = {
                    f"{name}_ms": events[0].elapsed_time(events[1]) for name, events in cuda_events.items()
                }
            else:
                profile_values = {f"{name}_ms": value for name, value in phase_cpu_ms.items()}
            profile_values["data_wait_ms"] = data_wait_ms
            profile_values["step_ms"] = (time.perf_counter() - step_started) * 1000.0
            profile_values["samples_per_second"] = effective_batch_size / max(profile_values["step_ms"] / 1000.0, 1e-9)

        if is_main or config.profiling_enabled:
            infos.append(
                {
                    "loss": loss.item(),
                    "learning_rate": optim.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    **profile_values,
                }
            )

        if config.profiling_enabled and global_step + 1 == config.profiling_warmup_steps:
            _profiling.snapshot(reset=True)
            if gpu_monitor is not None:
                gpu_monitor.summary(reset=True)
            infos = []
            start_time = time.time()

        if infos and (global_step % logging_interval == 0):
            elapsed = time.time() - start_time
            avg_loss = sum(info["loss"] for info in infos) / len(infos)
            avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
            grad_norms = [info["grad_norm"] for info in infos if info.get("grad_norm") is not None]
            avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else None
            if is_main:
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} "
                    f"grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

            if config.profiling_enabled and global_step >= config.profiling_warmup_steps:
                profile_keys = (
                    "data_wait_ms",
                    "h2d_ms",
                    "forward_ms",
                    "backward_ms",
                    "optimizer_ms",
                    "step_ms",
                    "samples_per_second",
                )
                local_averages = {
                    key: sum(float(info.get(key, 0.0)) for info in infos) / len(infos) for key in profile_keys
                }
                local_gpu_values = gpu_monitor.summary(reset=True) if gpu_monitor is not None else {}
                local_profile_payload = {
                    "step": global_step,
                    "rank": rank,
                    "local_rank": local_rank,
                    "node_rank": node_rank(),
                    **local_averages,
                    **local_gpu_values,
                }
                with rank_step_metrics_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(local_profile_payload, ensure_ascii=False) + "\n")
                data_events = _profiling.snapshot(reset=True)
                data_payload = {
                    "step": global_step,
                    "rank": rank,
                    "events": data_events,
                }
                with rank_data_metrics_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(data_payload, ensure_ascii=False) + "\n")

                global_profile_values = aggregate_profile_values(
                    local_averages,
                    local_batch_size=effective_batch_size,
                    device=device,
                    use_ddp=use_ddp,
                )
                global_gpu_values = aggregate_gpu_values(
                    local_gpu_values,
                    device=device,
                    use_ddp=use_ddp,
                )
                if is_main:
                    profile_payload = {
                        "step": global_step,
                        "world_size": world_size,
                        **global_profile_values,
                        **global_gpu_values,
                    }
                    logging.info(
                        "profile step=%d global_samples/s=%.2f slowest_step=%.1fms "
                        "data_wait(avg/max)=%.1f/%.1fms forward(avg/max)=%.1f/%.1fms "
                        "backward(avg/max)=%.1f/%.1fms gpu_util(avg/min/max)=%.1f/%.1f/%.1f%%",
                        global_step,
                        global_profile_values["samples_per_second_effective"],
                        global_profile_values["step_ms_max"],
                        global_profile_values["data_wait_ms_avg"],
                        global_profile_values["data_wait_ms_max"],
                        global_profile_values["forward_ms_avg"],
                        global_profile_values["forward_ms_max"],
                        global_profile_values["backward_ms_avg"],
                        global_profile_values["backward_ms_max"],
                        global_gpu_values.get("gpu_util_avg", float("nan")),
                        global_gpu_values.get("gpu_util_avg_min", float("nan")),
                        global_gpu_values.get("gpu_util_max", float("nan")),
                    )
                    top_data_events = sorted(
                        data_events.items(),
                        key=lambda item: float(item[1].get("total_ms", 0.0)),
                        reverse=True,
                    )[:10]
                    if top_data_events:
                        logging.info(
                            "dataloader profile step=%d rank=0 top=%s",
                            global_step,
                            "; ".join(
                                f"{name}:total={float(values['total_ms']):.1f}ms/"
                                f"count={int(values['count'])}/max={float(values['max_ms']):.1f}ms/"
                                f"MBps={float(values['throughput_mb_s']):.1f}"
                                for name, values in top_data_events
                            ),
                        )
                    with profile_metrics_path.open("a", encoding="utf-8") as output:
                        output.write(json.dumps(profile_payload, ensure_ascii=False) + "\n")

            if is_main and config.wandb_enabled:
                log_payload = {
                    "loss": avg_loss,
                    "learning_rate": avg_lr,
                    "step": global_step,
                    "time_per_step": elapsed / logging_interval,
                }
                if avg_grad_norm is not None:
                    log_payload["grad_norm"] = avg_grad_norm
                wandb.log(log_payload, step=global_step)

            start_time = time.time()
            infos = []

        global_step += 1
        if torch_profiler is not None:
            torch_profiler.step()
        save_checkpoint(model, optim, global_step, config, is_main, data_config)
        if use_ddp and checkpoint_due(global_step, config):
            dist.barrier()

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
            )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    if torch_profiler is not None:
        torch_profiler.stop()
    if gpu_monitor is not None:
        gpu_monitor.stop()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()



def train_loop(config: _config.TrainConfig):
    """Run training and always release an initialized distributed process group."""
    try:
        _train_loop(config)
    finally:
        cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
