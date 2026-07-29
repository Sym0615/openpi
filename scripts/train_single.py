"""Build and run the InternA1 PyTorch config on one local GPU."""

import dataclasses
import datetime
import json
import os
from pathlib import Path

import torch
import train_pytorch

import openpi.training.config as _config


def main() -> None:
    openpi_workdir = Path(os.environ["OPENPI_WORKDIR"])
    run_root = Path(os.environ["OPENPI_RUN_ROOT"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise ValueError(
            f"scripts/train_single.py only supports one process, but WORLD_SIZE={world_size}. "
            "Use scripts/train_distributed.sh for distributed training."
        )

    visible_devices = [device.strip() for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
    visible_devices = [device for device in visible_devices if device]
    if len(visible_devices) != 1:
        raise ValueError(
            "Single-GPU training requires CUDA_VISIBLE_DEVICES to expose exactly one GPU, "
            f"got {os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}."
        )

    weight_path = Path(os.environ.get("OPENPI_WEIGHT_PATH", "/datahub/public/model/pi05_base_pytorch_bf16"))
    model_file = weight_path / "model.safetensors"
    model_config_file = weight_path / "config.json"

    if not model_file.is_file():
        raise FileNotFoundError(f"Missing PyTorch model weights: {model_file}")

    if model_config_file.is_file():
        weight_config = json.loads(model_config_file.read_text(encoding="utf-8"))
        if int(weight_config.get("action_dim", 32)) != 32:
            raise ValueError(f"Expected action_dim=32 in {model_config_file}, got {weight_config.get('action_dim')}")
        if int(weight_config.get("action_horizon", 16)) != 16:
            print(
                f"[train] weight action_horizon={weight_config.get('action_horizon')}; "
                "InternA1 training uses action_horizon=16. Weight compatibility will be checked during load."
            )

    def env_int(name: str, default: int) -> int:
        return int(os.environ.get(name, default))

    def env_float(name: str, default: float) -> float:
        return float(os.environ.get(name, default))

    def env_bool(name: str, *, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        if value.lower() in {"1", "true", "yes", "on"}:
            return True
        if value.lower() in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{name} must be a boolean value, got {value!r}")

    run_id = os.environ.get("OPENPI_RUN_ID")
    run_id = run_id or datetime.datetime.now(datetime.UTC).strftime("single_%Y%m%d_%H%M%S")
    os.environ["OPENPI_RUN_ID"] = run_id
    profiling_enabled = env_bool("OPENPI_PROFILE", default=False)
    torch_trace_enabled = env_bool("OPENPI_TORCH_TRACE", default=False)
    if torch_trace_enabled and not profiling_enabled:
        raise ValueError("OPENPI_TORCH_TRACE requires OPENPI_PROFILE=1")
    resume = env_bool("OPENPI_RESUME", default=False)
    os.environ["OPENPI_TIMESTAMPED_RUN_DIR"] = "1"

    config = _config.get_config("pi05_interna1")
    config = dataclasses.replace(
        config,
        exp_name=run_id,
        checkpoint_base_dir=str(run_root),
        pytorch_weight_path=str(weight_path),
        pytorch_training_precision="bfloat16",
        num_train_steps=env_int("OPENPI_TRAIN_STEPS", 120 if profiling_enabled else 10000),
        batch_size=env_int("OPENPI_BATCH_SIZE", 64),
        save_interval=env_int("OPENPI_SAVE_INTERVAL", 1000),
        log_interval=env_int("OPENPI_LOG_INTERVAL", 1),
        overwrite=env_bool("OPENPI_OVERWRITE", default=False),
        resume=resume,
        wandb_enabled=env_bool("OPENPI_WANDB", default=False),
        num_workers=env_int("OPENPI_NUM_WORKERS", 16),
        profiling_enabled=profiling_enabled,
        profiling_warmup_steps=env_int("OPENPI_PROFILE_WARMUP_STEPS", 10),
        profiling_log_interval=env_int("OPENPI_PROFILE_LOG_INTERVAL", 1),
        profiling_torch_trace=torch_trace_enabled,
        profiling_trace_active_steps=env_int("OPENPI_TRACE_ACTIVE_STEPS", 20),
        gpu_monitor_interval_s=env_float("OPENPI_GPU_MONITOR_INTERVAL", 1.0),
    )

    print(f"[train] workdir={openpi_workdir}")
    print(f"[train] run_dir_template={config.checkpoint_dir.parent}/{run_id}_<UTC start time>")
    print(f"[train] weights={config.pytorch_weight_path}")
    print(f"[train] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print(
        f"[train] distributed=False world_size=1 "
        f"profiling={config.profiling_enabled} torch_trace={config.profiling_torch_trace} "
        f"steps={config.num_train_steps} batch_size={config.batch_size} "
        f"num_workers={config.num_workers} resume={config.resume} overwrite={config.overwrite}"
    )

    if os.environ.get("OPENPI_DRY_RUN") == "1":
        print("[train] dry run complete; CUDA and training were not started")
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Single-GPU training requires CUDA, but torch.cuda.is_available() is false. "
            "Check the NVIDIA driver, CUDA runtime, and CUDA_VISIBLE_DEVICES."
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible CUDA device, got {torch.cuda.device_count()}.")
    if config.pytorch_training_precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support bfloat16 training.")

    print(f"[train] gpu={torch.cuda.get_device_name(0)} bf16_supported={torch.cuda.is_bf16_supported()}")

    train_pytorch.init_logging()
    train_pytorch.train_loop(config)


if __name__ == "__main__":
    main()
