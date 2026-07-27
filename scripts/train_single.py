"""Run a one-step single-GPU InternA1 PyTorch smoke training job."""

import dataclasses
import datetime
import json
import os
from pathlib import Path

import train_pytorch

import openpi.training.config as _config


def main() -> None:
    openpi_workdir = Path(os.environ["OPENPI_WORKDIR"])
    run_root = Path(os.environ["OPENPI_RUN_ROOT"])
    weight_path = Path("/datahub/public/model/pi05_base_pytorch_bf16")
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
                f"[train_single] weight action_horizon={weight_config.get('action_horizon')}; "
                "InternA1 training uses action_horizon=16. Weight compatibility will be checked during load."
            )

    run_id = os.environ.get("OPENPI_RUN_ID") or datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    profiling_enabled = os.environ.get("OPENPI_PROFILE", "0") == "1"
    exp_kind = "profile" if profiling_enabled else "smoke"
    exp_name = f"interna1_pytorch_{exp_kind}_{run_id}"

    def env_int(name: str, default: int) -> int:
        return int(os.environ.get(name, default))

    def env_float(name: str, default: float) -> float:
        return float(os.environ.get(name, default))

    config = _config.get_config("pi05_interna1")
    config = dataclasses.replace(
        config,
        exp_name=exp_name,
        checkpoint_base_dir=str(run_root / "checkpoints_pytorch"),
        pytorch_weight_path=str(weight_path),
        pytorch_training_precision="bfloat16",
        num_train_steps=env_int("OPENPI_TRAIN_STEPS", 120 if profiling_enabled else 10000),
        batch_size=env_int("OPENPI_BATCH_SIZE", 64),
        save_interval=1000,
        log_interval=env_int("OPENPI_LOG_INTERVAL", 1),
        overwrite=True,
        wandb_enabled=False,
        num_workers=env_int("OPENPI_NUM_WORKERS", 0),
        profiling_enabled=profiling_enabled,
        profiling_warmup_steps=env_int("OPENPI_PROFILE_WARMUP_STEPS", 20),
        profiling_log_interval=env_int("OPENPI_PROFILE_LOG_INTERVAL", 10),
        profiling_torch_trace=os.environ.get("OPENPI_TORCH_TRACE", "0") == "1",
        profiling_trace_active_steps=env_int("OPENPI_TRACE_ACTIVE_STEPS", 20),
        gpu_monitor_interval_s=env_float("OPENPI_GPU_MONITOR_INTERVAL", 1.0),
    )

    print(f"[train_single] workdir={openpi_workdir}")
    print(f"[train_single] checkpoint_dir={config.checkpoint_dir}")
    print(f"[train_single] weights={config.pytorch_weight_path}")
    print(f"[train_single] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print(
        f"[train_single] profiling={config.profiling_enabled} torch_trace={config.profiling_torch_trace} "
        f"steps={config.num_train_steps} batch_size={config.batch_size} num_workers={config.num_workers}"
    )

    if os.environ.get("OPENPI_DRY_RUN") == "1":
        print("[train_single] dry run complete; training was not started")
        return

    train_pytorch.init_logging()
    train_pytorch.train_loop(config)


if __name__ == "__main__":
    main()
