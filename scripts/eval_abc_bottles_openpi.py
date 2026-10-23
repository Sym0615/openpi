"""Evaluate ABC-Sim checkpoints with matching state projection and action semantics."""

import argparse
import contextlib
import dataclasses
from fractions import Fraction
import json
import logging
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import traceback

from abc_openpi_bridge import CAMERA_MAP
from abc_openpi_bridge import actions_to_abc
from abc_openpi_bridge import observation_to_openpi
from abc_openpi_bridge import receive_packet
from abc_openpi_bridge import send_packet
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    "/data/sunyiming/openpi/abc_sim_bottles/checkpoints/pi05_abc_sim_bottles/"
    "abc-bottles-ft-20261021-013046_20261020_173235_873127/5000"
)
PROMPT = "sim_put the plastic bottles in the bin"


def load_policy(args):
    import safetensors.torch
    import torch

    from openpi import transforms
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.policies.policy import Policy
    from openpi.training import checkpoints
    from openpi.training import config
    from openpi.training import petrel_lerobot_reader as reader

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "14")))
    torch.set_num_interop_threads(1)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but OpenPI cannot see a CUDA device.")
    precision = "bfloat16" if device == "cuda" else "float32"
    train_config = config.get_config(args.config_name)
    dataset_config = reader.read_openpi_petrel_config(train_config.data.dataset.dataset_config)
    adapters = [value["canonical_adapter"] for value in dataset_config["embodiments"].values()]
    if len(adapters) != 1:
        raise ValueError("ABC evaluation requires exactly one control profile.")
    adapter = adapters[0]
    stats_group = f"dataset/{train_config.data.repo_id}"
    model_config = dataclasses.replace(train_config.model, dtype=precision, pytorch_compile_mode=None)
    data_config = train_config.data.create(train_config.assets_dirs, model_config)
    full_stats = checkpoints.load_norm_stats(Path(args.checkpoint) / "assets", data_config.asset_id)
    selected_stats = transforms.select_norm_stats(full_stats, stats_group)
    if selected_stats["state"].q01.shape != (32,) or selected_stats["actions"].q01.shape != (30, 32):
        raise ValueError("Checkpoint norm-stat shapes do not match the selected training configuration.")
    logging.info("BUILD_MODEL device=%s precision=%s", device, precision)
    model = PI0Pytorch(model_config)
    missing, unexpected = safetensors.torch.load_model(
        model, str(Path(args.checkpoint) / "model.safetensors"), strict=True
    )
    if missing or unexpected:
        raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    policy = Policy(
        model,
        transforms=[
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.NormalizeByGroup(full_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(selected_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *data_config.repack_transforms.outputs,
        ],
        sample_kwargs={"num_steps": args.denoise_steps},
        is_pytorch=True,
        pytorch_device=device,
    )
    return (
        policy,
        adapter,
        {
            "device": device,
            "precision": precision,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "checkpoint": args.checkpoint,
            "config_name": args.config_name,
            "stats_group": stats_group,
            "joint_delta_indices": adapter["delta_from_state_indices"],
            "action_horizon": 30,
            "denoise_steps": args.denoise_steps,
        },
    )


def policy_worker(args):
    # Reserve stdout exclusively for framed packets, including during third-party imports.
    output = sys.stdout.buffer
    with contextlib.redirect_stdout(sys.stderr):
        try:
            policy, adapter, metadata = load_policy(args)
            import torch

            send_packet(output, {"kind": "ready", **metadata})
            rng = np.random.default_rng(args.policy_seed)
            while True:
                request, arrays = receive_packet(sys.stdin.buffer)
                if request["kind"] == "close":
                    break
                obs = {
                    "state": arrays["state"],
                    "images": {name: arrays[name] for name in CAMERA_MAP},
                    "masks": dict.fromkeys(CAMERA_MAP, True),
                    "prompt": request["prompt"],
                }
                sample = observation_to_openpi(obs, canonical_adapter=adapter, stats_group=metadata["stats_group"])
                if args.config_name == "pi05_abc_sim_bottles" and sample["prompt"] != PROMPT:
                    raise ValueError("The live prompt differs from the bottles training prompt.")
                noise = rng.standard_normal((30, 32)).astype(np.float32)
                started = time.monotonic()
                with torch.inference_mode():
                    result = policy.infer(sample, noise=noise)
                if metadata["device"] == "cuda":
                    torch.cuda.synchronize()
                actions = np.asarray(result["actions"])
                actions_to_abc(
                    actions, reference_state=obs["state"], delta_indices=metadata["joint_delta_indices"]
                )  # Validate the response before returning it.
                send_packet(
                    output,
                    {"kind": "prediction", "infer_seconds": time.monotonic() - started},
                    {"actions": actions},
                )
        except Exception as exc:
            traceback.print_exc()
            send_packet(output, {"kind": "error", "message": f"{type(exc).__name__}: {exc}"})
            raise


class PolicyProcess:
    def __init__(self, args, log_path):
        self.log = log_path.open("wb")
        command = [
            args.openpi_python,
            "-u",
            str(Path(__file__).resolve()),
            "--worker",
            "--config-name",
            args.config_name,
            "--checkpoint",
            args.checkpoint,
            "--device",
            args.device,
            "--denoise-steps",
            str(args.denoise_steps),
            "--policy-seed",
            str(args.policy_seed),
        ]
        self.process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            env={**os.environ, "JAX_PLATFORMS": "cpu"},
        )
        self.responses = queue.Queue()
        self.config_name = args.config_name
        threading.Thread(target=self._read_responses, daemon=True).start()

    def _read_responses(self):
        try:
            while True:
                self.responses.put(receive_packet(self.process.stdout))
        except Exception as exc:
            self.responses.put(exc)

    def receive(self, timeout):
        try:
            result = self.responses.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("OpenPI response timed out; inspect policy_worker.log.") from exc
        if isinstance(result, Exception):
            raise RuntimeError("OpenPI worker stopped; inspect policy_worker.log.") from result
        metadata, arrays = result
        if metadata["kind"] == "error":
            raise RuntimeError(metadata["message"])
        return metadata, arrays

    def infer(self, obs):
        prompt = str(obs["prompt"])
        if self.config_name == "pi05_abc_sim" and not prompt.startswith("sim_"):
            prompt = "sim_" + prompt
        send_packet(
            self.process.stdin,
            {"kind": "infer", "prompt": prompt},
            {"state": obs["state"], **obs["images"]},
        )
        metadata, arrays = self.receive(180)
        if metadata["kind"] != "prediction":
            raise RuntimeError(f"Unexpected OpenPI response: {metadata}")
        return arrays["actions"], metadata["infer_seconds"]

    def close(self):
        try:
            if self.process.poll() is None:
                send_packet(self.process.stdin, {"kind": "close"})
                self.process.wait(timeout=15)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        finally:
            self.process.stdin.close()
            self.process.stdout.close()
            self.log.close()


class RolloutVideo:
    def __init__(self, path, rate):
        import av

        self.av = av
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=rate)
        self.stream.width, self.stream.height = 1344, 368
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "20", "preset": "fast"}
        self.count = 0

    def write(self, obs, step, sim_time, reward):
        from PIL import Image
        from PIL import ImageDraw

        cameras = np.concatenate([obs["images"][name].transpose(1, 2, 0) for name in CAMERA_MAP], axis=1)
        sheet = Image.new("RGB", (1344, 368), "black")
        sheet.paste(Image.fromarray(cameras).resize((1344, 336), Image.Resampling.NEAREST), (0, 32))
        ImageDraw.Draw(sheet).text(
            (12, 8),
            f"OpenPI -> ABC | TOP / LEFT / RIGHT | step={step} t={sim_time:.2f}s reward={reward:.3f}",
            fill="white",
        )
        frame = self.av.VideoFrame.from_ndarray(np.asarray(sheet), format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.count += 1

    def close(self):
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()


def rollout(args):
    sys.path.insert(0, args.abc_root)
    import abc_sim
    import mujoco
    from PIL import Image

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    worker = env = video = None
    started = time.monotonic()
    try:
        logging.info(
            "ABC_ENV_START mujoco=%s physics=%s renderer=%s",
            mujoco.__version__,
            args.physics_backend,
            "mjwarp" if args.physics_backend == "mjwarp" else os.environ.get("MUJOCO_GL"),
        )
        env = abc_sim.make_env(
            task=args.task,
            prompt=args.prompt
            if args.prompt is not None
            else (PROMPT if args.config_name == "pi05_abc_sim_bottles" else None),
            render_cameras=args.physics_backend == "mujoco",
            camera_backend="mujoco",
            camera_height=168,
            camera_width=224,
            max_episode_steps=args.num_chunks * args.execute_steps,
            terminate_on_success=True,
        )
        if args.physics_backend == "mjwarp":
            from abc_openpi_mjwarp import MJWarpEnv

            env = MJWarpEnv(env, gpu_id=args.gpu_id)
        obs, reset_info = env.reset(seed=args.seed, randomize=True)
        observation_to_openpi(obs)
        initial_eval = env.evaluate_task().to_info(squeeze=True)
        if "num_active_bottles" in initial_eval and initial_eval["num_active_bottles"] <= 0:
            raise RuntimeError("ABC reset produced no active bottles.")
        sampled_randomization = reset_info.get("randomization")
        randomization = sampled_randomization.to_dict() if sampled_randomization is not None else {}
        (out / "randomization.json").write_text(json.dumps(randomization, indent=2) + "\n")
        for name, rgb in obs["images"].items():
            Image.fromarray(rgb.transpose(1, 2, 0)).save(out / f"initial_{name}.png")
        logging.info("ABC_ENV_OK task=%s state=14D images=3xCHW uint8", args.task)
        if args.physics_backend == "mjwarp":
            logging.info("GPU_SIM_READY physics=mjwarp renderer=mjwarp device=%s", env.device)
        worker = PolicyProcess(args, out / "policy_worker.log")
        policy_metadata, _ = worker.receive(600)
        if policy_metadata["kind"] != "ready":
            raise RuntimeError("OpenPI worker did not return a ready response.")
        logging.info("OPENPI_READY %s", policy_metadata)
        control_decimation = env._control_decimation  # noqa: SLF001 - ABC exposes no public accessor.
        control_dt = float(env.model.opt.timestep) * control_decimation
        video = RolloutVideo(
            out / "rollout_3cam.mp4",
            Fraction(1 / (control_dt * args.video_every_n_actions)).limit_denominator(10000),
        )
        video.write(obs, 0, float(env.data.time), initial_eval["reward"])
        states, qpos = [obs["state"].copy()], [env.data.qpos.copy()]
        predictions, executed, metrics = [], [], []
        reward = float(initial_eval["reward"])
        max_reward = reward
        steps = 0
        ever_success = bool(initial_eval["success"])
        terminated = truncated = False
        with (out / "chunk_metrics.jsonl").open("w", buffering=1) as metric_file:
            for chunk_index in range(args.num_chunks):
                raw_actions, infer_seconds = worker.infer(obs)
                actions, clipped = actions_to_abc(
                    raw_actions, reference_state=obs["state"], delta_indices=policy_metadata["joint_delta_indices"]
                )
                predictions.append(raw_actions.copy())
                actual_steps = 0
                chunk_started = time.monotonic()
                chunk_actions = actions[: args.execute_steps]
                for action_index, action in enumerate(chunk_actions):
                    record_frame = (steps + 1) % args.video_every_n_actions == 0
                    render_obs = record_frame or action_index == len(chunk_actions) - 1
                    next_obs, reward, terminated, truncated, info = env.step(action, render_obs=render_obs)
                    state = env.get_state()
                    if not np.isfinite(state).all() or not np.isfinite(env.data.qpos).all():
                        raise RuntimeError("Nonfinite ABC physics state.")
                    steps += 1
                    actual_steps += 1
                    max_reward = max(max_reward, reward)
                    executed.append(action.copy())
                    states.append(state.copy())
                    qpos.append(env.data.qpos.copy())
                    if next_obs is not None:
                        obs = next_obs
                    if (terminated or truncated) and next_obs is None:
                        obs = env.get_obs()
                    if record_frame or terminated or truncated:
                        video.write(obs, steps, float(env.data.time), reward)
                    if terminated or truncated:
                        break
                task_eval = info["task_eval"]
                ever_success = ever_success or bool(task_eval["success"])
                task_eval["ever_success"] = bool(task_eval.get("ever_success", ever_success))
                metric = {
                    "chunk": chunk_index,
                    "steps": steps,
                    "executed_steps": actual_steps,
                    "infer_seconds": infer_seconds,
                    "simulation_and_video_seconds": time.monotonic() - chunk_started,
                    "gripper_values_clipped": clipped,
                    "reward": reward,
                    "task_eval": task_eval,
                }
                metrics.append(metric)
                metric_file.write(json.dumps(metric) + "\n")
                logging.info(
                    "CHUNK %d/%d steps=%d infer=%.2fs sim_video=%.2fs reward=%.3f success=%s",
                    chunk_index + 1,
                    args.num_chunks,
                    steps,
                    infer_seconds,
                    metric["simulation_and_video_seconds"],
                    reward,
                    task_eval["ever_success"],
                )
                if terminated or truncated:
                    break
        final_eval = env.evaluate_task().to_info(squeeze=True)
        np.savez_compressed(
            out / "trajectory.npz",
            states_14d=np.stack(states),
            qpos=np.stack(qpos),
            predicted_chunks_32d=np.stack(predictions),
            executed_actions_14d=np.stack(executed),
        )
        video_frames = video.count
        video.close()
        video = None
        summary = {
            "status": "ROLLOUT_OK",
            "task": args.task,
            "prompt": obs["prompt"],
            "seed": args.seed,
            "policy_seed": args.policy_seed,
            "policy": policy_metadata,
            "steps": steps,
            "chunks": len(metrics),
            "execute_steps": args.execute_steps,
            "action_semantics": "absolute joints in radians; absolute grippers in [0,1]",
            "model_joint_delta_indices": policy_metadata["joint_delta_indices"],
            "physics": {
                "backend": args.physics_backend,
                "device": env.device if args.physics_backend == "mjwarp" else "cpu",
                "renderer": "mjwarp" if args.physics_backend == "mjwarp" else "mujoco",
                "mujoco_version": mujoco.__version__,
                "physics_dt": float(env.model.opt.timestep),
                "control_decimation": control_decimation,
                "control_hz": 1 / control_dt,
            },
            "simulated_seconds": steps * control_dt,
            "wall_seconds": time.monotonic() - started,
            "success": bool(final_eval.get("ever_success", ever_success)),
            "final_success": bool(final_eval["success"]),
            "max_reward": max_reward,
            "initial_task_eval": initial_eval,
            "final_task_eval": final_eval,
            "terminated": terminated,
            "truncated": truncated,
            "video": str(out / "rollout_3cam.mp4"),
            "video_frames": video_frames,
            "video_every_n_actions": args.video_every_n_actions,
            "scope": "One newly randomized ABC world, synchronous policy rollout; not an aggregate benchmark.",
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        logging.info("ROLLOUT_OK output=%s success=%s", out, summary["success"])
    except Exception as exc:
        (out / "failure.json").write_text(
            json.dumps({"error": str(exc), "traceback": traceback.format_exc()}, indent=2)
        )
        raise
    finally:
        if video is not None:
            video.close()
        if worker is not None:
            worker.close()
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--config-name", choices=("pi05_abc_sim_bottles", "pi05_abc_sim"), default="pi05_abc_sim_bottles"
    )
    parser.add_argument("--task", default="put_plastic_bottles_in_bin")
    parser.add_argument(
        "--prompt", default=None, help="Override the task prompt; otherwise use the live task instruction."
    )
    parser.add_argument("--abc-root", default="/sunyiming/abc")
    parser.add_argument("--openpi-python", default=str(ROOT / ".venv/bin/python"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--physics-backend", choices=("mujoco", "mjwarp"), default="mujoco")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--video-every-n-actions", type=int, default=1)
    parser.add_argument("--num-chunks", type=int, default=100)
    parser.add_argument("--execute-steps", type=int, default=15)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-seed", type=int, default=42)
    parser.add_argument("--out", default=str(ROOT / "runs/abc_openpi/step5000_seed42"))
    args = parser.parse_args()
    if args.config_name == "pi05_abc_sim" and args.checkpoint == DEFAULT_CHECKPOINT:
        parser.error("--checkpoint is required for pi05_abc_sim; select a checkpoint from the full-data training run.")
    if args.num_chunks < 1 or not 1 <= args.execute_steps <= 30 or args.denoise_steps < 1:
        parser.error("num-chunks/denoise-steps must be positive; execute-steps must be in [1,30].")
    if args.gpu_id < 0 or args.video_every_n_actions < 1:
        parser.error("gpu-id must be nonnegative; video-every-n-actions must be positive.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    if args.worker:
        policy_worker(args)
    else:
        rollout(args)


if __name__ == "__main__":
    main()
