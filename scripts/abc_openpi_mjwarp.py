"""Single-world rollout adapter over ABC's native MJWarp environment."""

# ruff: noqa: SLF001
import numpy as np


class MJWarpEnv:
    def __init__(self, base_env, *, gpu_id=0):
        from abc_sim.batched_env import BatchedWarpYAMEnv
        import warp as wp

        if not wp.is_cuda_available():
            raise RuntimeError("MJWarp evaluation requires a visible NVIDIA GPU and a working CUDA driver.")
        self.base_env = base_env
        self.device = f"cuda:{gpu_id}"
        self.batch = BatchedWarpYAMEnv(
            base_env,
            num_worlds=1,
            camera_height=168,
            camera_width=224,
            gpu_id=gpu_id,
        )
        self._last_eval = None

    @property
    def model(self):
        return self.base_env.model

    @property
    def data(self):
        return self.base_env.data

    @property
    def _control_decimation(self):
        return self.batch.control_decimation

    def _sync_trace_state(self):
        self.data.qpos[:] = self.batch.qpos[0]
        self.data.time = float(self.batch._runtime.d_warp.time.numpy()[0])

    @staticmethod
    def _single_observation(obs):
        return {
            "state": np.asarray(obs["state"][0], dtype=np.float32).copy(),
            "images": {name: pixels[0].detach().cpu().numpy().copy() for name, pixels in obs["images"].items()},
            "masks": dict.fromkeys(obs["images"], True),
            "prompt": obs["prompt"][0],
        }

    def reset(self, *, seed, randomize=True, options=None):
        if not randomize:
            raise ValueError("ABC's native MJWarp reset requires randomize=True.")
        obs = self.batch.reset([seed], options=options)
        self._sync_trace_state()
        self._last_eval = self.batch.evaluate()
        return self._single_observation(obs), {
            "randomization": self.batch.randomization[0],
            "seed": self.batch.seeds[0],
        }

    def get_state(self):
        return np.asarray(self.batch.state()[0], dtype=np.float32).copy()

    def get_obs(self):
        return self._single_observation(self.batch.get_obs())

    def evaluate_task(self):
        return self._last_eval

    def step(self, action, *, render_obs=True):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("MJWarp expects one finite 14D absolute joint/gripper action.")
        # The native API is step(), not step_actions().
        self._last_eval = self.batch.step(action[None])
        if self._last_eval is None:
            raise ValueError("The selected task has no evaluator.")
        self._sync_trace_state()
        self.base_env.cur_step += 1
        reward = self._last_eval.scalar_reward()
        info = {
            "task_reward": reward,
            "task_success": self._last_eval.scalar_success(),
            "task_eval": self._last_eval.to_info(squeeze=True),
        }
        terminated, truncated = self.base_env._episode_status(info)
        return self.get_obs() if render_obs else None, reward, terminated, truncated, info

    def close(self):
        self.batch.close()
