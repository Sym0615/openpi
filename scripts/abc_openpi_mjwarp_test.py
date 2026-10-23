"""Verify the single-world adapter delegates to ABC's native step API."""

# ruff: noqa: SLF001
from types import SimpleNamespace

import numpy as np
import torch

from scripts.abc_openpi_mjwarp import MJWarpEnv


def test_native_step_and_trace_state_are_synchronized():
    class Result:
        def scalar_reward(self):
            return 0.25

        def scalar_success(self):
            return False

        def to_info(self, *, squeeze):
            assert squeeze
            return {"reward": 0.25, "success": False}

    class NativeBatch:
        control_decimation = 10
        qpos = np.arange(16, dtype=np.float32)[None]
        _runtime = SimpleNamespace(d_warp=SimpleNamespace(time=SimpleNamespace(numpy=lambda: np.array([0.5]))))

        def step(self, actions):
            assert actions.shape == (1, 14)
            return Result()

        def get_obs(self):
            return {
                "state": np.zeros((1, 14)),
                "images": {"top": torch.zeros((1, 3, 2, 2), dtype=torch.uint8)},
                "prompt": ["test"],
            }

    env = object.__new__(MJWarpEnv)
    env.batch = NativeBatch()
    env.base_env = SimpleNamespace(
        data=SimpleNamespace(qpos=np.zeros(16), time=0),
        cur_step=0,
        _episode_status=lambda info: (False, False),
    )
    obs, reward, terminated, truncated, info = env.step(np.zeros(14))
    assert reward == 0.25
    assert not terminated
    assert not truncated
    assert env.base_env.cur_step == 1
    assert env.data.time == 0.5
    np.testing.assert_array_equal(env.data.qpos, env.batch.qpos[0])
    assert obs["state"].shape == (14,)
    assert obs["images"]["top"].shape == (3, 2, 2)
    assert info["task_eval"]["reward"] == 0.25
