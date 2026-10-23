"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import os

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.usd_helper import UsdHelper
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)
import numpy as np
import torch

from omni.isaac.core.utils.types import JointsState as SimJointState  # type: ignore
from omni.isaac.core.utils.stage import get_current_stage  # type: ignore


class CuroboPlanner:
    def __init__(self, robot_cfg: dict, robot_prim_path: str) -> None:
        self.robot_prim_path = robot_prim_path
        self.usd_helper = UsdHelper()
        self.usd_helper.load_stage(get_current_stage())
        self.robot_cfg = robot_cfg
        self.world_cfg = WorldConfig()
        self.tensor_args = TensorDeviceType()
        self.pose_metric = PoseCostMetric.create_grasp_approach_metric(
            offset_position=0.15, tstep_fraction=0.8
        )
        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=7,
            max_attempts=10,
            pose_cost_metric=None,
            enable_finetune_trajopt=True,
            time_dilation_factor=1.0,
        )
        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            self.tensor_args,
            interpolation_dt=0.01,
            collision_activation_distance=0.001,
            trajopt_tsteps=32,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            self_collision_check=True,
            collision_cache={"obb": 3000, "mesh": 3000},
            num_trajopt_seeds=12,
            num_graph_seeds=12,
            optimize_dt=True,
            trajopt_dt=None,
            trim_steps=None,
            project_pose_to_goal_frame=False,
        )
        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.warmup(warmup_js_trajopt=False)
        self.motion_gen.clear_world_cache()
        self.motion_gen.reset(reset_seed=False)
        self.ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            None,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=128,
            self_collision_check=True,
            self_collision_opt=False,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
            regularization=True,
        )
        self.ik_solver = IKSolver(self.ik_config)
        self.ordered_js_names = []
        self.dof_len = 7
        self.raw_js_names = []

    def update(self, ignore_list: list[str] = []) -> None:
        robot_name = self.robot_prim_path.split("/")[-1]
        obstacles = self.usd_helper.get_obstacles_from_stage(
            ignore_substring=[robot_name, "Camera"] + ignore_list,
            reference_prim_path=self.robot_prim_path,
        ).get_collision_check_world()
        self.motion_gen.update_world(obstacles)

    def plan(
        self,
        ee_translation_goal: np.ndarray,
        ee_orientation_goal: np.ndarray,
        sim_js: SimJointState,
        dof_names: list | None = None,
        grasp: bool = False,
    ) -> list[np.ndarray] | None:
        if os.environ.get("GENMANIP_VERBOSE") == "1":
            print("goal pos:", ee_translation_goal)
            print(
                "goal quat:",
                ee_orientation_goal,
                "norm=",
                np.linalg.norm(ee_orientation_goal),
            )
            print(
                "js len:",
                len(sim_js.positions),
                "names len:",
                len(self.ordered_js_names),
            )
            print(
                "finite:",
                np.isfinite(sim_js.positions).all(),
                np.isfinite(ee_translation_goal).all(),
                np.isfinite(ee_orientation_goal).all(),
            )

        if len(self.raw_js_names) == 0:
            self.raw_js_names = self.ordered_js_names
        ik_goal = Pose(
            position=self.tensor_args.to_device(ee_translation_goal),
            quaternion=self.tensor_args.to_device(ee_orientation_goal),
        )
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=self.ordered_js_names if dof_names is None else dof_names,
        )
        cu_js = cu_js.get_ordered_joint_state(self.ordered_js_names)
        plan_config = self.plan_config.clone()
        if grasp:
            plan_config.pose_cost_metric = self.pose_metric
        else:
            plan_config.pose_cost_metric = None
        result = self.motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)

        if os.environ.get("GENMANIP_VERBOSE") == "1":
            for k in ["status", "message", "error_code", "reason", "valid", "feasible"]:
                if hasattr(result, k):
                    print(k, getattr(result, k))

        if result.success is not None and result.success.item():
            cmd_plan = result.get_interpolated_plan()
            cmd_plan = cmd_plan.get_ordered_joint_state(self.raw_js_names)
            position_list = []
            for idx in range(len(cmd_plan.position)):
                joint_positions = cmd_plan.position[idx].cpu().numpy()  # type: ignore
                position_list.append(joint_positions[: self.dof_len])
            return position_list
        else:
            return None

    def ik_single(
        self, target_pose: np.ndarray, cur_joint_positions: np.ndarray
    ) -> np.ndarray | None:
        retract_config = self.tensor_args.to_device(cur_joint_positions.reshape(1, -1))
        seed_config = self.tensor_args.to_device(cur_joint_positions.reshape(1, 1, -1))
        pose = Pose(
            self.tensor_args.to_device(target_pose[:3]),
            self.tensor_args.to_device(target_pose[3:]),
        )
        ik_result = self.ik_solver.solve_single(
            pose, retract_config=retract_config, seed_config=seed_config
        )
        if not ik_result.success.item():
            return None
        return ik_result.js_solution.position.cpu().numpy().squeeze()  # type: ignore

    def fk_single(self, joint_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        joint_positions_tensor = torch.from_numpy(
            joint_positions.astype(np.float32)
        ).to(self.tensor_args.device)
        result = self.ik_solver.fk(joint_positions_tensor.unsqueeze(0))
        position = result.ee_position.cpu().numpy().squeeze()
        orientation = result.ee_quaternion.cpu().numpy().squeeze()
        return position, orientation
