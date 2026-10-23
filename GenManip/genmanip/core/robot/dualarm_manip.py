"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from typing import List

import numpy as np
from pydantic import Field
from scipy.spatial.transform import Rotation as R

from curobo.types.state import JointState
from omni.isaac.core.prims import XFormPrim  # type: ignore

from genmanip.core.robot.base import BaseEmbodiment, ManipRobotConfig
from genmanip.core.robot.utils import RobotFactory
from genmanip.utils.usd_utils import (
    set_drive_damping_and_stiffness,
    set_drive_max_force,
)


class ManipDualArmRobotConfig(ManipRobotConfig):
    """Pydantic model for dual-arm robot configuration, extending ManipRobotConfig."""

    left_arm_dof_indices: List[int] = Field(
        ..., description="DOF indices for the left arm"
    )
    right_arm_dof_indices: List[int] = Field(
        ..., description="DOF indices for the right arm"
    )
    body_dof_indices: List[int] = Field(
        ..., description="DOF indices for the robot body"
    )
    base_dof_indices: List[int] = Field(
        ..., description="DOF indices for the robot base"
    )
    default_lift_joint_position: float = Field(
        ..., description="Default position for the lift joint"
    )
    default_lift_joint_path: str = Field(
        ..., description="Path name or identifier for the lift joint"
    )
    robot_base_left: str = Field(
        ..., description="The prim path of left arm base relative to the robot"
    )
    robot_base_right: str = Field(
        ..., description="The prim path of right arm base relative to the robot"
    )
    base_joint_path_x: str = Field(..., description="The prim path of the base joint x")
    base_joint_path_y: str = Field(..., description="The prim path of the base joint y")
    base_joint_path_rotate: str = Field(
        ..., description="The prim path of the base joint rotate"
    )


@RobotFactory.register("manip/dualarm")
class DualArmEmbodiment(BaseEmbodiment):
    def __init__(self, config: ManipDualArmRobotConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        self.left_arm_dof_indices = config.left_arm_dof_indices
        self.right_arm_dof_indices = config.right_arm_dof_indices
        self.body_dof_indices = config.body_dof_indices
        self.base_dof_indices = config.base_dof_indices
        self.default_lift_joint_position = config.default_lift_joint_position
        self.default_lift_joint_path = config.default_lift_joint_path
        self.robot_base_left = XFormPrim(self.robot.prim_path + config.robot_base_left)
        self.robot_base_right = XFormPrim(
            self.robot.prim_path + config.robot_base_right
        )
        self.base_joint_path_x = config.base_joint_path_x
        self.base_joint_path_y = config.base_joint_path_y
        self.base_joint_path_rotate = config.base_joint_path_rotate

    def _initialize(self, default_joint_positions: list[float] | None = None) -> None:
        super()._initialize(default_joint_positions)
        single_arm_joint_position = [0.0] * self.arm_dof_num + self.gripper_open
        self.robot.set_joint_positions(
            [self.default_lift_joint_position] + single_arm_joint_position * 2,
            joint_indices=self.body_dof_indices
            + self.left_arm_dof_indices
            + self.right_arm_dof_indices,
        )
        self.robot_view.set_joint_positions(
            [self.default_lift_joint_position] + single_arm_joint_position * 2,
            joint_indices=self.body_dof_indices
            + self.left_arm_dof_indices
            + self.right_arm_dof_indices,
        )
        self.default_dof_indices = (
            self.left_arm_dof_indices + self.right_arm_dof_indices
        )
        set_drive_damping_and_stiffness(
            self.robot.prim_path + self.default_lift_joint_path,
            damping=10000,
            stiffness=1000000,
        )
        set_drive_max_force(
            self.robot.prim_path + self.default_lift_joint_path,
            1000000,
        )
        self.initial_base_pose = self.robot.get_joint_positions()[
            self.base_dof_indices
        ].copy()
        self.target_base_pose = self.robot.get_joint_positions()[self.base_dof_indices]

    def reset(self) -> None:
        self.target_base_pose = self.initial_base_pose.copy()

    def delta_move_to(self, delta_x, delta_y, delta_yaw):
        delta_x = np.clip(delta_x, -0.015, 0.015)
        delta_y = np.clip(delta_y, -0.015, 0.015)
        delta_yaw = np.clip(delta_yaw, -1, 1)

        delta_yaw = np.deg2rad(delta_yaw)
        self.target_base_pose += np.array([delta_x, delta_y, delta_yaw])

        self.robot_view.set_joint_position_targets(
            self.target_base_pose,
            joint_indices=self.base_dof_indices,
        )

    def _fk_single(  # type: ignore[override]
        self, joint_positions: np.ndarray
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        if self.left_planner is None or self.right_planner is None:
            raise ValueError("Planner is not initialized")
        return (
            self.left_planner.fk_single(joint_positions[: self.arm_dof_num]),
            self.right_planner.fk_single(
                joint_positions[
                    self.arm_dof_num + self.gripper_dof_num : -self.gripper_dof_num
                ]
            ),
        )

    def _plan_pose(  # type: ignore[override]
        self,
        goal_pose: tuple[np.ndarray, np.ndarray],
        joint_position: JointState,
        dof_name: list[str],
        grasp: bool = False,
        arm: str = "default",
    ) -> list[np.ndarray] | None:
        if arm == "left":
            return self.left_planner.plan(
                goal_pose[0],
                goal_pose[1],
                joint_position,
                self.robot.dof_names,
                grasp=grasp,
            )
        elif arm == "right":
            return self.right_planner.plan(
                goal_pose[0],
                goal_pose[1],
                joint_position,
                self.robot.dof_names,
                grasp=grasp,
            )
        else:
            raise ValueError(f"Invalid arm: {arm}")

    def convert_curobo_result_to_action(
        self, result: list[float], grasp: bool, arm: str = "default"
    ) -> np.ndarray:
        joint_positions = self.robot.get_joint_positions()
        if arm == "left":
            if (
                len(result) == self.arm_dof_num
                or len(result) == self.arm_dof_num + self.gripper_dof_num
            ):
                return np.concatenate(
                    [
                        result[: self.arm_dof_num],
                        self.gripper_open if not grasp else self.gripper_close,
                        joint_positions[self.right_arm_dof_indices],
                    ]
                )
            elif len(result) == len(self.default_dof_indices):
                return np.concatenate(
                    [
                        result[: self.arm_dof_num],
                        self.gripper_open if not grasp else self.gripper_close,
                        result[self.arm_dof_num + self.gripper_dof_num :],
                    ]
                )
            else:
                raise ValueError(f"Invalid result length: {len(result)}")
        elif arm == "right":
            if (
                len(result) == self.arm_dof_num
                or len(result) == self.arm_dof_num + self.gripper_dof_num
            ):
                return np.concatenate(
                    [
                        joint_positions[self.left_arm_dof_indices],
                        result[: self.arm_dof_num],
                        self.gripper_open if not grasp else self.gripper_close,
                    ]
                )
            elif len(result) == len(self.default_dof_indices):
                return np.concatenate(
                    [
                        result[: self.arm_dof_num + self.gripper_dof_num],
                        result[
                            self.arm_dof_num
                            + self.gripper_dof_num : self.arm_dof_num
                            + self.gripper_dof_num
                            + self.arm_dof_num
                        ],
                        self.gripper_open if not grasp else self.gripper_close,
                    ]
                )
            else:
                raise ValueError(f"Invalid result length: {len(result)}")
        else:
            raise ValueError(f"Invalid arm: {arm}")

    def _transform_goal_pose(
        self, goal_pose: tuple[np.ndarray, np.ndarray], arm: str = "default"
    ) -> tuple[np.ndarray, np.ndarray]:
        if arm == "left":
            robot_p, robot_q = self.robot_base_left.get_world_pose()
        elif arm == "right":
            robot_p, robot_q = self.robot_base_right.get_world_pose()
        elif arm == "default":
            robot_p, robot_q = self.robot.get_world_pose()
        else:
            raise ValueError(f"Invalid arm: {arm}")
        goal_matrix = np.eye(4)
        goal_matrix[:3, :3] = R.from_quat(goal_pose[1][[1, 2, 3, 0]]).as_matrix()
        goal_matrix[:3, 3] = goal_pose[0]
        robot_matrix = np.eye(4)
        robot_matrix[:3, :3] = R.from_quat(robot_q[[1, 2, 3, 0]]).as_matrix()
        robot_matrix[:3, 3] = robot_p
        goal_pose_matrix = np.linalg.inv(robot_matrix) @ goal_matrix
        return (
            goal_pose_matrix[:3, 3],
            R.from_matrix(goal_pose_matrix[:3, :3]).as_quat()[[3, 0, 1, 2]],
        )

    def reference_arm_type(self, target: np.ndarray) -> str:
        target2left = np.linalg.norm(target - self.robot_base_left.get_world_pose()[0])
        target2right = np.linalg.norm(
            target - self.robot_base_right.get_world_pose()[0]
        )
        if target2left < target2right:
            return "left"
        else:
            return "right"
