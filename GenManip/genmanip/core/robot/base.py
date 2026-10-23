"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from abc import abstractmethod

from mplib.planner import Planner as MplibPlanner
import numpy as np
from pydantic import BaseModel, Field
from scipy.spatial.transform import Rotation as R
from typing import List

from omni.isaac.core.robots.robot import Robot  # type: ignore

from genmanip.core.scene.scene_config import RobotConfig
from genmanip.core.robot.utils import RobotFactory
from genmanip.utils.planner.curobo.base import CuroboPlanner
from genmanip.utils.usd_utils.joint_utils import (
    get_all_joints_simple,
    get_all_body_from_joint,
)


class ManipRobotConfig(BaseModel):
    """Pydantic model for robot configuration. All fields are required."""

    embodiment_name: str = Field(..., description="Name of the robot embodiment")
    arm_name: str = Field(..., description="Name of the arm")
    gripper_name: str = Field(..., description="Name of the gripper")
    arm_dof_num: int = Field(
        ..., description="Number of degrees of freedom for the arm"
    )
    gripper_dof_num: int = Field(
        ..., description="Number of degrees of freedom for the gripper"
    )
    gripper_open: List[float] = Field(
        ..., description="Values representing the gripper fully open"
    )
    gripper_close: List[float] = Field(
        ..., description="Values representing the gripper fully closed"
    )
    default_arm_dof_indices: List[int] = Field(
        ..., description="Default DOF indices for the arm"
    )
    default_gripper_dof_indices: List[int] = Field(
        ..., description="Default DOF indices for the gripper"
    )


@RobotFactory.register("base")
class BaseEmbodiment:
    def __init__(self, config: ManipRobotConfig, *args, **kwargs) -> None:
        self.embodiment_name = config.embodiment_name
        self.arm_name = config.arm_name
        self.gripper_name = config.gripper_name
        self.arm_dof_num = config.arm_dof_num
        self.gripper_dof_num = config.gripper_dof_num
        self.gripper_open = config.gripper_open
        self.gripper_close = config.gripper_close
        self.default_arm_dof_indices = config.default_arm_dof_indices
        self.default_gripper_dof_indices = config.default_gripper_dof_indices

        self.robot = self.create_robot(
            scene_uid=kwargs["scene_uid"],
            default_config=kwargs["default_config"],
            robot_config=kwargs["robot_config"],
        )
        self.robot_view = self.robot._articulation_view

        planner_list = self.set_planner(kwargs["default_config"]["current_dir"])
        assert len(planner_list) in (1, 2), "Planner list must have 1 or 2 elements"
        if len(planner_list) == 1:
            self.planner = planner_list[0]
        else:
            self.left_planner, self.right_planner = planner_list

    @abstractmethod
    def create_robot(
        self, scene_uid: str, default_config: dict, robot_config: RobotConfig
    ) -> Robot:
        raise NotImplementedError("create_robot must be implemented in subclass")

    @abstractmethod
    def set_planner(self, current_dir: str) -> List[CuroboPlanner]:
        raise NotImplementedError("set_planner must be implemented in subclass")

    def _initialize(self, default_joint_positions: list[float] | None = None) -> None:
        self.robot.initialize()
        if default_joint_positions is not None:
            self.robot.set_joint_positions(default_joint_positions)
        else:
            if self.robot.get_joints_default_state() is not None:
                self.robot.set_joint_positions(
                    self.robot.get_joints_default_state().positions
                )
        self.default_dof_indices = [i for i in range(len(self.robot.dof_names))]
        self.joint_dict = {}
        get_all_joints_simple(self.robot.prim, self.joint_dict)
        self.link_dict = {}
        get_all_body_from_joint(self.joint_dict, self.link_dict)
        self.joint_positions_length = len(self.robot.get_joint_positions())

    def _post_initialize(self) -> None:
        self.robot.set_joints_default_state(
            positions=self.robot.get_joints_state().positions,
            velocities=self.robot.get_joints_state().velocities,
        )

    def initialize(self, default_joint_positions: list[float] | None = None) -> None:
        self._initialize(default_joint_positions)
        self._post_initialize()

    def set_joint_positions(
        self, joint_positions: list[float], joint_indices: list[int] | None = None
    ) -> None:
        if joint_indices is None:
            joint_indices = self.default_dof_indices
        self.robot.set_joint_positions(joint_positions, joint_indices=joint_indices)

    def get_joint_positions(
        self, joint_indices: list[int] | None = None
    ) -> list[float]:
        if joint_indices is None:
            joint_indices = self.default_dof_indices
        return self.robot.get_joint_positions(joint_indices=joint_indices)

    @abstractmethod
    def _plan_pose(
        self,
        goal_pose: tuple[np.ndarray, np.ndarray],
        joint_position: list[float],
        dof_name: list[str],
        grasp: bool = False,
        arm: str = "default",
    ) -> list[float] | None:
        raise NotImplementedError("_plan_pose must be implemented in subclass")

    def _transform_goal_pose(
        self, goal_pose: tuple[np.ndarray, np.ndarray], arm: str = "default"
    ) -> tuple[np.ndarray, np.ndarray]:
        robot_p, robot_q = self.robot.get_world_pose()
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

    def plan_pose(
        self,
        goal_pose: tuple[np.ndarray, np.ndarray],
        joint_position: list[float],
        grasp: bool = False,
        arm: str = "default",
    ) -> list[float] | None:
        if "planner" in self.__dict__:
            assert not isinstance(
                self.planner, MplibPlanner
            ), "mplib planner is not supported anymore"
        elif "left_planner" in self.__dict__ and "right_planner" in self.__dict__:
            assert not isinstance(self.left_planner, MplibPlanner) and not isinstance(
                self.right_planner, MplibPlanner
            ), "mplib planner is not supported anymore"
        else:
            raise ValueError("Planner is not initialized")
        dof_name = self.robot.dof_names
        goal_pose = self._transform_goal_pose(goal_pose, arm)
        return self._plan_pose(
            goal_pose, joint_position, dof_name, grasp=grasp, arm=arm
        )

    def convert_curobo_result_to_action(
        self, result: list[float], grasp: bool, arm: str = "default"
    ) -> np.ndarray:
        return np.concatenate(
            [
                result[: self.arm_dof_num],
                self.gripper_open if not grasp else self.gripper_close,
            ]
        )

    def convert_action_to_joint_state(
        self, action: list[float], arm: str = "default"
    ) -> list[float]:
        joint_positions = self.robot.get_joint_positions()
        joint_positions[self.default_dof_indices] = action
        return joint_positions

    def _fk_single(self, joint_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.planner is None:
            raise ValueError("Planner is not initialized")
        return self.planner.fk_single(joint_positions[: self.arm_dof_num])

    def fk_single(self, joint_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if joint_positions is None:
            print(f"Error: joint_positions is None in fk_single")
            return None
        if len(joint_positions) == self.joint_positions_length:
            return self._fk_single(joint_positions[self.default_dof_indices])
        elif (
            len(joint_positions) == self.arm_dof_num
            or len(joint_positions) == self.arm_dof_num + self.gripper_dof_num
            or len(joint_positions) == len(self.default_dof_indices)
        ):
            return self._fk_single(joint_positions)
        else:
            raise ValueError(f"Invalid joint positions length: {len(joint_positions)}")

    def _ik_single(
        self, pose: np.ndarray, cur_joint_positions: np.ndarray
    ) -> np.ndarray | None:
        if self.planner is None:
            raise ValueError("Planner is not initialized")
        return self.planner.ik_single(pose, cur_joint_positions)

    def ik_single(
        self,
        pose: np.ndarray,
        cur_joint_positions: np.ndarray,
        in_world_frame: bool = False,
    ) -> np.ndarray | None:
        if in_world_frame:
            p = pose[:3]
            q = pose[3:]
            transformed_pose = self._transform_goal_pose((p, q))
            pose = np.concatenate([transformed_pose[0], transformed_pose[1]])
        return self._ik_single(pose, cur_joint_positions)

    def reference_arm_type(self, target: np.ndarray) -> str:
        return "default"

    def reset(self) -> None:
        pass
