"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import os
from typing import List

from omni.isaac.core.robots.robot import Robot  # type: ignore
from omni.isaac.core.utils.prims import create_prim, get_prim_at_path  # type: ignore

from genmanip.core.robot.base import ManipRobotConfig
from genmanip.core.robot.singlearm_manip import SingleArmEmbodiment
from genmanip.core.robot.utils import RobotFactory
from genmanip.utils.planner.curobo.base import CuroboPlanner
from genmanip.utils.standalone.file_utils import load_yaml
from genmanip.utils.usd_utils import (
    get_world_pose_by_prim_path,
    set_drive_damping_and_stiffness,
    set_drive_max_force,
)
from genmanip.core.scene.scene_config import RobotConfig


@RobotFactory.register("manip/franka/robotiq")
class FrankaRobotiqEmbodiment(SingleArmEmbodiment):
    def __init__(self, *args, **kwargs) -> None:
        config = ManipRobotConfig(
            embodiment_name="franka",
            arm_name="franka",
            gripper_name="robotiq",
            arm_dof_num=7,
            gripper_dof_num=6,
            gripper_open=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            gripper_close=[0.7853, 0.7853, -0.7853, -0.7853, -0.7853, -0.7853],
            default_arm_dof_indices=[0, 1, 2, 3, 4, 5, 6],
            default_gripper_dof_indices=[7, 8, 9, 10, 11, 12],
        )
        super().__init__(config, *args, **kwargs)

    def create_robot(
        self, scene_uid: str, default_config: dict, robot_config: RobotConfig
    ) -> Robot:
        # Get the position and orientation of the franka robot
        position, orientation = get_world_pose_by_prim_path(
            f"/World/{scene_uid}/franka"
        )
        # Deactivate the franka robot
        prim = get_prim_at_path(f"/World/{scene_uid}/franka")
        if prim.IsActive() and prim.IsValid():
            prim.SetActive(False)
        # Create the franka_robotiq robot
        prim = create_prim(
            prim_path=f"/World/{scene_uid}/robotiq",
            prim_type="Xform",
            usd_path=os.path.join(
                default_config["ASSETS_DIR"], "robot_usds/robotiq/robot_mimic.usd"
            ),
        )
        # Create the franka_robotiq robot object
        robot = Robot(
            prim_path=f"/World/{scene_uid}/robotiq",
            name="franka_robotiq",
        )
        # Set default parameters for the franka_robotiq robot
        robot.set_solver_position_iteration_count(124)
        robot.set_stabilization_threshold(0.005)
        robot.set_solver_velocity_iteration_count(4)
        robot.set_world_pose(position, orientation)
        return robot

    def _initialize(self, default_joint_positions=None):
        super()._initialize(default_joint_positions)
        joint_list = [
            "base_link/finger_joint",
            "base_link/right_outer_knuckle_joint",
            "base_link/left_inner_knuckle_joint",
            "base_link/right_inner_knuckle_joint",
            "right_outer_finger/right_inner_finger_joint",
            "left_outer_finger/left_inner_finger_joint",
        ]
        for joint in joint_list:
            set_drive_damping_and_stiffness(
                self.robot.prim_path + f"/Robotiq_2F_85/{joint}",
                damping=0.01,
                stiffness=0.1,
            )
            set_drive_max_force(
                self.robot.prim_path + f"/Robotiq_2F_85/{joint}",
                10000.0,
            )
        joint_list = [
            # "base_link/right_outer_knuckle_joint",
            "base_link/left_inner_knuckle_joint",
            "base_link/right_inner_knuckle_joint",
            "right_outer_finger/right_inner_finger_joint",
            "left_outer_finger/left_inner_finger_joint",
        ]
        for joint in joint_list:
            set_drive_max_force(
                self.robot.prim_path + f"/Robotiq_2F_85/{joint}",
                0.0,
            )
        self.robot.set_joint_positions(
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.57079, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )

    def set_planner(self, current_dir: str) -> List[CuroboPlanner]:
        franka_cfg = load_yaml(
            os.path.join(current_dir, "configs/robots/curobo/franka.yml")
        )
        planner = CuroboPlanner(franka_cfg, self.robot.prim_path)
        planner.ordered_js_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        planner.dof_len = 7

        return [planner]
