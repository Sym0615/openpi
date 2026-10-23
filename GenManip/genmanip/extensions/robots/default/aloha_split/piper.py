"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import os
import numpy as np
from typing import List

from omni.isaac.core.robots.robot import Robot  # type: ignore
from omni.isaac.core.utils.prims import create_prim, get_prim_at_path  # type: ignore

from genmanip.core.robot.dualarm_manip import DualArmEmbodiment, ManipDualArmRobotConfig
from genmanip.core.robot.utils import RobotFactory
from genmanip.utils.standalone.file_utils import load_yaml
from genmanip.utils.planner.curobo.base import CuroboPlanner
from genmanip.core.scene.scene_config import RobotConfig


@RobotFactory.register("manip/mobile_aloha/piper")
class AlohaSplitEmbodiment(DualArmEmbodiment):
    def __init__(self, *args, **kwargs) -> None:
        config = ManipDualArmRobotConfig(
            embodiment_name="aloha_split",
            arm_name="piper",
            gripper_name="piper",
            arm_dof_num=6,
            gripper_dof_num=2,
            gripper_open=[0.05, -0.05],
            gripper_close=[0.0, 0.0],
            default_arm_dof_indices=[0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13],
            default_gripper_dof_indices=[6, 7, 14, 15],
            left_arm_dof_indices=[12, 14, 16, 18, 20, 22, 24, 25],
            right_arm_dof_indices=[13, 15, 17, 19, 21, 23, 26, 27],
            body_dof_indices=[7],
            base_dof_indices=[0, 1, 2],
            default_lift_joint_position=-0.3,
            default_lift_joint_path="/split_aloha_mid_360_with_piper/split_aloha_mid_360_with_piper/box_link/lifting_joint",
            robot_base_left="/split_aloha_mid_360_with_piper/split_aloha_mid_360_with_piper/fl/arm_base",
            robot_base_right="/split_aloha_mid_360_with_piper/split_aloha_mid_360_with_piper/fr/arm_base",
            base_joint_path_x="/split_aloha_mid_360_with_piper/split_aloha_mid_360_with_piper/dummy_base_x/mobile_translate_x",
            base_joint_path_y="/split_aloha_mid_360_with_piper/split_aloha_mid_360_with_piper/dummy_base_y/mobile_translate_y",
            base_joint_path_rotate="/split_aloha_mid_360_with_piper/split_aloha_mid_360_with_piper/dummy_base_rotate/mobile_rotate",
        )
        super().__init__(config, *args, **kwargs)

        self.robot_view.set_max_joint_velocities([2.0] * 28)

    def create_robot(
        self, scene_uid: str, default_config: dict, robot_config: RobotConfig
    ) -> Robot:
        # Deactivate the franka robot
        prim = get_prim_at_path(f"/World/{scene_uid}/franka")
        if prim.IsValid() and prim.IsActive():
            prim.SetActive(False)

        if robot_config.position is not None:
            position = robot_config.position
        else:
            position = np.array([-0.65, 0.0, 0.3])
        if robot_config.orientation is not None:
            orientation = robot_config.orientation
        else:
            orientation = np.array([1.0, 0.0, 0.0, 0.0])
        # Create the aloha_split robot
        prim = create_prim(
            prim_path=f"/World/{scene_uid}/aloha_split",
            prim_type="Xform",
            usd_path=os.path.join(
                default_config["ASSETS_DIR"],
                "robot_usds/split_aloha_mid_360/robot.usd",
            ),
            position=position,
            orientation=orientation,
        )

        # Create the aloha_split robot object
        robot = Robot(
            prim_path=f"/World/{scene_uid}/aloha_split",
            name="aloha_split",
        )

        # Set default parameters for the aloha_split robot
        robot.set_solver_position_iteration_count(128)
        robot.set_stabilization_threshold(0.005)
        robot.set_solver_velocity_iteration_count(4)
        return robot

    def set_planner(self, current_dir: str) -> List[CuroboPlanner]:
        piper_cfg = load_yaml(
            os.path.join(
                current_dir, "saved/assets/miscs/curobo/piper100/piper100_left_arm.yml"
            )
        )
        piper_cfg["robot_cfg"]["kinematics"]["usd_path"] = piper_cfg["robot_cfg"][
            "kinematics"
        ]["usd_path"].replace("${ASSETS_DIR}", current_dir + "/saved/assets")
        piper_cfg["robot_cfg"]["kinematics"]["urdf_path"] = piper_cfg["robot_cfg"][
            "kinematics"
        ]["urdf_path"].replace("${ASSETS_DIR}", current_dir + "/saved/assets")

        left_planner = CuroboPlanner(piper_cfg, self.robot.prim_path)
        right_planner = CuroboPlanner(piper_cfg, self.robot.prim_path)

        left_planner.ordered_js_names = [
            "fl_joint1",
            "fl_joint2",
            "fl_joint3",
            "fl_joint4",
            "fl_joint5",
            "fl_joint6",
        ]
        right_planner.ordered_js_names = [
            "fr_joint1",
            "fr_joint2",
            "fr_joint3",
            "fr_joint4",
            "fr_joint5",
            "fr_joint6",
        ]
        raw_js_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
        left_planner.raw_js_names = raw_js_names
        right_planner.raw_js_names = raw_js_names

        return [left_planner, right_planner]
