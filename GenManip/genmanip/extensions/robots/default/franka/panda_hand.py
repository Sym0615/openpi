"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import os
from typing import List

from omni.isaac.core.robots.robot import Robot  # type: ignore
from omni.isaac.franka import Franka  # type: ignore

from genmanip.core.robot.base import ManipRobotConfig
from genmanip.core.robot.singlearm_manip import SingleArmEmbodiment
from genmanip.core.robot.utils import RobotFactory
from genmanip.utils.planner.curobo.base import CuroboPlanner
from genmanip.utils.standalone.file_utils import load_yaml
from genmanip.core.scene.scene_config import RobotConfig


@RobotFactory.register("manip/franka/panda_hand")
class FrankaNormalEmbodiment(SingleArmEmbodiment):
    def __init__(self, *args, **kwargs) -> None:
        config = ManipRobotConfig(
            embodiment_name="franka",
            arm_name="franka",
            gripper_name="panda_hand",
            arm_dof_num=7,
            gripper_dof_num=2,
            gripper_open=[0.04, 0.04],
            gripper_close=[0.0, 0.0],
            default_arm_dof_indices=[0, 1, 2, 3, 4, 5, 6],
            default_gripper_dof_indices=[7, 8],
        )
        super().__init__(config, *args, **kwargs)

        self.robot_view.set_max_joint_velocities([2.0] * 9)

    def create_robot(
        self, scene_uid: str, default_config: dict, robot_config: RobotConfig
    ) -> Robot:
        # Create the franka robot
        robot = Franka(
            prim_path=f"/World/{scene_uid}/franka",
        )

        # Set default parameters for the franka robot
        robot.set_solver_position_iteration_count(128)
        robot.set_enabled_self_collisions(True)
        robot.set_stabilization_threshold(0.005)
        robot.set_solver_velocity_iteration_count(16)
        return robot

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
