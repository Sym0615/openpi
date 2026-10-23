"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import numpy as np

from genmanip.core.robot.base import BaseEmbodiment, ManipRobotConfig
from genmanip.core.robot.utils import RobotFactory


@RobotFactory.register("manip/singlearm")
class SingleArmEmbodiment(BaseEmbodiment):
    def __init__(self, config: ManipRobotConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

    def _plan_pose(  # type: ignore[override]
        self,
        goal_pose: tuple[np.ndarray, np.ndarray],
        joint_position: list[float],
        dof_name: list[str],
        grasp: bool = False,
        arm: str = "default",
    ) -> list[np.ndarray] | None:
        if self.planner is None:
            return None
        return self.planner.plan(
            goal_pose[0], goal_pose[1], joint_position, grasp=grasp
        )
