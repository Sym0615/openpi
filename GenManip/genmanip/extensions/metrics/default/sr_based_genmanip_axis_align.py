"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from typing import Any
import numpy as np
import open3d as o3d
from pydantic import BaseModel, Field
from scipy.spatial.transform import Rotation as R

from genmanip.core.metrics.base import BaseMetric
from genmanip.core.metrics.utils import MetricFactory


class SRBasedGenmanipAxisAlignConfig(BaseModel):
    obj1_uid: str = Field(..., description="UID of the object to align")
    obj2_uid: str = Field(..., description="UID of the object to align")
    obj1_axis: str = Field(..., description="Name of the axis of the first object")
    obj2_axis: str = Field(..., description="Name of the axis of the second object")
    comparison: str = Field(..., description="Comparison operator")
    threshold_deg: float = Field(..., description="Threshold value")


@MetricFactory.register("manip/default/sr_based_genmanip_axis_align")
class SRBasedGenmanipAxisAlign(BaseMetric):
    def __init__(
        self,
        skip_steps: int = 1,
        succ_cnts: int = 0,
        sub_goal_setting: dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.goal_setting = SRBasedGenmanipAxisAlignConfig(**sub_goal_setting)

    def _get_axis_vector(self, axis_name: str) -> np.ndarray:
        """
        Parses axis name to a 3D vector.

        Args:
            axis_name (str): The name of the axis (e.g., 'x', '-y', '+z').

        Returns:
            np.ndarray: A 3D unit vector representing the axis.
        """
        axis_name = axis_name.lower().strip()
        sign = 1.0
        if axis_name.startswith("-"):
            sign = -1.0
            axis_name = axis_name[1:]
        elif axis_name.startswith("+"):
            axis_name = axis_name[1:]

        vec = np.zeros(3)
        if axis_name == "x":
            vec[0] = 1.0
        elif axis_name == "y":
            vec[1] = 1.0
        elif axis_name == "z":
            vec[2] = 1.0
        else:
            raise ValueError(f"Invalid axis name: {axis_name}")

        return vec * sign

    def check_status(self, scene) -> bool:
        """
        Check if the alignment condition is met.

        Args:
            scene: The simulation scene object.

        Returns:
            bool: True if the condition is met, False otherwise.
        """
        object_1 = scene.object_list[self.goal_setting.obj1_uid]
        object_2 = scene.object_list[self.goal_setting.obj2_uid]
        object_1_pose = object_1.get_world_pose()
        object_2_pose = object_2.get_world_pose()

        rot_1 = R.from_quat(object_1_pose[1][[1, 2, 3, 0]]).as_matrix()
        rot_2 = R.from_quat(object_2_pose[1][[1, 2, 3, 0]]).as_matrix()

        # Get axis configurations from goal setting
        obj1_axis_name = self.goal_setting.obj1_axis
        obj2_axis_name = self.goal_setting.obj2_axis

        # Get local axis vectors
        vec1_local = self._get_axis_vector(obj1_axis_name)
        vec2_local = self._get_axis_vector(obj2_axis_name)

        # Transform local axes to world frame
        vec1_world = rot_1 @ vec1_local
        vec2_world = rot_2 @ vec2_local

        # Calculate angle between vectors in degrees
        dot_product = np.dot(vec1_world, vec2_world)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(dot_product))

        # Check condition against threshold
        if self.goal_setting.comparison in ["<", "less"]:
            return angle_deg < self.goal_setting.threshold_deg
        elif self.goal_setting.comparison in ["<=", "less_equal"]:
            return angle_deg <= self.goal_setting.threshold_deg
        elif self.goal_setting.comparison in [">", "greater"]:
            return angle_deg > self.goal_setting.threshold_deg
        elif self.goal_setting.comparison in [">=", "greater_equal"]:
            return angle_deg >= self.goal_setting.threshold_deg
        else:
            raise ValueError(
                f"Invalid comparison operator: {self.goal_setting.comparison}"
            )

    def get_info(self):
        return f"align {self.goal_setting.obj1_uid} {self.goal_setting.obj1_axis} with {self.goal_setting.obj2_uid} {self.goal_setting.obj2_axis} {self.goal_setting.comparison} {self.goal_setting.threshold_deg} degrees"
