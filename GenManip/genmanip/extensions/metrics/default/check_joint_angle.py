import math
from typing import Tuple, Any
from pydantic import BaseModel, Field, field_validator

from genmanip.core.metrics.base import BaseMetric
from genmanip.core.metrics.utils import MetricFactory


class CheckJointAngleConfig(BaseModel):
    articulation_obj_uid: str = Field(..., description="uid of the articulation root")
    joint_name: str = Field(..., description="DOF name of the articulation joint")
    angle_deg_range: Tuple[float, float] = Field(
        ..., description="Allowed joint angle range in degrees: (min_angle, max_angle)"
    )

    @field_validator("angle_deg_range")
    @classmethod
    def validate_angle_range(cls, v):
        if v[0] > v[1]:
            raise ValueError("min angle must be less than or equal to max angle")
        return v


@MetricFactory.register("manip/default/check_joint_angle")
class CheckJointAngle(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.setting = CheckJointAngleConfig(**sub_goal_setting)

        self.min_angle_rad = math.radians(self.setting.angle_deg_range[0])
        self.max_angle_rad = math.radians(self.setting.angle_deg_range[1])

        self._joint_dof_index = None

    def check_status(self, scene):
        _articulation = scene.articulation_list[self.setting.articulation_obj_uid]
        if self._joint_dof_index is None:
            if self.setting.joint_name not in _articulation.dof_names:
                raise RuntimeError(
                    f"Joint '{self.setting.joint_name}' not found in articulation, available joints: {_articulation.dof_names}"
                )

            self._joint_dof_index = _articulation.get_dof_index(self.setting.joint_name)

        angle_rad = _articulation.get_joint_positions(self._joint_dof_index)[0]

        return self.min_angle_rad <= angle_rad <= self.max_angle_rad
