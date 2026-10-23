"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from typing import Any
from pydantic import BaseModel, Field
import math

from genmanip.core.metrics.base import BaseMetric
from genmanip.core.metrics.utils import MetricFactory
from genmanip.extensions.metrics.default.sr_based_genmanip_relationship import (
    SRBasedGenmanipRelationship,
)


class InstallGearConfig(BaseModel):
    fix_gear_1_uid: str = Field(
        ..., description="Prim name or path of the fix_gear_1 object"
    )
    fix_gear_2_uid: str = Field(
        ..., description="Prim name or path of the fix_gear_2 object"
    )
    target_gear_uid: str = Field(
        ..., description="Prim name or path of the target_gear object"
    )
    xy_tolerance: float = Field(
        ..., description="XY position tolerance for gears alignment"
    )
    z_tolerance: float = Field(
        ..., description="Z position tolerance for gears alignment"
    )
    condition_type: str = Field(..., description="condition type")


@MetricFactory.register("manip/exciting_benchmark/install_gear")
class InstallGear(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.setting = InstallGearConfig(**sub_goal_setting)

    def check_status(self, scene):
        target_uids = [
            self.setting.fix_gear_1_uid,
            self.setting.fix_gear_2_uid,
            self.setting.target_gear_uid,
        ]
        pclist = SRBasedGenmanipRelationship.get_target_pc_list(scene, target_uids)

        fix_gear_1_vertices = pclist[self.setting.fix_gear_1_uid]
        fix_gear_2_vertices = pclist[self.setting.fix_gear_2_uid]
        target_gear_vertices = pclist[self.setting.target_gear_uid]

        if self.setting.condition_type == "is_coupling":
            # Rule: The gears are at the same height.
            fix_gear_1_z_center = fix_gear_1_vertices[:, 2].mean(axis=0)
            target_gear_z_center = target_gear_vertices[:, 2].mean(axis=0)

            if (
                abs(fix_gear_1_z_center - target_gear_z_center)
                > self.setting.z_tolerance
            ):
                return False

        # Rule: The target gear is in a reasonable position in the xy plane.
        xy_central_constraint = InstallGear.near_segment(
            target_gear_vertices[:, :2].mean(axis=0),
            fix_gear_1_vertices[:, :2].mean(axis=0),
            fix_gear_2_vertices[:, :2].mean(axis=0),
            self.setting.xy_tolerance,
        )

        return xy_central_constraint

    @staticmethod
    def near_segment(p, a, b, eps=0.01):
        ax, ay = a
        bx, by = b
        px, py = p

        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay

        ab2 = abx * abx + aby * aby
        if ab2 == 0:
            return math.hypot(apx, apy) < eps

        t = (apx * abx + apy * aby) / ab2
        if t < 0 or t > 1:
            return False

        dx = apx - t * abx
        dy = apy - t * aby
        return dx * dx + dy * dy < eps * eps
