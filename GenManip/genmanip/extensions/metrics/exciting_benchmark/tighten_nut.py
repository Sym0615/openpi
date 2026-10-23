"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from typing import Any
from pydantic import BaseModel, Field
import numpy as np

from genmanip.core.metrics.base import BaseMetric
from genmanip.core.metrics.utils import MetricFactory
from genmanip.extensions.metrics.default.sr_based_genmanip_relationship import (
    SRBasedGenmanipRelationship,
)


class TightenNutConfig(BaseModel):
    nut_uid: str = Field(..., description="Prim name or path of the nut object")
    bolt_uid: str = Field(..., description="Prim name or path of the bolt object")
    xy_tolerance: float = Field(
        ..., description="XY position tolerance for nut-bolt alignment"
    )
    condition_type: str = Field(..., description="condition type")


@MetricFactory.register("manip/exciting_benchmark/tighten_nut")
class TightenNut(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.setting = TightenNutConfig(**sub_goal_setting)

    def check_status(self, scene):
        target_uids = [self.setting.nut_uid, self.setting.bolt_uid]
        pclist = SRBasedGenmanipRelationship.get_target_pc_list(scene, target_uids)

        nut_vertices = pclist[self.setting.nut_uid]
        bolt_vertices = pclist[self.setting.bolt_uid]

        # Rule: The nut and bolt are close to each other in the xy plane.
        nut_xy_center = nut_vertices[:, :2].mean(axis=0)
        bolt_xy_center = bolt_vertices[:, :2].mean(axis=0)
        xy_distance = np.linalg.norm(nut_xy_center - bolt_xy_center)

        if xy_distance > self.setting.xy_tolerance:
            return False

        nut_min = nut_vertices.min(axis=0)
        nut_max = nut_vertices.max(axis=0)
        bolt_min = bolt_vertices.min(axis=0)
        bolt_max = bolt_vertices.max(axis=0)

        if self.setting.condition_type == "is_nut_pick_up":
            # Rule: Nut is heigher than bolt.
            if nut_max[2] > bolt_max[2]:
                return True

        else:
            # Rule: Nut at the center height of the bolt.
            if nut_min[2] > bolt_min[2] and nut_max[2] < bolt_max[2]:
                return True

        return False
