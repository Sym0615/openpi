"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from typing import Tuple, Any
from pydantic import BaseModel, Field, field_validator
import numpy as np

from genmanip.core.metrics.base import BaseMetric
from genmanip.core.metrics.utils import MetricFactory
from genmanip.extensions.metrics.default.sr_based_genmanip_relationship import (
    SRBasedGenmanipRelationship,
)
from genmanip.extensions.metrics.default.match_local_world_orientation import (
    MatchLocalWorldOrientation,
)


class PegInHoleConfig(BaseModel):
    peg_uid: str = Field(..., description="Prim name or path of the peg object")
    hole_uid: str = Field(
        ..., description="Prim name or path of the target hole object"
    )
    base_uid: str = Field(..., description="Prim name or path of the base hole object")
    insert_ratio_range: Tuple[float, float] = Field(
        ...,
        description="Valid insertion ratio range: (min_ratio, max_ratio), 0 <= min <= max <= 1",
    )
    xy_tolerance: float = Field(
        ..., description="XY position tolerance for peg-hole alignment"
    )
    condition_type: str = Field(..., description="condition type")

    @field_validator("insert_ratio_range")
    @classmethod
    def validate_insert_ratio_range(cls, v):
        min_r, max_r = v
        if not (0.0 <= min_r <= max_r <= 1.0):
            raise ValueError("insert_ratio_range must satisfy 0 <= min <= max <= 1")
        return v

    @field_validator("xy_tolerance")
    @classmethod
    def validate_xy_tolerance(cls, v):
        if v < 0:
            raise ValueError("xy_tolerance must be non-negative")
        return v


@MetricFactory.register("manip/exciting_benchmark/peg_in_hole")
class PegInHole(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.setting = PegInHoleConfig(**sub_goal_setting)
        self.check_peg_vertical = MatchLocalWorldOrientation(
            sub_goal_setting=sub_goal_setting
        )

    def check_status(self, scene):
        if self.setting.condition_type == "peg_away_from_base":
            return not self.is_peg_in_hole(scene, self.setting.base_uid)
        else:
            return self.is_peg_in_hole(scene, self.setting.hole_uid)

    def is_peg_in_hole(self, scene, hole_uid):
        target_uids = [self.setting.peg_uid, hole_uid]
        pclist = SRBasedGenmanipRelationship.get_target_pc_list(scene, target_uids)

        peg_vertices = pclist[self.setting.peg_uid]
        hole_vertices = pclist[hole_uid]

        peg_min = peg_vertices.min(axis=0)
        peg_max = peg_vertices.max(axis=0)
        hole_max = hole_vertices.max(axis=0)

        z_min_peg = peg_min[2]
        z_max_peg = peg_max[2]
        z_entry_hole = hole_max[2]

        # Rule 1: The bottom of peg is lower than the entrance,
        # and the top of peg is higher than the entrance.
        if not (z_min_peg < z_entry_hole < z_max_peg):
            return False

        # Rule 2: The ratio of peg in the hole meets the requirements.
        peg_height = z_max_peg - z_min_peg
        inserted_depth = z_entry_hole - z_min_peg
        insert_ratio = inserted_depth / peg_height

        if not (
            self.setting.insert_ratio_range[0]
            <= insert_ratio
            <= self.setting.insert_ratio_range[1]
        ):
            return False

        # Rule 3: The distance between the center of peg and hole in
        # the XY projection meets the requirements.
        peg_xy_center = peg_vertices[:, :2].mean(axis=0)
        hole_xy_center = hole_vertices[:, :2].mean(axis=0)
        xy_distance = np.linalg.norm(peg_xy_center - hole_xy_center)

        if xy_distance > self.setting.xy_tolerance:
            return False

        # Rule 4: The peg should be kept vertical.
        if not self.check_peg_vertical.check_status(scene=scene):
            return False

        return True
