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
from genmanip.extensions.metrics.default.match_local_world_orientation import (
    MatchLocalWorldOrientation,
    OrientationConstraintSpec,
)
from genmanip.extensions.metrics.default.sr_based_genmanip_relationship import (
    SRBasedGenmanipRelationship,
    calculate_distance_between_two_point_clouds,
)


class CollectCoffeeBeansConfig(BaseModel):
    jar_uid: str = Field(..., description="Prim name or path of the jar object")
    lid_uid: str = Field(..., description="Prim name or path of the lid object")
    xy_tolerance: float = Field(
        ..., description="XY position tolerance between lid and jar"
    )
    contact_tolerance: float = Field(
        ..., description="Contact tolerance between lid and jar"
    )
    constraint_spec: OrientationConstraintSpec = Field(
        ..., description="Orientation matching constraints"
    )


@MetricFactory.register("manip/exciting_benchmark/collect_coffee_beans")
class CollectCoffeeBeans(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.setting = CollectCoffeeBeansConfig(**sub_goal_setting)
        self.check_frame_orien = MatchLocalWorldOrientation(
            sub_goal_setting={
                "obj_uid": self.setting.lid_uid,
                "constraint_spec": self.setting.constraint_spec,
            }
        )

    def check_status(self, scene):
        target_uids = [
            self.setting.jar_uid,
            self.setting.lid_uid,
        ]
        pclist = SRBasedGenmanipRelationship.get_target_pc_list(scene, target_uids)
        jar_vertices = pclist[self.setting.jar_uid]
        lid_vertices = pclist[self.setting.lid_uid]

        # OPT: AABB pre-check — if world-frame AABBs are separated by more
        # than contact_tolerance the true NN distance is at least as large,
        # so Rule 1 must fail. Skips the expensive kNN on ~10k x ~10k points
        # for every step where the lid is still being carried.
        jar_min = jar_vertices.min(axis=0)
        jar_max = jar_vertices.max(axis=0)
        lid_min = lid_vertices.min(axis=0)
        lid_max = lid_vertices.max(axis=0)
        gap = np.maximum(
            0.0,
            np.maximum(jar_min, lid_min) - np.minimum(jar_max, lid_max),
        )
        aabb_dist = float(np.linalg.norm(gap))
        if aabb_dist >= self.setting.contact_tolerance:
            return False

        # Rule 1: Make sure the lid is in contact with the jar.
        dist = calculate_distance_between_two_point_clouds(jar_vertices, lid_vertices)
        if not dist < self.setting.contact_tolerance:
            return False

        # Rule 2: Confirm that the lid and the jar are on the same central axis.
        jar_xy_center = jar_vertices[:, :2].mean(axis=0)
        lid_xy_center = lid_vertices[:, :2].mean(axis=0)
        xy_distance = np.linalg.norm(jar_xy_center - lid_xy_center)

        if xy_distance > self.setting.xy_tolerance:
            return False

        # Rule 3: Make sure the lid is parallel to the ground.
        if not self.check_frame_orien.check_status(scene=scene):
            return False

        return True
