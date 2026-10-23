"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from typing import Any
from pydantic import BaseModel, Field

from genmanip.core.metrics.base import BaseMetric
from genmanip.core.metrics.utils import MetricFactory
from genmanip.extensions.metrics.default.match_local_world_orientation import (
    MatchLocalWorldOrientation,
)
from genmanip.extensions.metrics.default.sr_based_genmanip_relationship import (
    SRBasedGenmanipRelationship,
    calculate_distance_between_two_point_clouds,
)


class FrameAgainstPenHolderConfig(BaseModel):
    obj1_uid: str = Field(..., description="UID of the object to measure the distance")
    obj2_uid: str | None = Field(
        ..., description="UID of the object to measure the distance"
    )
    tolerance: float = Field(
        ..., description="distance tolerance between obj1_uid and obj2_uid"
    )
    condition_type: str = Field(..., description="condition type")


@MetricFactory.register("manip/exciting_benchmark/frame_against_pen_holder")
class FrameAgainstPenHolder(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.setting = FrameAgainstPenHolderConfig(**sub_goal_setting)
        if self.setting.condition_type == "check_holder_posture":
            self.check_frame_orien = MatchLocalWorldOrientation(
                sub_goal_setting=sub_goal_setting
            )
        self.init_height = None

    def check_status(self, scene):
        # Rule: The picture frame is pick up.
        if self.setting.condition_type == "is_pick_up":
            target_uids = [self.setting.obj1_uid]
            pclist = SRBasedGenmanipRelationship.get_target_pc_list(scene, target_uids)
            center_height = pclist[self.setting.obj1_uid][:, 2].mean(axis=0)
            if self.init_height is None:
                self.init_height = center_height

            if center_height > self.init_height + 0.001:
                return True

        # Rule: The picture frame is near the pen holder.
        elif self.setting.condition_type == "is_near":
            target_uids = [self.setting.obj1_uid, self.setting.obj2_uid]
            pclist = SRBasedGenmanipRelationship.get_target_pc_list(scene, target_uids)

            dist = calculate_distance_between_two_point_clouds(
                pclist[self.setting.obj1_uid], pclist[self.setting.obj2_uid]
            )
            if dist < self.setting.tolerance:
                return True

        # Rule: The back of the picture frame rests against the pen holder, and keeping it upright.
        else:
            if self.check_frame_orien.check_status(scene=scene):
                return True

        return False
