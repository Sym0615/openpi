from typing import List, Any
from pydantic import BaseModel, Field

import copy
import numpy as np

from genmanip.core.metrics.base import BaseMetric
from genmanip.core.metrics.utils import MetricFactory
from genmanip.extensions.metrics.default.sr_based_genmanip_relationship import (
    SRBasedGenmanipRelationship,
    check_subgoal_finished_rigid,
    check_subgoal_finished_articulation,
)


class CountMultiObjectRelationshipConfig(BaseModel):
    obj1_uid_list: List[str] = Field(
        ..., description="UIDs of the object to measure the relationship"
    )
    obj2_uid: str | None = Field(
        default=None, description="UID of the object to measure the relationship"
    )
    position: str | None = Field(
        default=None, description="Position of the object to measure the relationship"
    )
    min_num_threshold: int = Field(
        default=1,
        description="The minimum number of values ​​that satisfy the relationship between obj1_uid and obj2_uid.",
    )
    another_obj2_uid: str | None = Field(
        default=None,
        description="UID of the another object to measure the relationship",
    )
    status: List[List[float]] | None = Field(
        default=None, description="Status of the object to measure the relationship"
    )
    obj1_pc_num: int | None = Field(default=None, description="obj1 Point cloud count")
    obj2_pc_num: int | None = Field(default=None, description="obj1 Point cloud count")


@MetricFactory.register("manip/default/count_multi_object_relationship")
class CountMultiObjectRelationship(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.goal_setting = CountMultiObjectRelationshipConfig(**sub_goal_setting)
        self.obj1_uid_list_num = len(self.goal_setting.obj1_uid_list)

    def check_status(self, scene):
        articulation_list = scene.articulation_list

        target_uids = copy.deepcopy(self.goal_setting.obj1_uid_list)
        if self.goal_setting.obj2_uid:
            target_uids.append(self.goal_setting.obj2_uid)
        if self.goal_setting.another_obj2_uid:
            target_uids.append(self.goal_setting.another_obj2_uid)
        pclist = SRBasedGenmanipRelationship.get_target_pc_list(scene, target_uids)

        self.succ_num = 0

        if self.goal_setting.position:
            if self.goal_setting.another_obj2_uid is not None:
                pcd3 = pclist[self.goal_setting.another_obj2_uid]
            else:
                pcd3 = None

            if self.goal_setting.obj2_uid is None:
                raise ValueError("obj2_uid is required for position relationship")

            for idx in range(self.obj1_uid_list_num):
                obj1_uid = self.goal_setting.obj1_uid_list[idx]
                pcd1 = pclist[obj1_uid]
                pcd2 = pclist[self.goal_setting.obj2_uid]

                if self.goal_setting.obj1_pc_num is not None:
                    pcd1 = self._uniform_downsample(pcd1, self.goal_setting.obj1_pc_num)

                if self.goal_setting.obj2_pc_num is not None:
                    pcd2 = self._uniform_downsample(pcd2, self.goal_setting.obj2_pc_num)

                check_status_flag = check_subgoal_finished_rigid(
                    self.goal_setting.position,
                    pcd1,
                    pcd2,
                    pcd3,
                )
                self.succ_num += 1 if check_status_flag else 0

                # early stop
                _r = self._early_stop(idx)
                if _r is not None:
                    return _r
        elif self.goal_setting.status is not None:
            for idx in range(self.obj1_uid_list_num):
                obj1_uid = self.goal_setting.obj1_uid_list[idx]
                check_status_flag = check_subgoal_finished_articulation(
                    self.goal_setting.status, articulation_list[obj1_uid]
                )
                self.succ_num += 1 if check_status_flag else 0

                # early stop
                _r = self._early_stop(idx)
                if _r is not None:
                    return _r

        return False

    def _early_stop(self, idx):
        if (
            self.goal_setting.min_num_threshold - self.succ_num
            > self.obj1_uid_list_num - idx - 1
        ):
            return False
        elif self.goal_setting.min_num_threshold - self.succ_num == 0:
            return True
        else:
            return None

    def _uniform_downsample(self, arr: np.ndarray, num: int) -> np.ndarray:
        n = len(arr)
        if num >= n:
            return arr

        idx = np.linspace(0, n - 1, num, dtype=int)

        return arr[idx]
