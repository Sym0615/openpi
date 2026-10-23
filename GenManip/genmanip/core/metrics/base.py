"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from typing import Any

from genmanip.core.metrics.utils import MetricFactory

if TYPE_CHECKING:
    from genmanip.core.scene.scene import Scene


@MetricFactory.register("base")
class BaseMetric(ABC):
    def __init__(
        self,
        skip_steps: int = 1,
        succ_cnts: int = 0,
        sub_goal_setting: dict[str, Any] = {},
        never_reset: bool = False,
        **kwargs,
    ) -> None:
        self.skip_steps = skip_steps
        self.succ_cnts = succ_cnts
        self.goal_setting = sub_goal_setting
        self.weight = self.goal_setting.get("metric_weight", 1.0)
        self.never_reset = never_reset
        self.kwargs = kwargs

        self._step_cnt = 0
        self._succ_cnt = 0

        self.succ_flag = False

    @property
    def status(self) -> bool:
        return self.succ_flag

    @abstractmethod
    def check_status(self, scene: Scene) -> bool:
        raise NotImplementedError("check_status must be implemented in subclass")

    def update(self, scene: Scene):
        if self.never_reset and self.succ_flag:
            return

        if self._step_cnt % self.skip_steps == 0:
            _status = self.check_status(scene)

            if _status:
                self._succ_cnt += self.skip_steps
            elif not self.never_reset:
                self._succ_cnt = 0

        self._step_cnt += 1
        self.succ_flag = self._succ_cnt > self.succ_cnts

    def get_info(self):
        return str(self.goal_setting)
