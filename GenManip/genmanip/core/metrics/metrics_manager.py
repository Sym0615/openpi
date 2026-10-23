"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from concurrent.futures import ThreadPoolExecutor
import time

from genmanip.core.metrics.utils import MetricFactory


class MetricsManager:
    def __init__(self, goal_setting: list[list[dict]]):
        self.goal_setting = self.ensure_list_nesting(goal_setting, 3)
        self.metric_score = self.init_score_list(self.goal_setting)

        self.long_stride_idx = -1
        self.cur_union_metric = self.get_next_metric()

    @staticmethod
    def ensure_list_nesting(x: list, levels: int) -> list:
        depth = 0
        _x = x
        while isinstance(_x, list) and depth < levels:
            depth += 1
            _x = _x[0] if _x else []

        for _ in range(levels - depth):
            x = [x]
        return x

    @staticmethod
    def init_score_list(x: list | dict) -> list | float:
        if not isinstance(x, list):
            return 0.0
        return [MetricsManager.init_score_list(e) for e in x]

    def print_metrics_info(self, metrics: list):
        for metric_list in metrics:
            for metric in metric_list:
                print(metric.get_info())
            print("-" * 100)

    def get_next_metric(self) -> list | None:
        if not self.goal_setting:
            return None

        next_metric_setting = self.goal_setting.pop(0)
        self.long_stride_idx += 1

        metrics = [
            [MetricFactory.build(cfg["type"], **cfg) for cfg in inner]
            for inner in next_metric_setting
        ]
        self.print_metrics_info(metrics)
        return metrics

    def step(self, scene) -> float | None:
        # OPT: per-frame cache that metrics can share (transformed point
        # clouds, convex hulls). Cleared at the top of every step so results
        # can never go stale across ticks.
        scene._frame_pc_cache = {}
        scene._frame_hull_cache = {}
        scene._frame_tree_cache = {}
        if self.cur_union_metric is None:
            return self.calc_overall_score()
        if isinstance(self.metric_score, float):
            raise ValueError("Status flag is a float, not a list")
        union_metrics = self.metric_score[self.long_stride_idx]

        for union_idx, collection_metrics in enumerate(self.cur_union_metric):
            for collection_idx, metric in enumerate(collection_metrics):
                metric.update(scene)
                union_metrics[union_idx][collection_idx] = metric.weight * (
                    1 if metric.status else 0
                )

        if any(
            [
                all(metric.status for metric in collection_metrics)
                for collection_metrics in self.cur_union_metric
            ]
        ):
            self.cur_union_metric = self.get_next_metric()

        return self.calc_overall_score()

    # def step(self, scene) -> float | None:
    #     if self.cur_union_metric is None:
    #         return self.calc_overall_score()
    #     if isinstance(self.metric_score, bool):
    #         raise ValueError("Status flag is a boolean, not a list")
    #     union_metrics = self.metric_score[self.long_stride_idx]

    #     tasks = []
    #     with ThreadPoolExecutor() as executor:
    #         for union_idx, collection_metrics in enumerate(self.cur_union_metric):
    #             for collection_idx, metric in enumerate(collection_metrics):

    #                 future = executor.submit(metric.update, scene)
    #                 tasks.append((future, union_idx, collection_idx, metric))

    #         for future, union_idx, collection_idx, metric in tasks:
    #             future.result()
    #             union_metrics[union_idx][collection_idx] = metric.status

    #     if all(
    #         union_metrics[i][j]
    #         for i in range(len(union_metrics))
    #         for j in range(len(union_metrics[i]))
    #     ):
    #         self.cur_union_metric = self.get_next_metric()

    #     return self.calc_overall_score()

    def calc_overall_score(self) -> float:
        def mean(xs):
            return sum(xs) / len(xs) if xs else 0.0

        outer_values = []
        if isinstance(self.metric_score, float):
            raise ValueError("Status flag is a float, not a list")
        for union_metric_status in self.metric_score:
            middle_values = []
            for collection_metric_status in union_metric_status:
                # Compute average for the innermost list
                leaf_avg = round(mean(collection_metric_status), 5)
                middle_values.append(leaf_avg)

            # Second level: take the maximum of inner averages
            max_middle = max(middle_values) if middle_values else 0.0
            outer_values.append(max_middle)

        # Top level: average the second-level maxima
        score = mean(outer_values)

        return score
