"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from collections import defaultdict, deque
import copy
import random

from genmanip.core.scene.scene_config import SceneConfig


def process_scene_graph(
    scene_config: SceneConfig, object_list_key: list[str]
) -> list[dict]:
    TABLE_UID = "00000000000000000000000000000000"
    if scene_config.layout_config.scene_graph is None:
        raise ValueError("Scene graph is not defined")
    scene_graph = copy.deepcopy(scene_config.layout_config.scene_graph)
    on_table_uids = collect_on_table_uids(scene_graph)
    for uid in on_table_uids:
        scene_graph.append(
            {
                "obj1_uid": uid,
                "obj2_uid": TABLE_UID,
                "position": "on",
            }
        )
    mentioned_uids = collect_metioned_uids(scene_graph)
    for uid in object_list_key:
        if uid not in mentioned_uids:
            scene_graph.append(
                {
                    "obj1_uid": uid,
                    "obj2_uid": TABLE_UID,
                    "position": "on",
                }
            )
    sorted_uids = topo_sort_by_on(scene_graph)
    sorted_scene_graph = sort_scene_graph_by_topo_sort(scene_graph, sorted_uids)
    return sorted_scene_graph


def topo_sort_by_on(scene_graph: list[dict]) -> list[str]:
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    all_nodes = set()
    for edge in scene_graph:
        u, v, pos = edge["obj1_uid"], edge["obj2_uid"], edge["position"]
        all_nodes.update([u, v])
        if pos == "on" or pos == "top":
            graph[v].append(u)
            in_degree[u] += 1
    for node in all_nodes:
        in_degree.setdefault(node, 0)
    queue = deque()
    for node, deg in in_degree.items():
        if deg == 0:
            queue.append(node)
    result = []
    while queue:
        current_layer = list(queue)
        random.shuffle(current_layer)
        result.extend(current_layer)
        queue.clear()
        for node in current_layer:
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
    return result


def collect_metioned_uids(scene_graph: list[dict]) -> list[str]:
    mentioned_uids = set()
    for edge in scene_graph:
        mentioned_uids.add(edge["obj1_uid"])
        mentioned_uids.add(edge["obj2_uid"])
    return list(mentioned_uids)


def collect_on_table_uids(scene_graph: list[dict]) -> list[str]:
    on_table_uids = set()
    for edge in scene_graph:
        on_table_uids.add(edge["obj1_uid"])
        on_table_uids.add(edge["obj2_uid"])
        if edge["position"] == "on" or edge["position"] == "top":
            on_table_uids.remove(edge["obj1_uid"])
    return list(on_table_uids)


def sort_scene_graph_by_topo_sort(
    scene_graph: list[dict[str, str]], sorted_uids: list[str]
) -> list[dict]:
    sorted_scene_graph = []
    seen_edges = set()
    seen_uids = set()

    def is_valid_edge(uid: str, edge: dict) -> bool:
        return (edge["obj1_uid"] == uid and edge["obj2_uid"] in seen_uids) or (
            edge["obj2_uid"] == uid and edge["obj1_uid"] in seen_uids
        )

    def transform_edge(edge: dict) -> dict:
        if edge["position"] not in ("on", "top"):
            return {
                "obj1_uid": edge["obj2_uid"],
                "obj2_uid": edge["obj1_uid"],
                "position": flip_position(edge["position"]),
            }
        return edge

    for uid in sorted_uids:
        current_layer = []
        for i, edge in enumerate(scene_graph):
            if i in seen_edges:
                continue
            if is_valid_edge(uid, edge):
                if edge["obj2_uid"] == uid and edge["position"] not in ("on", "top"):
                    current_layer.append(transform_edge(edge))
                else:
                    current_layer.append(edge)
                seen_edges.add(i)
        seen_uids.add(uid)
        sorted_scene_graph.append(current_layer)
    return sorted_scene_graph


def flip_position(position: str) -> str:
    flips = {
        "left": "right",
        "right": "left",
        "front": "back",
        "back": "front",
        "top": "bottom",
        "bottom": "top",
    }
    return flips.get(position, position)
