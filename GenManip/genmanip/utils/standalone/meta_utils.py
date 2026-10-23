"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import random


def safe_projection(key, dict1, dict2):
    if key not in dict1:
        return
    if isinstance(dict1[key], list):
        for i in range(len(dict1[key])):
            if dict1[key][i] is not None and dict1[key][i] in dict2:
                dict1[key][i] = dict2[dict1[key][i]]
    else:
        if dict1[key] is not None and dict1[key] in dict2:
            dict1[key] = dict2[dict1[key]]


def random_choice_from_object_or_list(object_or_list):
    if isinstance(object_or_list, list):
        return random.choice(object_or_list)
    else:
        return object_or_list


def any_projection(
    any_dict: dict,
    meta_to_fine_projection: dict,
    **kwargs,
) -> dict:
    # scene graph projection
    if (
        "obj1_uid" in any_dict and "obj2_uid" in any_dict and "position" in any_dict
    ) or ("obj1_uid" in any_dict and "status" in any_dict):
        return scene_graph_projection(any_dict, meta_to_fine_projection)
    elif "type" in any_dict and any_dict["type"] == "custom_motion":
        return custom_motion_projection(any_dict, meta_to_fine_projection)
    elif "type" in any_dict and any_dict["type"] == "move_to":
        return move_to_projection(any_dict, meta_to_fine_projection)
    elif any_dict.get("type", None) == "manip/default/sr_based_genmanip_axis_align":
        return axis_align_projection(any_dict, meta_to_fine_projection)
    elif any_dict.get("type", None) == "manip/default/sr_based_genmanip_range":
        return range_projection(any_dict, meta_to_fine_projection)
    else:
        return any_dict


def axis_align_projection(
    axis_align: dict,
    meta_to_fine_projection: dict,
) -> dict:
    safe_projection("obj1_uid", axis_align, meta_to_fine_projection)
    safe_projection("obj2_uid", axis_align, meta_to_fine_projection)
    return axis_align


def range_projection(
    range: dict,
    meta_to_fine_projection: dict,
) -> dict:
    safe_projection("obj1_uid", range, meta_to_fine_projection)
    safe_projection("x_rel_object_uid", range, meta_to_fine_projection)
    safe_projection("y_rel_object_uid", range, meta_to_fine_projection)
    safe_projection("z_rel_object_uid", range, meta_to_fine_projection)
    return range


def scene_graph_projection(
    scene_graph: dict,
    meta_to_fine_projection: dict,
) -> dict:
    safe_projection("obj1_uid", scene_graph, meta_to_fine_projection)
    safe_projection("obj2_uid", scene_graph, meta_to_fine_projection)
    safe_projection("ignored_uid", scene_graph, meta_to_fine_projection)
    return scene_graph


def move_to_projection(
    move_to: dict,
    meta_to_fine_projection: dict,
) -> dict:
    safe_projection("rel_object_uid", move_to, meta_to_fine_projection)
    return move_to


def custom_motion_projection(
    custom_motion: dict,
    meta_to_fine_projection: dict,
) -> dict:
    for info_list in custom_motion["motion_list"].values():
        for info in info_list:
            safe_projection("rel_object_uid", info, meta_to_fine_projection)
            safe_projection("debug_rel_object_uid", info, meta_to_fine_projection)
    return custom_motion


def any_random_choice_process(
    any_dict: dict,
    **kwargs,
) -> dict:
    if "obj1_uid" in any_dict and "obj2_uid" in any_dict and "position" in any_dict:
        return scene_graph_random_choice_process(any_dict, **kwargs)
    else:
        return any_dict


def scene_graph_random_choice_process(
    scene_graph: dict,
    is_benchmark: bool = False,
) -> dict:
    assert not is_benchmark or (
        (
            (
                isinstance(scene_graph["obj1_uid"], list)
                and len(scene_graph["obj1_uid"]) == 1
            )
            or (not isinstance(scene_graph["obj1_uid"], list))
        )
        and (
            (
                isinstance(scene_graph["obj2_uid"], list)
                and len(scene_graph["obj2_uid"]) == 1
            )
            or (not isinstance(scene_graph["obj2_uid"], list))
        )
    ), "obj1_uid and obj2_uid must be string or a list with only one element in benchmark mode"
    scene_graph["obj1_uid"] = random_choice_from_object_or_list(scene_graph["obj1_uid"])
    scene_graph["obj2_uid"] = random_choice_from_object_or_list(scene_graph["obj2_uid"])
    scene_graph["position"] = random_choice_from_object_or_list(scene_graph["position"])
    return scene_graph
