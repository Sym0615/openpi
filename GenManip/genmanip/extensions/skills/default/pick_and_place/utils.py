"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from genmanip.core.robot.base import BaseEmbodiment
from genmanip.utils.standalone.transform_utils import (
    adjust_orientation,
    adjust_translation_along_quaternion,
    rot_orientation_by_z_axis,
    rot_orientation_by_axis,
)


def prepare_motion_planning_payload(
    action_meta_info: dict,
    steps: int = 30,
    aug_distance: float = 0.0,
    pre_grasp_distance: float = 0.08,
    grasp_distance: float = 0,
    post_grasp_distance: float = 0.16,
    pre_place_distance: float = 0.16,
    place_distance: float = 0.02,
    post_place_distance: float = 0.08,
) -> list[dict]:
    action_list = []
    if pre_grasp_distance is not None:
        action_list.append(
            {
                "name": "pre_grasp",
                "translation": adjust_translation_along_quaternion(
                    action_meta_info["initial_grasp"]["position"],
                    action_meta_info["initial_grasp"]["orientation"],
                    pre_grasp_distance,
                    aug_distance=aug_distance,
                ),
                "orientation": action_meta_info["initial_grasp"]["orientation"],
                "steps": steps,
                "grasp": False,
            }
        )
    action_list.append(
        {
            "name": "grasp",
            "translation": adjust_translation_along_quaternion(
                action_meta_info["initial_grasp"]["position"],
                action_meta_info["initial_grasp"]["orientation"],
                grasp_distance,
            ),
            "orientation": action_meta_info["initial_grasp"]["orientation"],
            "steps": steps,
            "grasp": False,
        }
    )
    if post_grasp_distance is not None:
        action_list.append(
            {
                "name": "post_grasp",
                "translation": adjust_translation_along_quaternion(
                    action_meta_info["initial_grasp"]["position"],
                    action_meta_info["initial_grasp"]["orientation"],
                    post_grasp_distance,
                    aug_distance=aug_distance,
                ),
                "orientation": action_meta_info["initial_grasp"]["orientation"],
                "steps": steps,
                "grasp": True,
            }
        )
    if pre_place_distance is not None:
        action_list.append(
            {
                "name": "pre_place",
                "translation": adjust_translation_along_quaternion(
                    action_meta_info["finial_grasp"]["position"],
                    action_meta_info["finial_grasp"]["orientation"],
                    pre_place_distance,
                    aug_distance=aug_distance,
                ),
                "orientation": action_meta_info["finial_grasp"]["orientation"],
                "steps": steps,
                "grasp": True,
            }
        )
    action_list.append(
        {
            "name": "place",
            "translation": adjust_translation_along_quaternion(
                action_meta_info["finial_grasp"]["position"],
                action_meta_info["finial_grasp"]["orientation"],
                place_distance,
            ),
            "orientation": action_meta_info["finial_grasp"]["orientation"],
            "steps": steps,
            "grasp": True,
        }
    )
    if post_place_distance is not None:
        action_list.append(
            {
                "name": "post_place",
                "translation": adjust_translation_along_quaternion(
                    action_meta_info["finial_grasp"]["position"],
                    action_meta_info["finial_grasp"]["orientation"],
                    post_place_distance,
                    aug_distance=aug_distance,
                ),
                "orientation": action_meta_info["finial_grasp"]["orientation"],
                "steps": steps,
                "grasp": False,
            }
        )
    return action_list


def adjust_grasp_by_embodiment(
    grasp: dict,
    embodiment: BaseEmbodiment,
) -> dict:
    grasp["orientation"] = adjust_orientation(grasp["orientation"])
    if embodiment.embodiment_name == "franka":
        if embodiment.gripper_name == "panda_hand":
            grasp["translation"] = adjust_translation_along_quaternion(
                grasp["translation"],
                grasp["orientation"],
                0.08,
                aug_distance=0.0,
            )
        elif embodiment.gripper_name == "robotiq":
            # robotiq 的 grasp pose 需要绕 z 轴旋转 45 度
            grasp["orientation"] = rot_orientation_by_z_axis(grasp["orientation"], -45)
            grasp["translation"] = adjust_translation_along_quaternion(
                grasp["translation"],
                grasp["orientation"],
                0.15,
                aug_distance=0.0,
            )
    elif embodiment.embodiment_name == "aloha_split":
        if embodiment.gripper_name == "piper":
            grasp["orientation"] = rot_orientation_by_z_axis(grasp["orientation"], -90)
            grasp["translation"] = adjust_translation_along_quaternion(
                grasp["translation"],
                grasp["orientation"],
                0.11,
                aug_distance=0.0,
            )
    elif embodiment.embodiment_name == "lift2":
        if embodiment.gripper_name == "lift2":
            grasp["translation"] = adjust_translation_along_quaternion(
                grasp["translation"],
                grasp["orientation"],
                0.135,
                aug_distance=0.0,
            )
            grasp["orientation"] = rot_orientation_by_axis(
                grasp["orientation"], "y", 90
            )
            grasp["orientation"] = rot_orientation_by_axis(
                grasp["orientation"], "z", 180
            )
    return grasp
