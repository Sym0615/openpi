"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from copy import deepcopy
from filelock import SoftFileLock, Timeout
import pickle
import numpy as np
import os
import random

from genmanip.core.scene.scene_config import SceneConfig
from genmanip.utils.standalone.file_utils import load_dict_from_pkl, make_dir
from genmanip.utils.standalone.meta_utils import any_random_choice_process


def check_planning_finished(task_name, num_episode, default_config):
    lock_file = os.path.join(
        os.path.join(
            default_config["DEMONSTRATION_DIR"],
            task_name,
            "trajectory",
            "log_soft.lock",
        )
    )
    try:
        with SoftFileLock(lock_file, timeout=600.0):
            log_pkl_path = os.path.join(
                os.path.join(
                    default_config["DEMONSTRATION_DIR"],
                    task_name,
                    "trajectory",
                    "log.pkl",
                )
            )
            if os.path.exists(log_pkl_path):
                try:
                    log = load_dict_from_pkl(log_pkl_path)
                except (OSError, EOFError, ValueError, TypeError, pickle.PickleError) as exc:
                    print(f"Warning: failed to read planning log {log_pkl_path}: {exc}")
                    return False
                if "success" in log and log["success"] >= num_episode:
                    return True
            return False
    except Timeout:
        raise RuntimeError(
            f"Filelock timeout, try to delete the lock file by python standalone_tools/cleanup_lockfiles.py"
        )


def check_evalgen_finished(task_name, num_test, default_config):
    lock_file = os.path.join(
        os.path.join(default_config["TASKS_DIR"], task_name, "log_soft.lock")
    )
    try:
        with SoftFileLock(lock_file, timeout=600.0):
            log_pkl_path = os.path.join(
                os.path.join(default_config["TASKS_DIR"], task_name, "log.pkl")
            )
            if os.path.exists(log_pkl_path):
                try:
                    log = load_dict_from_pkl(log_pkl_path)
                except (OSError, EOFError, ValueError, TypeError, pickle.PickleError) as exc:
                    print(f"Warning: failed to read evalgen log {log_pkl_path}: {exc}")
                    return False
                if "success" in log and log["success"] >= num_test:
                    return True
            return False
    except Timeout:
        raise RuntimeError(
            f"Filelock timeout, try to delete the lock file by python standalone_tools/cleanup_lockfiles.py"
        )


def check_eval_finished(scene_config: SceneConfig, default_config: dict) -> int:
    if scene_config.num_test is None:
        raise ValueError("Num test is not set")
    lock_file = os.path.join(
        default_config["EVAL_RESULT_DIR"], scene_config.task_name, "eval_soft.lock"
    )
    try:
        with SoftFileLock(lock_file, timeout=600.0):
            task_dir = os.path.join(
                default_config["EVAL_RESULT_DIR"],
                scene_config.task_name,
            )
            if os.path.exists(task_dir):
                evaluation_num = len(os.listdir(task_dir)) - 1
                if evaluation_num >= scene_config.num_test:
                    if os.path.exists(lock_file):
                        os.remove(lock_file)
                    return -1
                else:
                    make_dir(os.path.join(task_dir, str(evaluation_num).zfill(3)))
                    return evaluation_num
            return 0
    except Timeout:
        raise RuntimeError(
            f"Filelock timeout, try to delete the lock file by python standalone_tools/cleanup_lockfiles.py"
        )


def get_random_position_candidate() -> list[str]:
    return ["top", "near", "left", "right", "front", "back"]


def is_triple_layer_list(goal: list) -> bool:
    if (
        isinstance(goal, list)
        and len(goal) > 0
        and isinstance(goal[0], list)
        and len(goal[0]) > 0
        and isinstance(goal[0][0], list)
        and len(goal[0][0]) > 0
    ):
        return True
    return False


def corse_process_task_data(scene_config: SceneConfig) -> dict:
    task_data = {}
    if scene_config.generation_config.randomization_hack_flag and is_triple_layer_list(
        scene_config.generation_config.goal
    ):
        scene_config.generation_config.goal = random.choice(
            scene_config.generation_config.goal
        )
    task_data["goal"] = deepcopy(scene_config.generation_config.goal)

    # process goal config
    def _process_goal(goal: list | dict, is_benchmark: bool = False):
        if isinstance(goal, list):
            return [_process_goal(subgoal, is_benchmark) for subgoal in goal]
        elif isinstance(goal, dict):
            return any_random_choice_process(goal, is_benchmark=is_benchmark)

    task_data["goal"] = _process_goal(
        task_data["goal"], is_benchmark=scene_config.mode == "benchmark"
    )

    if scene_config.generation_config.action_path.actions is not None:
        for subaction in scene_config.generation_config.action_path.actions:
            subaction = any_random_choice_process(
                subaction, is_benchmark=scene_config.mode == "benchmark"
            )
    return task_data


def concat_instruction(
    task_data: dict,
    sequence: int | list | tuple | None = None,
    record_arm_info: bool = False,
) -> str:
    instruction = ""
    goals = task_data["goal"][0]
    if sequence is not None:
        if isinstance(sequence, int):
            goals = [goals[sequence]]
        elif isinstance(sequence, (list, tuple)):
            goals = [goals[i] for i in sequence]
        else:
            raise TypeError("sequence must be int or list/tuple")
    for i, subgoal in enumerate(goals):
        if "obj1_uid" in subgoal and "obj2_uid" in subgoal:
            obj1_caption = (
                task_data["object_infos"][subgoal["obj1_uid"]]["caption"]
                .lower()
                .replace(".", "")
            )
            obj2_caption = (
                task_data["object_infos"][subgoal["obj2_uid"]]["caption"]
                .lower()
                .replace(".", "")
            )
            position = subgoal["position"]
            if i == 0:
                instruction += (
                    f"Move the {obj1_caption} to the {position} of the {obj2_caption}"
                )
            else:
                instruction += f", and move the {obj1_caption} to the {position} of the {obj2_caption}"
        elif "obj1_uid" in subgoal:
            obj1_caption = (
                task_data["object_infos"][subgoal["obj1_uid"]]["caption"]
                .lower()
                .replace(".", "")
            )
            if i == 0:
                instruction += f"Open the {obj1_caption}"
            else:
                instruction += f", open the {obj1_caption}"
        if record_arm_info and subgoal.get("arm", None) is not None:
            instruction += f" with {subgoal['arm']} arm"
    instruction += "."
    return instruction


def rewrite_instruction(task_data: dict, demogen_config: SceneConfig) -> None:
    if demogen_config.domain_randomization.rewrite_instruction:
        sequence = demogen_config.domain_randomization.rewrite_sequece
        task_data["instruction"] = concat_instruction(
            task_data,
            sequence,
            record_arm_info=demogen_config.domain_randomization.record_arm_info,
        )
    else:
        task_data["instruction"] = demogen_config.instruction


def get_action_list(task_data: dict, demogen_config: SceneConfig) -> dict:
    if demogen_config.generation_config.action_path.mode == "auto":
        goal_data = deepcopy(task_data["goal"])
        task_data["task_path"] = random.choice(goal_data)
    elif demogen_config.generation_config.action_path.mode == "manual":
        task_data["task_path"] = demogen_config.generation_config.action_path.actions
    if (
        demogen_config.generation_config.is_shuffle
        and task_data["task_path"] is not None
    ):
        random.shuffle(task_data["task_path"])
    return task_data


def refine_task_data(task_data: dict, demogen_config: SceneConfig) -> dict:
    rewrite_instruction(task_data, demogen_config)
    task_data = get_action_list(task_data, demogen_config)
    return task_data


def adjust_arm_gripper_action_by_embodiment(
    arm_action: np.ndarray,
    gripper_action: np.ndarray,
    embodiment_name: str,
) -> np.ndarray:
    if embodiment_name == "aloha_split":
        return np.concatenate(
            [arm_action[:6], gripper_action[:2], arm_action[6:], gripper_action[2:]]
        )
    elif embodiment_name == "lift2":
        return np.concatenate(
            [arm_action[:6], gripper_action[:2], arm_action[6:], gripper_action[2:]]
        )
    else:
        return np.concatenate([arm_action, gripper_action])
