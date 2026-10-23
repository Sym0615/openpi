"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""


def process_archived_robot_config(robot_config: dict) -> str:
    if "config" not in robot_config or "gripper_type" not in robot_config["config"]:
        return robot_config["type"]
    robot_type = None
    if robot_config["type"] == "franka":
        if robot_config["config"]["gripper_type"] == "panda_hand":
            robot_type = "manip/franka/panda_hand"
        elif robot_config["config"]["gripper_type"] == "robotiq":
            robot_type = "manip/franka/robotiq"
    elif robot_config["type"] == "aloha_split":
        robot_type = "manip/mobile_aloha/piper"
    elif robot_config["type"] == "lift2":
        robot_type = "manip/lift2/R5a"
    if robot_type is None:
        raise ValueError(f"Unsupported robot type: {robot_config['type']}")
    else:
        print(
            f"Robot type and robot gripper config is archived, please use the new key {robot_type} instead of {robot_config['type']} and {robot_config['config']['gripper_type']}"
        )
        return robot_type


def process_archived_config(config: dict) -> dict:
    if "robots" in config:
        for robot_config in config["robots"]:
            robot_config["type"] = process_archived_robot_config(robot_config)
            if "config" in robot_config and "gripper_type" in robot_config["config"]:
                robot_config["config"].pop("gripper_type")
    if "generation_config" in config:
        if isinstance(config["generation_config"]["articulation"], list):
            articulation_config = {}
            for articulation in config["generation_config"]["articulation"]:
                articulation_config[articulation["uid"]] = articulation
            config["generation_config"]["articulation"] = articulation_config
    return config
