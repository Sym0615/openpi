"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import ctypes
from datetime import datetime
import logging
import os
import json
from pathlib import Path
import pickle
from typing import Any

import lmdb
import numpy as np
import shutil

from omni.isaac.core.articulations import Articulation  # type: ignore
from omni.isaac.core.prims import XFormPrim  # type: ignore
from omni.isaac.sensor import Camera  # type: ignore

from genmanip.core.robot.base import BaseEmbodiment
from genmanip.utils.usd_utils.joint_utils import create_joint_xform_list
from genmanip.utils.usd_utils.camera_utils import get_intrinsic_matrix
from genmanip.utils.standalone.transform_utils import pose_to_transform


DEFAULT_RGB_SCALE_FACTOR = 256000.0


def collect_task_data(
    object_list: dict[str, XFormPrim],
    articulation_list: dict[str, Articulation],
    articulation_part_list: dict[str, XFormPrim],
    robot_list: list[BaseEmbodiment],
    camera_data: dict[str, Camera],
    task_data: dict,
    usd_path_list: dict[str, str],
    preload_object_meta_info: dict[str, dict],
) -> dict:
    task_data["initial_scene_graph"] = None
    task_data["initial_layout"] = {}

    for key in object_list:
        if not object_list[key].prim.IsActive():
            continue
        task_data["initial_layout"][key] = {}
        task_data["initial_layout"][key]["type"] = "object"
        task_data["initial_layout"][key]["position"] = object_list[
            key
        ].get_world_pose()[0]
        task_data["initial_layout"][key]["orientation"] = object_list[
            key
        ].get_world_pose()[1]
        task_data["initial_layout"][key]["scale"] = object_list[key].get_local_scale()
        if key in usd_path_list:
            task_data["initial_layout"][key]["path"] = usd_path_list[key]
        else:
            task_data["initial_layout"][key]["path"] = ""
        if key in preload_object_meta_info:
            task_data["initial_layout"][key]["add_colliders"] = (
                preload_object_meta_info[key]["add_colliders"]
            )
            task_data["initial_layout"][key]["add_rigid_body"] = (
                preload_object_meta_info[key]["add_rigid_body"]
            )
        else:
            task_data["initial_layout"][key]["add_colliders"] = True
            task_data["initial_layout"][key]["add_rigid_body"] = True
        task_data["initial_layout"][key]["prim_path"] = object_list[key].prim_path
        if key in articulation_part_list:
            task_data["initial_layout"][key]["is_articulation_part"] = True
        else:
            task_data["initial_layout"][key]["is_articulation_part"] = False

    for key, articulation in articulation_list.items():
        task_data["initial_layout"][key] = {}
        task_data["initial_layout"][key]["type"] = "articulation"
        task_data["initial_layout"][key]["position"] = articulation.get_world_pose()[0]
        task_data["initial_layout"][key]["orientation"] = articulation.get_world_pose()[
            1
        ]
        task_data["initial_layout"][key]["scale"] = articulation.get_local_scale()
        task_data["initial_layout"][key][
            "joint_positions"
        ] = articulation.get_joint_positions()
        task_data["initial_layout"][key]["prim_path"] = articulation.prim_path

    for embodiment in robot_list:
        task_data["initial_layout"][embodiment.robot.name] = {}
        task_data["initial_layout"][embodiment.robot.name]["type"] = "robot"
        task_data["initial_layout"][embodiment.robot.name][
            "position"
        ] = embodiment.robot.get_world_pose()[0]
        task_data["initial_layout"][embodiment.robot.name][
            "orientation"
        ] = embodiment.robot.get_world_pose()[1]
        task_data["initial_layout"][embodiment.robot.name][
            "joint_positions"
        ] = embodiment.robot.get_joint_positions()

    task_data["camera_data"] = camera_data
    return task_data


def clip_float_values(
    float_array: np.ndarray, min_value: float, max_value: float
) -> np.ndarray:
    return np.clip(float_array, min_value, max_value)


def encode_seg_mask(seg_mask: np.ndarray) -> np.ndarray:
    assert seg_mask.shape[0] % 2 == 0
    assert seg_mask.shape[1] % 2 == 0
    h, w = seg_mask.shape[0], seg_mask.shape[1]
    reshaped = seg_mask.reshape(h // 2, 2, w // 2, 2)
    reshaped = reshaped.transpose(0, 2, 1, 3).reshape(h // 2, w // 2, 4)
    encoded = np.zeros((h // 2, w // 2, 4), dtype=np.uint8)
    encoded[..., 0] = reshaped[..., 0]
    encoded[..., 1] = reshaped[..., 1]
    encoded[..., 2] = reshaped[..., 2]
    encoded[..., 3] = reshaped[..., 3]
    return encoded


def decode_seg_mask(seg_mask: np.ndarray) -> np.ndarray:
    assert seg_mask.shape[2] == 4
    h, w = seg_mask.shape[0], seg_mask.shape[1]
    decoded = np.zeros((h * 2, w * 2), dtype=np.uint8)
    decoded[0::2, 0::2] = seg_mask[..., 0]
    decoded[0::2, 1::2] = seg_mask[..., 1]
    decoded[1::2, 0::2] = seg_mask[..., 2]
    decoded[1::2, 1::2] = seg_mask[..., 3]
    return decoded


def float_array_to_rgb_image(
    float_array: np.ndarray,
    scale_factor: float = DEFAULT_RGB_SCALE_FACTOR,
    drop_blue: bool = False,
    min_inttype: int = 0,
    max_inttype: int = 2**24 - 1,
) -> np.ndarray:
    scaled_array = np.round(float_array * scale_factor)
    scaled_array = clip_float_values(scaled_array, min_inttype, max_inttype)
    int_array = scaled_array.astype(np.uint32)
    r = int_array // (256 * 256)
    g = (int_array // 256) % 256
    b = int_array % 256 if not drop_blue else np.zeros_like(int_array)
    rgb_array = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return rgb_array


def image_to_float_array(
    image: np.ndarray, scale_factor: float | None = None
) -> np.ndarray:
    image_array = np.asarray(image)
    if scale_factor is None:
        scale_factor = DEFAULT_RGB_SCALE_FACTOR
    float_array = np.dot(image_array, [65536, 256, 1])
    return float_array / scale_factor


class PlanningRecorder:
    def __init__(
        self,
        cameras: dict[str, Camera],
        embodiment: BaseEmbodiment,
        object_list: dict[str, XFormPrim],
        instruction: str,
        log_dir: str = "logs",
        max_size: int = 1,  # Size in TB
        name: str = "",
        task_data: dict = {},
        logger: logging.Logger | None = None,
        articulation_list: dict[str, Articulation] = {},
    ) -> None:
        if name == "":
            self.name = datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")
        else:
            self.name = name
        self.instruction = instruction
        self.data_dir = Path(f"{log_dir}/{self.name}")
        self.logger = logger
        if self.logger is not None:
            self.logger.info(f"Generated data will be saved to: {self.data_dir}")
        self.max_size = int(max_size * 1024**4)
        self.env = None
        self.json_data_logger = {}
        self.scalar_data_logger = {}
        self.color_image_logger = {}
        self.depth_image_logger = {}
        self.obj_mask_logger = {}

        self.cameras = cameras
        if "camera1" in self.cameras:
            self.cameras.pop("camera1")
        self.embodiment = embodiment

        self.task_data = task_data
        self.log_num_steps = 0
        self.frame_status = {}
        self.current_action = []

        self.joint_xform_list = create_joint_xform_list(embodiment.robot)
        self.object_list = object_list
        self.articulation_list = articulation_list
        self.articulation_part_list = {
            key: create_joint_xform_list(articulation)
            for key, articulation in articulation_list.items()
        }
        self.load_static_info()

        # Initialize last action
        self.last_action = embodiment.robot.get_joint_positions()[
            embodiment.default_dof_indices
        ]
        self.last_grasp = -1.0 if embodiment.gripper_dof_num == 1 else [-1.0, -1.0]
        self.last_arm = "default"
        self.last_base_motion = np.array([0.0, 0.0, 0.0])

    def load_static_info(self) -> None:
        articulation_mapping = {}
        for key in self.articulation_part_list:
            for part_name, part_xform in self.articulation_part_list[key].items():
                articulation_mapping[str(key + "/" + part_name)] = {
                    "articulation_id": key,
                    "part_name": part_name,
                    "prim_path": str(part_xform.prim_path),
                }
        self.add_json_data(f"observation/articulation_mapping", articulation_mapping)

    def load_dynamic_info(
        self,
        action: np.ndarray | None,
        gripper_action: float | list[float] | None,
        arm: str | None = "default",
        name: str | list[str] | None = None,
        base_motion: np.ndarray = np.array([0.0, 0.0, 0.0]),
    ) -> None:
        if action is None:
            action = self.last_action
        else:
            self.last_action = action
        if gripper_action is None:
            gripper_action = self.last_grasp  # type: ignore
            if isinstance(gripper_action, list) and len(gripper_action) == 2:
                if arm == "left":
                    gripper_action = gripper_action[0]
                elif arm == "right":
                    gripper_action = gripper_action[1]
        if arm is None:
            arm = self.last_arm
        else:
            self.last_arm = arm
        action = np.array(action)
        if arm == "default":
            gripper_close = gripper_action
        elif arm == "left":
            gripper_close = [gripper_action, -1]
        elif arm == "right":
            gripper_close = [-1, gripper_action]
        else:
            raise ValueError(f"Invalid arm: {arm}")
        self.last_grasp = gripper_action
        if isinstance(name, list):
            self.add_action_name_frame(name)
        elif isinstance(name, str):
            self.add_name_frame(name)
        self.add_scalar_data(
            f"arm_action", action[self.embodiment.default_arm_dof_indices]
        )
        self.add_scalar_data(
            f"gripper_action", action[self.embodiment.default_gripper_dof_indices]
        )
        self.add_scalar_data(f"gripper_close", gripper_close)
        self.add_scalar_data(f"base_motion", base_motion)
        self.add_scalar_data(f"name", name)
        for key in self.object_list:
            self.add_scalar_data(
                f"observation/obj_pose/{key}/position",
                self.object_list[key].get_world_pose()[0],
            )
            self.add_scalar_data(
                f"observation/obj_pose/{key}/orientation",
                self.object_list[key].get_world_pose()[1],
            )
            self.add_scalar_data(
                f"observation/obj_pose/{key}/scale",
                self.object_list[key].get_local_scale(),
            )
        for key in self.articulation_list:
            self.add_scalar_data(
                f"observation/articulation_pose/{key}/position",
                self.articulation_list[key].get_world_pose()[0],
            )
            self.add_scalar_data(
                f"observation/articulation_pose/{key}/orientation",
                self.articulation_list[key].get_world_pose()[1],
            )
            self.add_scalar_data(
                f"observation/articulation_pose/{key}/scale",
                self.articulation_list[key].get_local_scale(),
            )
            self.add_scalar_data(
                f"observation/articulation_pose/{key}/qpos",
                self.articulation_list[key].get_joint_positions(),
            )
        for key in self.articulation_part_list:
            for part_name, part_xform in self.articulation_part_list[key].items():
                self.add_scalar_data(
                    f"observation/articulation_part_pose/{key}/{part_name}/position",
                    part_xform.get_world_pose()[0],
                )
                self.add_scalar_data(
                    f"observation/articulation_part_pose/{key}/{part_name}/orientation",
                    part_xform.get_world_pose()[1],
                )
                self.add_scalar_data(
                    f"observation/articulation_part_pose/{key}/{part_name}/scale",
                    part_xform.get_local_scale(),
                )
        self.add_scalar_data(
            f"observation/robot/qpos", self.embodiment.robot.get_joint_positions()
        )
        joint_world_pose = {}
        for joint_name, joint_xform in self.joint_xform_list.items():
            joint_world_pose[joint_name] = joint_xform.get_world_pose()
        self.add_scalar_data(f"observation/robot/joint_world_pose", joint_world_pose)
        self.add_scalar_data(
            f"observation/robot/qvel", self.embodiment.robot.get_joint_velocities()
        )
        self.add_scalar_data(
            f"observation/robot/robot2env_pose",
            pose_to_transform(self.embodiment.robot.get_world_pose()),
        )
        self.log_num_steps += 1

    def add_action_name_frame(self, name_list: list[str]) -> None:
        for name in self.current_action:
            if name not in name_list:
                self.frame_status[name + "/end"] = self.log_num_steps
        for name in name_list:
            if name not in self.frame_status:
                self.frame_status[name + "/start"] = self.log_num_steps
        self.current_action = name_list

    def add_name_frame(self, name: str) -> None:
        if name not in self.frame_status:
            self.frame_status[name] = self.log_num_steps

    def add_scalar_data(self, key: str, value: Any) -> None:
        if key not in self.scalar_data_logger:
            self.scalar_data_logger[key] = []
        self.scalar_data_logger[key].append(value)

    def add_color_image(self, key: str, value: np.ndarray) -> None:
        if key not in self.color_image_logger:
            self.color_image_logger[key] = []
        self.color_image_logger[key].append(value)

    def add_obj_mask(self, key: str, value: np.ndarray) -> None:
        if key not in self.obj_mask_logger:
            self.obj_mask_logger[key] = []
        self.obj_mask_logger[key].append(value)

    def add_depth_image(self, key: str, value: np.ndarray) -> None:
        if key not in self.depth_image_logger:
            self.depth_image_logger[key] = []
        self.depth_image_logger[key].append(value)

    def add_json_data(self, key: str, data: Any) -> None:
        self.json_data_logger[key] = data

    def save(
        self, task_name: str | None = None, config_path: str | None = None
    ) -> bool:
        # if os.path.exists(self.data_dir):
        #     return False
        self.data_dir.mkdir(parents=True, exist_ok=True)
        log_dir_lmdb = self.data_dir / "lmdb"
        meta_info = {}
        meta_info["max_size"] = self.max_size
        meta_info["num_steps"] = self.log_num_steps
        meta_info["language_instruction"] = self.task_data["instruction"]
        meta_info["task_data"] = self.task_data
        meta_info["task_data"]["frame_status"] = self.frame_status
        meta_info["keys"] = {}
        if task_name is not None:
            meta_info["task_name"] = task_name
        else:
            meta_info["task_name"] = ""
        meta_info["episode_name"] = self.name
        if (
            "arm_action" not in self.scalar_data_logger
            or len(self.scalar_data_logger["arm_action"]) == 0
        ):
            pickle.dump(
                meta_info, open(os.path.join(self.data_dir, "meta_info.pkl"), "wb")
            )
            if config_path is not None:
                shutil.copy(config_path, self.data_dir / "config.yaml")
            self.set_permissions(str(self.data_dir))
            return True
        self.env = lmdb.open(str(log_dir_lmdb), map_size=self.max_size)
        txn = self.env.begin(write=True)
        with open(log_dir_lmdb / "info.json", "w") as f:
            json.dump(self.json_data_logger, f)
        txn.put("json_data".encode("utf-8"), pickle.dumps(self.json_data_logger))
        meta_info["keys"]["json_data"] = ["json_data".encode("utf-8")]
        meta_info["keys"]["scalar_data"] = []
        for key, value in self.scalar_data_logger.items():
            txn.put(key.encode("utf-8"), pickle.dumps(value))
            meta_info["keys"]["scalar_data"].append(key.encode("utf-8"))
        txn.commit()
        self.env.close()
        pickle.dump(meta_info, open(os.path.join(self.data_dir, "meta_info.pkl"), "wb"))
        if config_path is not None:
            shutil.copy(config_path, self.data_dir / "config.yaml")
        self.set_permissions(str(self.data_dir))
        if self.logger is not None:
            self.logger.info(
                f"Save data with length {self.log_num_steps} to {self.data_dir}"
            )
        return True

    def release(self, trim: bool = True) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except (OSError, RuntimeError, lmdb.Error) as exc:
                if self.logger is not None:
                    self.logger.warning("Failed to close LMDB environment: %s", exc)
            self.env = None

        self.json_data_logger.clear()
        self.scalar_data_logger.clear()
        self.color_image_logger.clear()
        self.depth_image_logger.clear()
        self.obj_mask_logger.clear()
        self.frame_status.clear()

        self.cameras = {}
        self.tcp_xform_list = []
        self.joint_xform_list = {}

        import gc

        gc.collect()

        if not trim:
            return
        try:
            libc = ctypes.CDLL("libc.so.6")
            malloc_trim = getattr(libc, "malloc_trim", None)
            if malloc_trim is not None:
                malloc_trim(0)
        except (AttributeError, OSError, RuntimeError, TypeError) as exc:
            if self.logger is not None:
                self.logger.warning("Failed to call malloc_trim: %s", exc)

    def set_permissions(self, path: str) -> None:
        os.chmod(path, 0o777)
        for root, dirs, files in os.walk(path):
            for dir_name in dirs:
                os.chmod(os.path.join(root, dir_name), 0o777)
            for file_name in files:
                os.chmod(os.path.join(root, file_name), 0o777)
