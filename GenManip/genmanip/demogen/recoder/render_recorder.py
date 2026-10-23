"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import ctypes
from datetime import datetime
import os
import json
import logging
from pathlib import Path
import pickle
import queue
import shutil
import threading
from typing import Any

import cv2
import lmdb
import numpy as np
import roboticstoolbox as rtb

from omni.isaac.sensor import Camera  # type: ignore

from genmanip.core.robot.base import BaseEmbodiment
from genmanip.utils.usd_utils import (
    collect_camera_info,
    create_joint_xform_list,
    create_tcp_xform_list,
    get_intrinsic_matrix,
    get_pixel_from_world_point,
    get_tcp_2d_trace,
    get_tcp_3d_trace,
)
from genmanip.utils.standalone.transform_utils import (
    compute_delta_eepose,
    compute_pose2,
    pose_to_transform,
    transform_to_pose,
)

DEFAULT_RGB_SCALE_FACTOR = 256000.0
_STREAM_SENTINEL: tuple[None, None] = (None, None)


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


def fload_array_to_uint16_png(float_array: np.ndarray) -> np.ndarray:
    scaled_array = np.round(float_array * 10000)
    scaled_array = clip_float_values(scaled_array, 0, 65535)
    scaled_array = scaled_array.astype(np.uint16)
    return scaled_array


def uint16_array_to_float_array(uint16_array: np.ndarray) -> np.ndarray:
    float_array = uint16_array.astype(np.float32)
    float_array = float_array / 10000
    return float_array


def _encode_jpg_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img.astype(np.uint8))
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg) failed")
    return pickle.dumps(buf)


def _encode_depth_png_bytes(img: np.ndarray) -> bytes:
    img = fload_array_to_uint16_png(img)
    ok, buf = cv2.imencode(".png", img.astype(np.uint16))
    if not ok:
        raise RuntimeError("cv2.imencode(.png depth) failed")
    return pickle.dumps(buf)


def _encode_seg_png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img.astype(np.uint8))
    if not ok:
        raise RuntimeError("cv2.imencode(.png seg) failed")
    return pickle.dumps(buf)


class RenderRecorder:
    def __init__(
        self,
        cameras: dict[str, Camera],
        embodiment: BaseEmbodiment,
        instruction: str,
        log_dir: str = "logs",
        max_size: int = 10,  # Size in TB
        name: str = "",
        task_data: dict = {},
        tcp_config: list[dict] = [],
        encode_workers: int = 4,
        write_queue_size: int = 64,
        lmdb_commit_every: int = 256,
        record_depth: bool = True,
        logger: logging.Logger | None = None,
    ):
        if name == "":
            self.name = datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")
        else:
            self.name = name
        self.instruction = instruction
        self.data_dir = Path(log_dir)
        self.logger = logger
        if self.logger is not None:
            self.logger.info(f"Generated data will be saved to: {self.data_dir}")
        self.max_size = int(max_size * 1024**4)
        self.json_data_logger = {}
        self.scalar_data_logger = {}
        self.env = None
        self.cameras = cameras
        self.embodiment = embodiment
        self.log_num_steps = 0
        self.task_data = task_data
        self.tcp_config = tcp_config
        self.record_depth = record_depth
        self._encode_workers = max(1, int(encode_workers))
        self._write_queue_size = max(1, int(write_queue_size))
        self._lmdb_commit_every = max(1, int(lmdb_commit_every))

        self._stream_started = False
        self._stream_lock = threading.Lock()
        self._write_queue: queue.Queue[Any] | None = None
        self._writer_thread: threading.Thread | None = None
        self._encoder = None
        self._stream_error: Exception | None = None
        self._stream_keys_color: dict[str, list[bytes]] = {}
        self._stream_keys_depth: dict[str, list[bytes]] = {}
        self._stream_keys_mask: dict[str, list[bytes]] = {}
        self._stream_index_color: dict[str, int] = {}
        self._stream_index_depth: dict[str, int] = {}
        self._stream_index_mask: dict[str, int] = {}

        self.tcp_xform_list = create_tcp_xform_list(
            self.embodiment.robot, self.tcp_config
        )
        self.frame_status = {}
        self.load_static_info()
        self.joint_xform_list = create_joint_xform_list(self.embodiment.robot)

    def load_static_info(self) -> None:
        # camera params
        for camera_name, camera in self.cameras.items():
            intrinsics_matrix = get_intrinsic_matrix(camera)
            self.add_json_data(
                f"observation/{camera_name}/camera_params",
                intrinsics_matrix.tolist(),
            )

        # tcp config
        self.add_json_data(f"observation/tcp_config", self.tcp_config)

    def load_dynamic_info(
        self,
        obj_info: dict[str, dict],
        action: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        gripper_close: np.ndarray,
        name: str | None = None,
        base_motion: np.ndarray = np.array([0.0, 0.0, 0.0]),
        pointcloud: dict[str, np.ndarray] | None = None,
    ) -> None:
        # name frame
        self.add_name_frame(name)
        self.add_scalar_data(f"name", name)

        # observation
        for camera_name, camera in self.cameras.items():
            camera_info = collect_camera_info(camera)
            self.add_color_image(
                f"observation/{camera_name}/color_image", camera_info["rgb"]
            )
            if "depth" in camera_info:
                self.add_depth_image(
                    f"observation/{camera_name}/depth_image", camera_info["depth"]
                )
            if "obj_mask" in camera_info:
                self.add_obj_mask(
                    f"observation/{camera_name}/semantic_mask", camera_info["obj_mask"]
                )
                self.add_scalar_data(
                    f"observation/{camera_name}/semantic_mask_id2labels",
                    camera_info["obj_mask_id2labels"],
                )
            if "bbox2d_tight" in camera_info:
                self.add_scalar_data(
                    f"observation/{camera_name}/bbox2d_tight",
                    camera_info["bbox2d_tight"],
                )
                self.add_scalar_data(
                    f"observation/{camera_name}/bbox2d_tight_id2labels",
                    camera_info["bbox2d_tight_id2labels"],
                )
            if "bbox2d_loose" in camera_info:
                self.add_scalar_data(
                    f"observation/{camera_name}/bbox2d_loose",
                    camera_info["bbox2d_loose"],
                )
                self.add_scalar_data(
                    f"observation/{camera_name}/bbox2d_loose_id2labels",
                    camera_info["bbox2d_loose_id2labels"],
                )
            if "bbox3d" in camera_info:
                self.add_scalar_data(
                    f"observation/{camera_name}/bbox3d", camera_info["bbox3d"]
                )
                self.add_scalar_data(
                    f"observation/{camera_name}/bbox3d_id2labels",
                    camera_info["bbox3d_id2labels"],
                )
            self.add_scalar_data(
                f"observation/{camera_name}/tcp_2d_trace",
                get_tcp_2d_trace(camera, self.tcp_xform_list),
            )
            self.add_scalar_data(
                f"observation/{camera_name}/camera2env_pose",
                pose_to_transform((camera_info["p"], camera_info["q"])),
            )

        # actions
        self.add_scalar_data(f"ee_pose_action", self.embodiment.fk_single(action))
        self.add_scalar_data(
            f"arm_action", action[self.embodiment.default_arm_dof_indices]
        )
        self.add_scalar_data(
            f"gripper_action", action[self.embodiment.default_gripper_dof_indices]
        )
        self.add_scalar_data(f"gripper_close", gripper_close)
        self.add_scalar_data(f"base_motion", base_motion)

        # states
        self.add_scalar_data(f"observation/robot/qpos", qpos)
        self.add_scalar_data(f"observation/robot/qvel", qvel)
        self.add_scalar_data(
            f"observation/robot/ee_pose_state", self.embodiment.fk_single(qpos)
        )
        self.add_scalar_data(
            f"observation/robot/robot2env_pose",
            pose_to_transform(self.embodiment.robot.get_world_pose()),
        )

        # intermediate data
        if pointcloud is not None:
            self.add_scalar_data(f"observation/pointcloud", pointcloud)
        tcp_3d_trace = get_tcp_3d_trace(self.tcp_xform_list)
        self.add_scalar_data(f"tcp_3d_trace", tcp_3d_trace)

        # object info
        for key in obj_info:
            self.add_scalar_data(
                f"observation/obj_pose/{key}/position",
                obj_info[key]["position"],
            )
            self.add_scalar_data(
                f"observation/obj_pose/{key}/orientation",
                obj_info[key]["orientation"],
            )
            self.add_scalar_data(
                f"observation/obj_pose/{key}/scale", obj_info[key]["scale"]
            )
        joint_world_pose = {}
        for joint_name, joint_xform in self.joint_xform_list.items():
            joint_world_pose[joint_name] = joint_xform.get_world_pose()
        self.add_scalar_data(f"observation/robot/joint_world_pose", joint_world_pose)

        self.log_num_steps += 1

    def add_name_frame(self, name: str | None) -> None:
        if name is None:
            return
        if name not in self.frame_status:
            self.frame_status[name] = self.log_num_steps

    def add_scalar_data(self, key: str, value: Any) -> None:
        if key not in self.scalar_data_logger:
            self.scalar_data_logger[key] = []
        self.scalar_data_logger[key].append(value)

    def add_color_image(self, key: str, value: np.ndarray) -> None:
        self._stream_put_image(kind="color", key=key, value=value)

    def add_obj_mask(self, key: str, value: np.ndarray) -> None:
        self._stream_put_image(kind="mask", key=key, value=value)

    def add_depth_image(self, key: str, value: np.ndarray) -> None:
        if not self.record_depth:
            return
        self._stream_put_image(kind="depth", key=key, value=value)

    def add_json_data(self, key: str, data: Any) -> None:
        self.json_data_logger[key] = data

    def _ensure_streaming_started(self) -> None:
        if self._stream_started:
            return
        with self._stream_lock:
            if self._stream_started:
                return

            self.data_dir.mkdir(parents=True, exist_ok=True)
            log_dir_lmdb = self.data_dir / "lmdb"
            if os.path.exists(log_dir_lmdb):
                e = FileExistsError(f"LMDB existed at {log_dir_lmdb}")
                self._stream_error = e
                raise e

            log_dir_lmdb.mkdir(parents=True, exist_ok=True)
            self.env = lmdb.open(str(log_dir_lmdb), map_size=self.max_size)
            self._write_queue = queue.Queue(maxsize=self._write_queue_size)
            from concurrent.futures import ThreadPoolExecutor

            self._encoder = ThreadPoolExecutor(max_workers=self._encode_workers)
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name="RenderRecorderLMDBWriter",
                daemon=True,
            )
            self._writer_thread.start()
            self._stream_started = True

    def _writer_loop(self) -> None:
        if self.env is None or self._write_queue is None:
            return
        txn = None
        num_puts = 0
        try:
            txn = self.env.begin(write=True)
            while True:
                item = self._write_queue.get()
                try:
                    if item == _STREAM_SENTINEL:
                        break
                    key, future = item
                    if key is None or future is None:
                        continue
                    try:
                        value = future.result()
                        txn.put(key, value)
                        num_puts += 1
                        if num_puts % self._lmdb_commit_every == 0:
                            txn.commit()
                            txn = self.env.begin(write=True)
                    except (RuntimeError, TypeError, ValueError, lmdb.Error) as e:
                        self._stream_error = e
                finally:
                    self._write_queue.task_done()
        except (RuntimeError, OSError, TypeError, ValueError, lmdb.Error) as e:
            self._stream_error = e
        finally:
            try:
                if txn is not None:
                    txn.commit()
            except (RuntimeError, OSError, TypeError, ValueError, lmdb.Error) as exc:
                if self.logger is not None:
                    self.logger.warning("Failed to commit LMDB transaction: %s", exc)

    def _stream_put_image(self, kind: str, key: str, value: np.ndarray) -> None:
        self._ensure_streaming_started()

        if self._encoder is None or self._write_queue is None:
            raise RuntimeError("streaming is enabled but not initialized")

        if kind == "color":
            idx = self._stream_index_color.get(key, 0)
            self._stream_index_color[key] = idx + 1
            lmdb_key = f"{key}/{idx:04d}".encode("utf-8")
            self._stream_keys_color.setdefault(key, []).append(lmdb_key)
            encoder = _encode_jpg_bytes
        elif kind == "depth":
            idx = self._stream_index_depth.get(key, 0)
            self._stream_index_depth[key] = idx + 1
            lmdb_key = f"{key}/{idx:04d}".encode("utf-8")
            self._stream_keys_depth.setdefault(key, []).append(lmdb_key)
            encoder = _encode_depth_png_bytes
        elif kind == "mask":
            idx = self._stream_index_mask.get(key, 0)
            self._stream_index_mask[key] = idx + 1
            lmdb_key = f"{key}/{idx:04d}".encode("utf-8")
            self._stream_keys_mask.setdefault(key, []).append(lmdb_key)
            encoder = _encode_seg_png_bytes
        else:
            raise ValueError(f"unknown image kind: {kind}")

        future = self._encoder.submit(encoder, value)
        self._write_queue.put((lmdb_key, future))

        if self._stream_error is not None:
            raise RuntimeError(
                "streaming writer encountered an error"
            ) from self._stream_error

    def _finalize_streaming(self) -> None:
        if not self._stream_started:
            return
        if (
            self._encoder is None
            or self._write_queue is None
            or self._writer_thread is None
        ):
            return
        self._write_queue.put(_STREAM_SENTINEL)
        self._writer_thread.join()
        self._encoder.shutdown(wait=True, cancel_futures=False)
        self._encoder = None

        if self._stream_error is not None:
            raise RuntimeError(
                "streaming writer encountered an error"
            ) from self._stream_error

    def add_delta_info(self) -> None:
        # delta arm action
        delta_arm_actions = []
        delta_arm_actions.append(
            np.array(self.scalar_data_logger["arm_action"][0])
            - np.array(
                self.scalar_data_logger["observation/robot/qpos"][0][
                    self.embodiment.default_dof_indices
                ][self.embodiment.default_arm_dof_indices]
            )
        )
        for i in range(1, len(self.scalar_data_logger["arm_action"])):
            delta_arm_actions.append(
                np.array(self.scalar_data_logger["arm_action"][i])
                - np.array(self.scalar_data_logger["arm_action"][i - 1])
            )
        for delta_arm_action in delta_arm_actions:
            self.add_scalar_data("delta_arm_action", delta_arm_action.tolist())

        # delta ee pose action
        delta_ee_pose_actions = []
        if len(self.scalar_data_logger["ee_pose_action"][0][0]) == 3:
            delta_ee_pose_actions.append(
                compute_delta_eepose(
                    self.scalar_data_logger["ee_pose_action"][0],
                    self.scalar_data_logger["observation/robot/ee_pose_state"][0],
                )
            )
            for i in range(1, len(self.scalar_data_logger["arm_action"])):
                delta_ee_pose_actions.append(
                    compute_delta_eepose(
                        self.scalar_data_logger["ee_pose_action"][i],
                        self.scalar_data_logger["ee_pose_action"][i - 1],
                    )
                )
            for delta_ee_pose_action in delta_ee_pose_actions:
                self.add_scalar_data("delta_ee_pose_action", delta_ee_pose_action)
        else:
            delta_ee_pose_actions.append(
                (
                    compute_delta_eepose(
                        self.scalar_data_logger["ee_pose_action"][0][0],
                        self.scalar_data_logger["observation/robot/ee_pose_state"][0][
                            0
                        ],
                    ),
                    compute_delta_eepose(
                        self.scalar_data_logger["ee_pose_action"][0][1],
                        self.scalar_data_logger["observation/robot/ee_pose_state"][0][
                            1
                        ],
                    ),
                )
            )
            for i in range(1, len(self.scalar_data_logger["arm_action"])):
                delta_ee_pose_actions.append(
                    (
                        compute_delta_eepose(
                            self.scalar_data_logger["ee_pose_action"][i][0],
                            self.scalar_data_logger["ee_pose_action"][i - 1][0],
                        ),
                        compute_delta_eepose(
                            self.scalar_data_logger["ee_pose_action"][i][1],
                            self.scalar_data_logger["ee_pose_action"][i - 1][1],
                        ),
                    )
                )
            for delta_ee_pose_action in delta_ee_pose_actions:
                self.add_scalar_data("delta_ee_pose_action", delta_ee_pose_action)

    def add_grasp_point_3d(self) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
        tcp_list1 = []
        tcp_list2 = []
        eepose_list = []
        task_split_frame = []
        object_name = []
        object_position = []
        object_orientation = []
        for name, frame in self.frame_status.items():
            if name.split("/")[-1] == "pre_grasp":
                task_split_frame.append(frame)
            if name.split("/")[-1] == "post_grasp":
                object_name.append(
                    self.task_data["task_path"][int(name.split("/")[0])]["obj1_uid"]
                )
                object_position.append(
                    self.scalar_data_logger[
                        f"observation/obj_pose/{object_name[-1]}/position"
                    ][frame]
                )
                object_orientation.append(
                    self.scalar_data_logger[
                        f"observation/obj_pose/{object_name[-1]}/orientation"
                    ][frame]
                )
                tcp_list1.append(self.scalar_data_logger[f"tcp_3d_trace"][frame][2])
                tcp_list2.append(self.scalar_data_logger[f"tcp_3d_trace"][frame][3])
                eepose_list.append(
                    self.scalar_data_logger[f"observation/robot/ee_pose_state"][frame]
                )
        task_split_frame.append(len(self.scalar_data_logger["arm_action"]))
        last_frame = task_split_frame.pop(0)
        current_tcp_list = {}
        current_eepose_list = {}
        for idx, (name, object_p, object_q, tcp1, tcp2, eepose, frame) in enumerate(
            zip(
                object_name,
                object_position,
                object_orientation,
                tcp_list1,
                tcp_list2,
                eepose_list,
                task_split_frame,
            )
        ):
            current_object_p = self.scalar_data_logger[
                f"observation/obj_pose/{name}/position"
            ][last_frame]
            current_object_q = self.scalar_data_logger[
                f"observation/obj_pose/{name}/orientation"
            ][last_frame]
            tcp = {}
            camera_pose = {}
            current_tcp1, _ = compute_pose2(
                (current_object_p, current_object_q),
                (object_p, object_q),
                (tcp1, np.array([1.0, 0.0, 0.0, 0.0])),
            )
            current_tcp2, _ = compute_pose2(
                (current_object_p, current_object_q),
                (object_p, object_q),
                (tcp2, np.array([1.0, 0.0, 0.0, 0.0])),
            )
            for camera_name, camera in self.cameras.items():
                tcp[camera_name] = []
                camera_pose[camera_name] = camera.get_world_pose()
            for camera_name, camera in self.cameras.items():
                camera.set_world_pose(
                    *transform_to_pose(
                        self.scalar_data_logger[
                            f"observation/{camera_name}/camera2env_pose"
                        ][last_frame]
                    )
                )
                tcp[camera_name].append(
                    get_pixel_from_world_point(
                        camera,
                        current_tcp1.reshape(3, 1),
                    )
                )
                tcp[camera_name].append(
                    get_pixel_from_world_point(
                        camera,
                        current_tcp2.reshape(3, 1),
                    )
                )
            for camera_name, camera in self.cameras.items():
                camera.set_world_pose(*camera_pose[camera_name])
            if len(eepose[0]) == 3:
                current_eepose = compute_pose2(
                    (current_object_p, current_object_q),
                    (object_p, object_q),
                    eepose,
                )
            else:
                current_eepose = (
                    compute_pose2(
                        (current_object_p, current_object_q),
                        (object_p, object_q),
                        eepose[0],
                    ),
                    compute_pose2(
                        (current_object_p, current_object_q),
                        (object_p, object_q),
                        eepose[1],
                    ),
                )
            tcp["world"] = []
            tcp["world"].append(current_tcp1)
            tcp["world"].append(current_tcp2)
            current_tcp_list[f"{idx}"] = tcp
            current_eepose_list[f"{idx}"] = current_eepose
            last_frame = frame
        return current_tcp_list, current_eepose_list

    def save(
        self, task_name: str | None = None, config_path: str | None = None
    ) -> bool:
        # if os.path.exists(self.data_dir):
        #     return False
        self.data_dir.mkdir(parents=True, exist_ok=True)
        log_dir_lmdb = self.data_dir / "lmdb"
        if (
            os.path.exists(log_dir_lmdb)
            and not self._stream_started
            and self.env is None
        ):
            if self.logger is not None:
                self.logger.info("LMDB Existed, Skipping save.")
            return False

        meta_info = {}
        meta_info["max_size"] = self.max_size
        meta_info["num_steps"] = self.log_num_steps
        meta_info["language_instruction"] = self.instruction
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

        self.add_delta_info()
        current_tcp_list, current_eepose_list = self.add_grasp_point_3d()
        meta_info["task_data"]["grasp_point"] = current_tcp_list
        meta_info["task_data"]["grasp_pose"] = current_eepose_list

        self._finalize_streaming()

        if self.env is None:
            self.env = lmdb.open(str(log_dir_lmdb), map_size=self.max_size)
        txn = self.env.begin(write=True)
        with open(log_dir_lmdb / "info.json", "w") as f:
            json.dump(self.json_data_logger, f)
        meta_info["keys"]["scalar_data"] = []
        for key, value in self.scalar_data_logger.items():
            txn.put(key.encode("utf-8"), pickle.dumps(value))
            meta_info["keys"]["scalar_data"].append(key.encode("utf-8"))
        for k, v in self._stream_keys_color.items():
            meta_info["keys"][k] = v
        if self.record_depth:
            for k, v in self._stream_keys_depth.items():
                meta_info["keys"][k] = v
        for k, v in self._stream_keys_mask.items():
            meta_info["keys"][k] = v
        txn.put("json_data".encode("utf-8"), pickle.dumps(self.json_data_logger))
        meta_info["keys"]["json_data"] = ["json_data".encode("utf-8")]

        txn.commit()
        if self.env is not None:
            self.env.close()
            self.env = None

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
        try:
            self._finalize_streaming()
        except (RuntimeError, TypeError, ValueError, lmdb.Error) as exc:
            if self.logger is not None:
                self.logger.warning("Failed to finalize streaming: %s", exc)
        if self.env is not None:
            try:
                self.env.close()
            except (OSError, RuntimeError, lmdb.Error) as exc:
                if self.logger is not None:
                    self.logger.warning("Failed to close LMDB environment: %s", exc)
            self.env = None

        self.json_data_logger.clear()
        self.scalar_data_logger.clear()
        self.frame_status.clear()
        self._stream_keys_color.clear()
        self._stream_keys_depth.clear()
        self._stream_keys_mask.clear()
        self._stream_index_color.clear()
        self._stream_index_depth.clear()
        self._stream_index_mask.clear()
        self._stream_error = None
        self._stream_started = False
        self._write_queue = None
        self._writer_thread = None

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
