"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def adjust_orientation(origin_ori: np.ndarray) -> np.ndarray:
    ori = R.from_quat(origin_ori[[1, 2, 3, 0]])
    if ori.apply(np.array([1, 0, 0]))[0] < 0:
        ori = ori * R.from_euler("z", 180, degrees=True)
    return ori.as_quat()[[3, 0, 1, 2]]


def rot_orientation_by_z_axis(ori: np.ndarray, angle: float) -> np.ndarray:
    return rot_orientation_by_axis(ori, "z", angle)


def rot_orientation_by_axis(
    origin_ori: np.ndarray, axis: str, angle: float
) -> np.ndarray:
    ori = R.from_quat(origin_ori[[1, 2, 3, 0]])
    ori = ori * R.from_euler(axis, angle, degrees=True)
    return ori.as_quat()[[3, 0, 1, 2]]


def adjust_translation_along_quaternion(
    translation: np.ndarray,
    quaternion: np.ndarray,
    distance: float,
    aug_distance: float = 0.0,
) -> np.ndarray:
    rotation = R.from_quat(quaternion[[1, 2, 3, 0]])
    direction_vector = rotation.apply([0, 0, 1])
    reverse_direction = -direction_vector
    new_translation = translation + reverse_direction * distance
    arbitrary_vector = (
        np.array([1, 0, 0]) if direction_vector[0] == 0 else np.array([0, 1, 0])
    )
    perp_vector1 = np.cross(direction_vector, arbitrary_vector)
    perp_vector2 = np.cross(direction_vector, perp_vector1)
    perp_vector1 /= np.linalg.norm(perp_vector1)
    perp_vector2 /= np.linalg.norm(perp_vector2)
    random_shift = np.random.uniform(-aug_distance, aug_distance, size=2)
    new_translation += random_shift[0] * perp_vector1 + random_shift[1] * perp_vector2
    return new_translation


def compute_final_pose(
    P_A0: np.ndarray,
    Q_A0: np.ndarray,
    P_B0: np.ndarray,
    Q_B0: np.ndarray,
    P_A1: np.ndarray,
    Q_A1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rot_A0 = R.from_quat(Q_A0[[1, 2, 3, 0]])
    rot_A1 = R.from_quat(Q_A1[[1, 2, 3, 0]])
    rot_B0 = R.from_quat(Q_B0[[1, 2, 3, 0]])
    rot_BA = rot_A0.inv() * rot_B0
    t_BA = rot_A0.inv().apply(P_B0 - P_A0)
    P_B1 = rot_A1.apply(t_BA) + P_A1
    rot_B1 = rot_A1 * rot_BA
    Q_B1 = rot_B1.as_quat()[[3, 0, 1, 2]]
    return P_B1, Q_B1


def compute_delta_eepose(
    pose1: tuple[np.ndarray, np.ndarray], pose2: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    return the delta eepose between two poses: pose1 - pose2
    """
    pose1_transform = pose_to_transform(pose1)
    pose2_transform = pose_to_transform(pose2)
    delta_transform = pose1_transform @ np.linalg.inv(pose2_transform)
    return transform_to_pose(delta_transform)


def transform_to_pose(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trans = transform[:3, 3]
    quat = R.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return trans, quat


def pose_to_transform(pose: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    trans, quat = pose
    transform = np.eye(4)
    transform[:3, 3] = trans
    transform[:3, :3] = R.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    return transform


def compute_pose2(
    current_pose1: tuple[np.ndarray, np.ndarray],
    previous_pose1: tuple[np.ndarray, np.ndarray],
    previous_pose2: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    current_pose1_transform = pose_to_transform(current_pose1)
    previous_pose1_transform = pose_to_transform(previous_pose1)
    previous_pose2_transform = pose_to_transform(previous_pose2)
    current_pose2_transform = (
        previous_pose2_transform
        @ np.linalg.inv(previous_pose1_transform)
        @ current_pose1_transform
    )
    return transform_to_pose(current_pose2_transform)


def pose_frame_to_world(
    pose_in_frame: tuple[np.ndarray, np.ndarray],
    frame_in_world: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    pose_in_frame: T_frame_pose
    frame_in_world: T_world_frame
    return: T_world_pose
    """
    T_world_frame = pose_to_transform(frame_in_world)
    T_frame_pose = pose_to_transform(pose_in_frame)

    T_world_pose = T_world_frame @ T_frame_pose
    return transform_to_pose(T_world_pose)


def pose_world_to_frame(
    pose_in_world: tuple[np.ndarray, np.ndarray],
    frame_in_world: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    pose_in_world: T_world_pose
    frame_in_world: T_world_frame
    return: T_frame_pose
    """
    T_world_frame = pose_to_transform(frame_in_world)
    T_world_pose = pose_to_transform(pose_in_world)

    T_frame_world = np.linalg.inv(T_world_frame)
    T_frame_pose = T_frame_world @ T_world_pose
    return transform_to_pose(T_frame_pose)
