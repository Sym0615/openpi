"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import numpy as np
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation as R


def joint_positions_to_ee_pose_translation_euler(
    joint_positions: np.ndarray,
) -> np.ndarray:
    ee_pose = rtb.models.Panda().fkine(q=joint_positions, end="panda_hand").A  # type: ignore[attr-defined]
    translation = ee_pose[:3, 3]
    euler_angles = R.from_matrix(ee_pose[:3, :3]).as_euler("xyz", degrees=True)
    return np.concatenate([translation, euler_angles])


def joint_positions_to_position_and_orientation(
    joint_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ee_pose = rtb.models.Panda().fkine(q=joint_positions, end="panda_hand").A  # type: ignore[attr-defined]
    translation = ee_pose[:3, 3]
    orientation = R.from_matrix(ee_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return translation, orientation


def joint_position_to_end_effector_pose(
    joint_position: np.ndarray, panda: rtb.models.Panda | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if panda is None:
        panda = rtb.models.Panda()
    hand_pose = panda.fkine(q=joint_position, end="panda_hand").A  # type: ignore[attr-defined]
    position = hand_pose[:3, 3]
    rotation = hand_pose[:3, :3]
    orientation = R.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    return position, orientation
