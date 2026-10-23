"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import copy

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def forward_transform_mesh(
    mesh: o3d.geometry.TriangleMesh,
    scale_factors: np.ndarray,
    quaternion: np.ndarray,
    translation_vector: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    vertices = mesh.vertices
    vertices = vertices * scale_factors
    rotation = R.from_quat(quaternion)
    vertices = rotation.apply(vertices)
    vertices = vertices + translation_vector
    transformed_mesh = copy.deepcopy(mesh)
    transformed_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return transformed_mesh


def forward_transform_point_cloud(
    points: np.ndarray,
    scale_factors: np.ndarray,
    quaternion: np.ndarray,
    translation_vector: np.ndarray,
) -> np.ndarray:
    points = points * scale_factors
    rotation = R.from_quat(quaternion)
    points = rotation.apply(points)
    points = points + translation_vector
    return points


def inverse_transform_mesh(
    mesh: o3d.geometry.TriangleMesh,
    scale_factors: np.ndarray,
    quaternion: np.ndarray,
    translation_vector: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    vertices = mesh.vertices
    vertices = vertices - translation_vector
    rotation = R.from_quat(quaternion)
    vertices = rotation.inv().apply(vertices)
    vertices = vertices / scale_factors
    transformed_mesh = copy.deepcopy(mesh)
    transformed_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return transformed_mesh


def inverse_transform_point_cloud(
    points: np.ndarray,
    scale_factors: np.ndarray,
    quaternion: np.ndarray,
    translation_vector: np.ndarray,
) -> np.ndarray:
    points = points - translation_vector
    rotation = R.from_quat(quaternion)
    points = rotation.inv().apply(points)
    points = points / scale_factors
    return points


def transform_between_meshes(
    mesh_A: o3d.geometry.TriangleMesh,
    scale_A: np.ndarray,
    quat_A: np.ndarray,
    trans_A: np.ndarray,
    scale_B: np.ndarray,
    quat_B: np.ndarray,
    trans_B: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    mesh_in_world_frame = inverse_transform_mesh(mesh_A, scale_A, quat_A, trans_A)
    transformed_mesh_B = forward_transform_mesh(
        mesh_in_world_frame, scale_B, quat_B, trans_B
    )
    return transformed_mesh_B


def transform_between_point_clouds(
    points_A: np.ndarray,
    scale_A: np.ndarray,
    quat_A: np.ndarray,
    trans_A: np.ndarray,
    scale_B: np.ndarray,
    quat_B: np.ndarray,
    trans_B: np.ndarray,
) -> np.ndarray:
    points_in_world_frame = inverse_transform_point_cloud(
        points_A, scale_A, quat_A, trans_A
    )
    transformed_points_B = forward_transform_point_cloud(
        points_in_world_frame, scale_B, quat_B, trans_B
    )

    return transformed_points_B
