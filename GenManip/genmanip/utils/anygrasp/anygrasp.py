"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import random
import ssl
import requests
from requests.adapters import HTTPAdapter

import numpy as np
import open3d as o3d
import urllib3
from scipy.spatial.transform import Rotation as R
from urllib3.exceptions import InsecureRequestWarning
from urllib3.util.ssl_ import create_urllib3_context

from omni.isaac.sensor import Camera  # type: ignore

from genmanip.utils.usd_utils.camera_utils import get_src, get_intrinsic_matrix

urllib3.disable_warnings(InsecureRequestWarning)


class HostnameIgnoringAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs) -> None:
        context = create_urllib3_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        kwargs["ssl_context"] = context
        return super().init_poolmanager(*args, **kwargs)


def anygrasp_response_to_list(grasp_groups: list[dict]) -> list[dict]:
    grasp_list = []
    for grasp in grasp_groups:
        grasp_list.append(
            {
                "translation": grasp["translation"],
                "orientation": grasp["rotation_matrix"],
                "depth": grasp["depth"],
                "score": grasp["score"],
            }
        )
    return grasp_list


def compute_distance_list_from_point_list_to_mesh(
    mesh: o3d.geometry.TriangleMesh, points: np.ndarray
) -> tuple[list[np.ndarray], list[float]]:
    pcd_mesh = mesh.sample_points_uniformly(number_of_points=10000)
    mesh_points = np.asarray(pcd_mesh.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd_mesh)
    nearest_points = []
    distances = []
    for point in points:
        [_, idx, dist] = kdtree.search_knn_vector_3d(point, 1)
        nearest_point = mesh_points[idx[0]]
        distance = np.sqrt(dist[0])
        nearest_points.append(nearest_point)
        distances.append(distance)
    return nearest_points, distances


def find_closest_grasp_to_mesh(
    mesh: o3d.geometry.TriangleMesh,
    grasp_list: list[dict],
    distance_threshold: float = 0.09,
    angle_threshold: float | None = None,
    distance_only: bool = False,
    idx: list[int] = [0],
) -> tuple[dict | None, np.ndarray | None]:
    grasp_points = np.array([grasp["translation"] for grasp in grasp_list])
    nearest_points, distances = compute_distance_list_from_point_list_to_mesh(
        mesh, grasp_points
    )
    distances = np.array(distances)
    valid_indices = np.where(distances <= distance_threshold)[0]
    if len(valid_indices) == 0:
        return None, None

    if distance_only:
        min_index = valid_indices[np.argmin(distances[valid_indices])]
        closest_grasp = grasp_list[min_index]
        closest_distance = distances[min_index]
    else:
        filtered_grasps = []
        for i in valid_indices:
            grasp = grasp_list[i]
            grasp_quaternion = grasp["orientation"]
            grasp_rotation = R.from_quat(grasp_quaternion[[1, 2, 3, 0]])
            if angle_threshold is not None:
                relative_rotation = (
                    R.from_quat(np.array([1, 0, 0, 0])).inv() * grasp_rotation
                )
                angle = relative_rotation.magnitude() * (180 / np.pi)
                if angle <= angle_threshold:
                    filtered_grasps.append((grasp, distances[i]))
            else:
                filtered_grasps.append((grasp, distances[i]))

        if not filtered_grasps:
            return None, None

        filtered_grasps.sort(key=lambda x: x[0]["score"], reverse=True)
        _idx = [id for id in idx if id < len(filtered_grasps)]
        if len(_idx) == 0:
            return None, None
        choose_idx = random.choice(_idx)
        closest_grasp, closest_distance = filtered_grasps[choose_idx]

    return closest_grasp, closest_distance


def find_closest_grasp_to_point(
    point: np.ndarray,
    grasp_list: list[dict],
    distance_threshold: float = 0.09,
    angle_threshold: float | None = None,
    distance_only: bool = False,
) -> dict | None:
    grasp_points = np.array([grasp["translation"] for grasp in grasp_list])
    distances = np.linalg.norm(grasp_points - np.array(point), axis=1)
    within_threshold_indices = np.where(distances <= distance_threshold)[0]
    if len(within_threshold_indices) == 0:
        return None
    if distance_only:
        min_index = within_threshold_indices[
            np.argmin(distances[within_threshold_indices])
        ]
        closest_grasp = grasp_list[min_index]
    else:
        filtered_grasps = []
        for i in within_threshold_indices:
            grasp = grasp_list[i]
            grasp_quaternion = grasp["orientation"]
            grasp_rotation = R.from_quat(grasp_quaternion[[1, 2, 3, 0]])
            if angle_threshold is not None:
                relative_rotation = (
                    R.from_quat(np.array([1, 0, 0, 0])).inv() * grasp_rotation
                )
                angle = relative_rotation.magnitude() * (180 / np.pi)
                if angle <= angle_threshold:
                    filtered_grasps.append(grasp)
            else:
                filtered_grasps.append(grasp)
        if not filtered_grasps:
            return None
        sorted_grasps = sorted(filtered_grasps, key=lambda x: x["score"], reverse=True)
        closest_grasp = sorted_grasps[0]
    return closest_grasp


def get_grasp_pose(
    colors: np.ndarray,
    depth: np.ndarray,
    fx: float = 560.0,
    fy: float = 560.0,
    cx: float = 640.0,
    cy: float = 360.0,
    scale: float = 1000.0,
    address: str = "127.0.0.1",
    port: str = "5001",
) -> list[dict] | None:
    data = {
        "colors": colors.tolist(),
        "depths": (depth * scale).tolist(),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "scale": scale,
        "dense_mode": True,
    }
    session = requests.Session()
    session.mount("https://", HostnameIgnoringAdapter())
    response = requests.post(
        f"http://{address}:{port}/process", json=data, verify=False
    )
    # response = requests.post(f"http://{address}:{port}/process", json=data)
    if response.status_code == 200:
        processed_data = response.json()
        return anygrasp_response_to_list(processed_data["grasp_groups"])
    else:
        print("Failed to process data:", response.text)


def request_anygrasp(
    camera: Camera,
    address: dict[str, list[str]] = {"127.0.0.1": ["5001"]},
) -> list[dict] | None:
    address_list = [(key, port) for key, value in address.items() for port in value]
    addr, port = random.choice(address_list)
    colors = get_src(camera, "rgb")
    colors = np.asarray(colors)
    depth = get_src(camera, "depth")
    depth = np.asarray(depth)
    intrinsics = get_intrinsic_matrix(camera)
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    grasp_list = get_grasp_pose(
        colors=colors,
        depth=depth,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        address=addr,
        port=port,
    )
    world_grasp_list = []
    if grasp_list is None or (isinstance(grasp_list, list) and len(grasp_list) == 0):
        print(f"grasp list has nothing")
        return None
    for grasp in grasp_list:
        point, pose = get_world_grasp_from_camera_coords(
            camera.get_local_pose()[0],
            camera.get_local_pose()[1],
            grasp["translation"],
            grasp["orientation"],
        )
        world_grasp_list.append(
            {
                "translation": point,
                "orientation": pose,
                "depth": grasp["depth"],
                "score": grasp["score"],
            }
        )
    return world_grasp_list


def get_init_grasp(
    camera: Camera,
    mesh: o3d.geometry.TriangleMesh,
    address: dict[str, list[str]] = {"127.0.0.1": ["5001"]},
    allow_fixed_grasp: bool = False,
    force_fixed_grasp: bool = False,
    idx: list[int] = [0],
) -> dict:
    if force_fixed_grasp:
        top_point_of_mesh = mesh.get_max_bound()
        init_grasp = {
            "translation": np.array([0.0, 0.0, top_point_of_mesh[2]]),
            "orientation": np.array([0.0, 1.0, 0.0, 0.0]),
        }
        return init_grasp
    world_grasp_list = request_anygrasp(camera, address)
    if world_grasp_list is None or len(world_grasp_list) == 0:
        raise ValueError("server return empty grasp list")
    init_grasp, _ = find_closest_grasp_to_mesh(
        mesh, world_grasp_list, distance_threshold=0.08, idx=idx
    )
    if init_grasp is None:
        if allow_fixed_grasp:
            top_point_of_mesh = mesh.get_max_bound()
            init_grasp = {
                "translation": np.array([0.0, 0.0, top_point_of_mesh[2]]),
                "orientation": np.array([0.0, 1.0, 0.0, 0.0]),
            }
        else:
            raise ValueError("find no valid grasp")
    return init_grasp


def get_world_grasp_from_camera_coords(
    camera_position: np.ndarray,
    camera_quaternion: np.ndarray,
    point_3d: np.ndarray,
    matrix_grasp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_cam_to_world = R.from_quat(camera_quaternion[[1, 2, 3, 0]])
    transform_costum_pose = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    correction_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    point_in_world_frame = (
        rotation_cam_to_world.as_matrix()
        @ correction_matrix
        @ transform_costum_pose
        @ point_3d
        + camera_position
    )
    rotation_quat_cam = R.from_matrix(matrix_grasp)
    rotation_world = (
        rotation_cam_to_world
        * R.from_matrix(correction_matrix @ transform_costum_pose)
        * rotation_quat_cam
        * R.from_matrix(np.linalg.inv(transform_costum_pose))
    ).as_quat()[[3, 0, 1, 2]]
    return point_in_world_frame, rotation_world
