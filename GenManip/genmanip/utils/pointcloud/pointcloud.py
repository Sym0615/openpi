"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from copy import deepcopy
from filelock import SoftFileLock, Timeout
import os
from pathlib import Path
import uuid

import numpy as np
import open3d as o3d
from tqdm import tqdm

from omni.isaac.core.prims import XFormPrim  # type: ignore
from omni.isaac.core.utils.prims import get_prim_at_path, get_prim_parent, get_prim_path  # type: ignore

from genmanip.utils.pointcloud.transform import (
    forward_transform_mesh,
    inverse_transform_mesh,
    transform_between_meshes,
    transform_between_point_clouds,
)
from genmanip.utils.pointcloud.utils import MeshInfo, PointCloudInfo
from genmanip.utils.usd_utils.transform_utils import (
    quat_wxyz_to_xyzw,
    resolve_prim_local_transform,
)
from genmanip.utils.usd_utils import get_mesh_from_prim
from genmanip.utils.standalone.pc_utils import get_pcd_from_mesh

MESH_FILELOCK_TIMEOUT_SECONDS = float(
    os.environ.get("GENMANIP_MESH_FILELOCK_TIMEOUT", "60.0")
)


def get_current_mesh(
    object: XFormPrim, mesh_dict: MeshInfo
) -> o3d.geometry.TriangleMesh:
    scale = np.array([1, 1, 1])
    trans, quat = object.get_world_pose()
    quat = quat[[1, 2, 3, 0]]
    return transform_between_meshes(
        mesh_dict.mesh,
        mesh_dict.scale,
        mesh_dict.quat,
        mesh_dict.trans,
        scale,
        quat,
        trans,
    )


def get_current_meshList(
    object_list: dict[str, XFormPrim], mesh_dict: dict[str, MeshInfo]
) -> dict[str, o3d.geometry.TriangleMesh]:
    updatedMeshList = {}
    for key in mesh_dict:
        mesh = get_current_mesh(object_list[key], mesh_dict[key])
        if mesh is not None:
            updatedMeshList[key] = mesh
    return updatedMeshList


def get_current_pcList_by_meshList(
    object_list: dict[str, XFormPrim], mesh_list: dict[str, MeshInfo]
) -> dict[str, np.ndarray]:
    mesh_list = get_current_meshList(object_list, mesh_list)
    return meshlist_to_pclist(mesh_list)


def get_current_pointCloud(
    object: XFormPrim, point_dict: PointCloudInfo
) -> o3d.geometry.PointCloud:
    scale = np.array([1, 1, 1])
    trans, quat = object.get_world_pose()
    quat = quat[[1, 2, 3, 0]]
    return transform_between_point_clouds(
        point_dict.points,
        point_dict.scale,
        point_dict.quat,
        point_dict.trans,
        scale,
        quat,
        trans,
    )


def get_current_pointCloutList(
    object_list: dict[str, XFormPrim], point_dict: dict[str, PointCloudInfo]
) -> dict[str, o3d.geometry.PointCloud]:
    updatedPointCloudList = {}
    for key in point_dict:
        updatedPointCloudList[key] = get_current_pointCloud(
            object_list[key], point_dict[key]
        )
    return updatedPointCloudList


def _read_cached_mesh(
    mesh_path: str, remove_if_corrupted: bool = False
) -> o3d.geometry.TriangleMesh | None:
    if not os.path.exists(mesh_path):
        return None
    try:
        return o3d.io.read_triangle_mesh(mesh_path)
    except (RuntimeError, OSError, ValueError):
        if remove_if_corrupted:
            try:
                os.remove(mesh_path)
            except FileNotFoundError:
                # Mesh file may be removed by another cleanup step.
                pass
            except OSError as exc:
                print(
                    f"Warning: failed to remove corrupted mesh file {mesh_path}: {exc}"
                )
        return None


def _build_mesh_from_prim(object: XFormPrim) -> o3d.geometry.TriangleMesh | None:
    try:
        mesh = get_mesh_from_prim(object.prim)
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None
    scale = object.get_local_scale()
    trans, quat = object.get_local_pose()
    quat = quat[[1, 2, 3, 0]]
    return inverse_transform_mesh(mesh, scale, quat, trans)


def _write_mesh_atomically(mesh_path: str, mesh: o3d.geometry.TriangleMesh) -> None:
    base, ext = os.path.splitext(mesh_path)
    temp_path = f"{base}.tmp-{uuid.uuid4().hex}{ext}"
    try:
        o3d.io.write_triangle_mesh(temp_path, mesh)
        os.replace(temp_path, mesh_path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                # File can disappear between existence check and remove in concurrent use.
                pass
            except OSError as exc:
                print(
                    f"Warning: failed to cleanup temporary mesh file {temp_path}: {exc}"
                )
                pass


def get_mesh_info_by_load(object: XFormPrim, mesh_path: str) -> MeshInfo | None:
    # Fast path: avoid lock contention when cache already exists.
    mesh = _read_cached_mesh(mesh_path)
    if mesh is not None:
        mesh_info = MeshInfo()
        mesh_info.mesh = get_world_mesh(mesh, object.prim_path)
        mesh_info.trans, mesh_info.quat = object.get_world_pose()
        mesh_info.quat = mesh_info.quat[[1, 2, 3, 0]]
        mesh_info.scale = np.array([1, 1, 1])
        return mesh_info

    lock = SoftFileLock(mesh_path + "_soft.lock", timeout=MESH_FILELOCK_TIMEOUT_SECONDS)
    try:
        with lock:
            mesh = _read_cached_mesh(mesh_path, remove_if_corrupted=True)
            if mesh is None:
                Path(mesh_path).parent.mkdir(parents=True, exist_ok=True)
                mesh = _build_mesh_from_prim(object)
                if mesh is None:
                    return None
                print(f"save mesh to {mesh_path}")
                _write_mesh_atomically(mesh_path, mesh)
    except Timeout:
        # In large distributed runs, fallback to local mesh build instead of stalling.
        mesh = _read_cached_mesh(mesh_path)
        if mesh is None:
            mesh = _build_mesh_from_prim(object)

    if mesh is None:
        raise Exception(f"Failed to load mesh from {mesh_path}")
    mesh_info = MeshInfo()
    mesh_info.mesh = get_world_mesh(mesh, object.prim_path)
    mesh_info.trans, mesh_info.quat = object.get_world_pose()
    mesh_info.quat = mesh_info.quat[[1, 2, 3, 0]]
    mesh_info.scale = np.array([1, 1, 1])
    return mesh_info


def get_mesh_info(object: XFormPrim) -> MeshInfo | None:
    try:
        mesh = get_mesh_from_prim(object.prim)
    except (RuntimeError, ValueError, TypeError, AttributeError):
        return None
    scale = object.get_local_scale()
    trans, quat = object.get_local_pose()
    quat = quat[[1, 2, 3, 0]]
    mesh = inverse_transform_mesh(mesh, scale, quat, trans)
    mesh_info = MeshInfo()
    mesh_info.mesh = get_world_mesh(mesh, object.prim_path)
    mesh_info.trans, mesh_info.quat = object.get_world_pose()
    mesh_info.quat = mesh_info.quat[[1, 2, 3, 0]]
    mesh_info.scale = np.array([1, 1, 1])
    return mesh_info


def get_world_mesh(
    mesh: o3d.geometry.TriangleMesh, prim_path: str
) -> o3d.geometry.TriangleMesh:
    prim = get_prim_at_path(prim_path)
    mesh = deepcopy(mesh)
    while get_prim_path(prim) != "/":
        local_transform = resolve_prim_local_transform(prim)
        mesh = forward_transform_mesh(
            mesh,
            local_transform.scale,
            quat_wxyz_to_xyzw(local_transform.quat_wxyz),
            local_transform.translation,
        )
        prim = get_prim_parent(prim)
    return mesh


def meshDict2pointCloudDict(
    mesh_dict: dict[str, MeshInfo],
) -> dict[str, PointCloudInfo]:
    pointcloud_dict = {}
    for key in mesh_dict:
        pointCloud_info = mesh_info2pointCloud_info(mesh_dict[key])
        if pointCloud_info is not None:
            pointcloud_dict[key] = pointCloud_info
    return pointcloud_dict


def meshlist_to_pclist(
    meshlist: dict[str, o3d.geometry.TriangleMesh],
) -> dict[str, np.ndarray]:
    pointcloudlist = {}
    for key in meshlist:
        try:
            if key == "00000000000000000000000000000000":
                pointcloudlist[key] = np.asarray(
                    get_pcd_from_mesh(meshlist[key], num_points=100000).points
                )
            else:
                pointcloudlist[key] = np.asarray(
                    get_pcd_from_mesh(meshlist[key], num_points=10000).points
                )
        except (RuntimeError, ValueError):
            continue
    return pointcloudlist


def mesh_info2pointCloud_info(mesh_info: MeshInfo) -> PointCloudInfo | None:
    try:
        points = np.asarray(get_pcd_from_mesh(mesh_info.mesh).points)
        scale = mesh_info.scale
        trans = mesh_info.trans
        quat = mesh_info.quat
        return PointCloudInfo(
            points=points,
            scale=scale,
            trans=trans,
            quat=quat,
        )
    except (RuntimeError, ValueError, TypeError, AttributeError):
        return None


def objectList2meshList(
    object_list: dict[str, XFormPrim], mesh_folder_path: str | None = None
) -> dict[str, MeshInfo]:
    mesh_dict = {}
    for key in tqdm(object_list):
        if key == "defaultGroundPlane":
            continue
        if mesh_folder_path is not None:
            mesh_info = get_mesh_info_by_load(
                object_list[key], os.path.join(mesh_folder_path, f"{key}.obj")
            )
        else:
            mesh_info = get_mesh_info(object_list[key])
        if mesh_info is not None:
            mesh_dict[key] = mesh_info
    return mesh_dict


def objectList2pointCloudList(
    object_list: dict[str, XFormPrim], visualize: bool = False
) -> dict[str, PointCloudInfo]:
    mesh_dict = objectList2meshList(object_list)
    point_dict = meshDict2pointCloudDict(mesh_dict)
    if visualize:
        all_points = []
        for key in point_dict:
            all_points.append(point_dict[key].points)
        all_points = np.vstack(all_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        o3d.visualization.draw_geometries([pcd])  # type: ignore
    return point_dict
