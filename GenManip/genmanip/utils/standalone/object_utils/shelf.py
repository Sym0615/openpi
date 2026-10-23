"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import random

import numpy as np
import open3d as o3d
from shapely.geometry import Polygon

from genmanip.utils.standalone.pc_utils import sort_boundary_points


class Shelf:
    def __init__(
        self,
        centroid: np.ndarray,
        boundary_points: np.ndarray,
        color: list[float] | None = None,
    ) -> None:
        self.centroid = centroid
        self.boundary_polygon = Polygon(boundary_points)
        self.z = centroid[2]
        self.z_min = None
        self.z_max = None
        self.color = color if color else [random.random() for _ in range(3)]  # 随机颜色
        self.sorted_boundary_points = self.sort_boundary_points(boundary_points)

    def sort_boundary_points(self, boundary_points: np.ndarray) -> np.ndarray:
        boundary_points_sorted = sort_boundary_points(
            boundary_points, self.centroid[:2]
        )
        return boundary_points_sorted


def extract_shelf_planes(
    pcd: o3d.geometry.PointCloud,
    upward_threshold: float = 0.9,
    eps: float = 0.05,
    min_points: int = 100,
) -> list[Shelf]:
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.normalize_normals()
    normals = np.asarray(pcd.normals)
    mask = normals[:, 2] > upward_threshold
    filtered_pcd = pcd.select_by_index(np.where(mask)[0])
    if len(filtered_pcd.points) < min_points:
        print("筛选后的点云点数不足，无法进行聚类分析。")
        return []
    labels = np.array(
        filtered_pcd.cluster_dbscan(
            eps=eps, min_points=min_points, print_progress=False
        )
    )
    if labels.size == 0 or labels.max() == -1:
        print("没有检测到任何聚类或所有点均为噪声。")
        return []
    max_label = labels.max()
    shelves = []
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_pcd = filtered_pcd.select_by_index(cluster_indices)
        centroid = cluster_pcd.get_center()
        hull, _ = cluster_pcd.compute_convex_hull()
        hull_vertices = np.asarray(hull.vertices)
        hull_points_2d = hull_vertices[:, :2]
        sorted_hull_points_2d = sort_boundary_points(hull_points_2d, centroid[:2])
        if not isinstance(sorted_hull_points_2d, np.ndarray):
            sorted_hull_points_2d = np.array(sorted_hull_points_2d)
        hull_points_3d = np.hstack(
            (
                sorted_hull_points_2d,
                np.full((sorted_hull_points_2d.shape[0], 1), centroid[2]),
            )
        )
        shelf = Shelf(
            centroid=np.asarray(centroid), boundary_points=sorted_hull_points_2d
        )
        shelves.append(shelf)
    shelves.sort(key=lambda x: x.z)
    for i, shelf in enumerate(shelves):
        if i == 0:
            shelf.z_min = -np.inf
        else:
            shelf.z_min = shelves[i - 1].z_max
        if i < len(shelves) - 1:
            shelf.z_max = (shelf.z + shelves[i + 1].z) / 2
        else:
            shelf.z_max = np.inf
    return shelves


def visualize_shelves(shelves: list[Shelf]) -> None:
    geometries = []
    centroids = [shelf.centroid for shelf in shelves]
    if centroids:
        centroid_pcd = o3d.geometry.PointCloud()
        centroid_pcd.points = o3d.utility.Vector3dVector(centroids)
        centroid_pcd.colors = o3d.utility.Vector3dVector([[0, 0, 0] for _ in centroids])
        geometries.append(centroid_pcd)
    for shelf in shelves:
        hull_points_2d = shelf.sorted_boundary_points
        z = shelf.z
        hull_points_3d = np.hstack(
            (hull_points_2d, np.full((hull_points_2d.shape[0], 1), z))
        )
        if not np.array_equal(hull_points_3d[0], hull_points_3d[-1]):
            hull_points_3d = np.vstack([hull_points_3d, hull_points_3d[0]])
        lines = [[i, i + 1] for i in range(len(hull_points_3d) - 1)]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(hull_points_3d),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector([shelf.color for _ in lines])
        geometries.append(line_set)
    o3d.visualization.draw_geometries(geometries)  # type: ignore
