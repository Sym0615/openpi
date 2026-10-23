"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import math

import numpy as np
from scipy.spatial.transform import Rotation as R

from omni.isaac.core.prims import XFormPrim  # type: ignore
from omni.isaac.sensor import Camera  # type: ignore

from genmanip.utils.standalone.pc_utils import get_world_corners_from_bbox3d
from genmanip.utils.standalone.transform_utils import pose_to_transform


def get_tcp_3d_trace(tcp_xform_list: list[XFormPrim]) -> list[np.ndarray]:
    tcp_3d_trace = []
    for tcp in tcp_xform_list:
        position, orientation = tcp.get_world_pose()
        tcp_3d_trace.append(position)
    return tcp_3d_trace


def get_tcp_2d_trace(
    camera: Camera, tcp_xform_list: list[XFormPrim]
) -> list[np.ndarray]:
    tcp_3d_trace = get_tcp_3d_trace(tcp_xform_list)
    tcp_2d_trace = []
    for tcp in tcp_3d_trace:
        pixel = get_pixel_from_world_point(camera, tcp.reshape(3, 1))[0]
        tcp_2d_trace.append(pixel)
    return tcp_2d_trace


def collect_camera_info(camera: Camera) -> dict:
    info = {}
    info["p"], info["q"] = camera.get_world_pose()
    info["rgb"] = get_src(camera, "rgb")
    if camera._custom_annotators["distance_to_image_plane"] is not None:
        info["depth"] = get_src(camera, "depth")
    if camera._custom_annotators["semantic_segmentation"] is not None:
        seg_data = get_src(camera, "seg")
        if seg_data is not None and isinstance(seg_data, dict):
            info["obj_mask"] = seg_data["mask"]
            info["obj_mask_id2labels"] = seg_data["id2labels"]
    if camera._custom_annotators["bounding_box_2d_tight"] is not None:
        result = get_src(camera, "bbox2d_tight")
        if result is not None:
            info["bbox2d_tight"], info["bbox2d_tight_id2labels"] = result
    if camera._custom_annotators["bounding_box_2d_loose"] is not None:
        result = get_src(camera, "bbox2d_loose")
        if result is not None:
            info["bbox2d_loose"], info["bbox2d_loose_id2labels"] = result
    if camera._custom_annotators["bounding_box_3d"] is not None:
        result = get_src(camera, "bbox3d")
        if result is not None:
            info["bbox3d"], info["bbox3d_id2labels"] = result
    if camera._custom_annotators["motion_vectors"] is not None:
        info["motion_vectors"] = get_src(camera, "motion_vectors")
    info["focal_length"] = camera.get_focal_length()
    info["focus_distance"] = camera.get_focus_distance()
    info["frequency"] = camera.get_frequency()
    info["horizontal_aperture"] = camera.get_horizontal_aperture()
    info["horizontal_fov"] = camera.get_horizontal_fov()
    info["vertical_aperture"] = camera.get_vertical_aperture()
    info["vertical_fov"] = camera.get_vertical_fov()
    info["intrinsics_matrix"] = get_intrinsic_matrix(camera)
    return info


def collect_camera_info_eval(camera: Camera) -> dict:
    info = {}
    info["p"], info["q"] = camera.get_world_pose()
    info["rgb"] = get_src(camera, "rgb")
    if camera._custom_annotators["distance_to_image_plane"] is not None:
        info["depth"] = get_src(camera, "depth")
    else:
        info["depth"] = None
    info["focal_length"] = camera.get_focal_length()
    info["focus_distance"] = camera.get_focus_distance()
    info["frequency"] = camera.get_frequency()
    info["horizontal_aperture"] = camera.get_horizontal_aperture()
    info["horizontal_fov"] = camera.get_horizontal_fov()
    info["vertical_aperture"] = camera.get_vertical_aperture()
    info["vertical_fov"] = camera.get_vertical_fov()
    info["intrinsics_matrix"] = get_intrinsic_matrix(camera)
    return info


def get_eval_camera_data(camera_list: dict) -> dict:
    # OPT: eval only consumes img["rgb"]; depth / intrinsics / pose are all
    # discarded by the caller, so skip those USD queries entirely.
    return {
        camera_name: {"rgb": get_src(camera, "rgb")}
        for camera_name, camera in camera_list.items()
    }


def get_depth(camera: Camera) -> np.ndarray | None:
    depth = camera._custom_annotators["distance_to_image_plane"].get_data()
    if isinstance(depth, np.ndarray) and depth.size > 0:
        return depth
    else:
        return None


def get_pointcloud(camera: Camera) -> np.ndarray | None:
    cloud = camera._custom_annotators["pointcloud"].get_data()["data"]
    if isinstance(cloud, np.ndarray) and cloud.size > 0:
        return cloud
    else:
        return None


def get_objectmask(camera: Camera) -> dict | None:
    annotator = camera._custom_annotators["semantic_segmentation"]
    annotation_data = annotator.get_data()
    mask = annotation_data["data"]
    idToLabels = annotation_data["info"]["idToLabels"]
    if isinstance(mask, np.ndarray) and mask.size > 0:
        return dict(mask=mask.astype(np.int8), id2labels=idToLabels)
    else:
        return None


def get_rgb(camera: Camera) -> np.ndarray | None:
    frame = camera.get_rgba()
    if isinstance(frame, np.ndarray) and frame.size > 0:
        frame = frame[:, :, :3]
        return frame
    else:
        return None


def get_bounding_box_2d_tight(camera: Camera) -> tuple[np.ndarray, dict]:
    annotator = camera._custom_annotators["bounding_box_2d_tight"]
    annotation_data = annotator.get_data()
    bbox = annotation_data["data"]
    info = annotation_data["info"]
    return bbox, info["idToLabels"]


def get_bounding_box_2d_loose(camera: Camera) -> tuple[np.ndarray, dict]:
    annotator = camera._custom_annotators["bounding_box_2d_loose"]
    annotation_data = annotator.get_data()
    bbox = annotation_data["data"]
    info = annotation_data["info"]
    return bbox, info["idToLabels"]


def get_bounding_box_3d(camera: Camera) -> tuple[list[dict], dict]:
    annotator = camera._custom_annotators["bounding_box_3d"]
    annotation_data = annotator.get_data()
    bbox = annotation_data["data"]
    info = annotation_data["info"]
    bbox_data = []
    for box in bbox:
        extents = {}
        (
            extents["class"],
            extents["x_min"],
            extents["y_min"],
            extents["z_min"],
            extents["x_max"],
            extents["y_max"],
            extents["z_max"],
            extents["transform"],
            _,
        ) = box
        extents["corners"] = get_world_corners_from_bbox3d(extents)
        bbox_data.append(extents)
    return bbox_data, info["idToLabels"]


def get_motion_vectors(camera: Camera) -> np.ndarray:
    annotator = camera._custom_annotators["motion_vectors"]
    annotation_data = annotator.get_data()
    motion_vectors = annotation_data
    return motion_vectors


def get_src(camera: Camera, type: str) -> np.ndarray | dict | tuple | None:
    if type == "rgb":
        return get_rgb(camera)
    if type == "depth":
        return get_depth(camera)
    if type == "cloud":
        return get_pointcloud(camera)
    if type == "seg":
        return get_objectmask(camera)
    if type == "bbox2d_tight":
        return get_bounding_box_2d_tight(camera)
    if type == "bbox2d_loose":
        return get_bounding_box_2d_loose(camera)
    if type == "bbox3d":
        return get_bounding_box_3d(camera)
    if type == "motion_vectors":
        return get_motion_vectors(camera)


def get_world_point_from_pixel_(camera: Camera, point: np.ndarray) -> np.ndarray:
    depth = np.asarray(get_src(camera, "depth"))
    return camera.get_world_points_from_image_coords(
        np.array([int(point[0]), int(point[1])]).reshape(-1, 2),
        np.array([depth[int(point[1]), int(point[0])]]).reshape(-1),
    )[0]


def get_pixel_from_world_point_(camera: Camera, point: np.ndarray) -> np.ndarray:
    return camera.get_image_coords_from_world_points(point.reshape(-1, 3))


def get_world_point_from_pixel(camera: Camera, point: np.ndarray) -> np.ndarray:
    intrinsic = get_intrinsic_matrix(camera)
    translation, quaternion = camera.get_world_pose()
    depth = get_src(camera, "depth")
    intrinsic = np.array(intrinsic)
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    x, y = point[0], point[1]
    depth = np.asarray(depth)
    Z = depth[int(y)][int(x)]
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    point_in_camera_frame = np.array([X, Y, Z])
    add_rotation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    point_in_camera_frame = add_rotation @ point_in_camera_frame
    camera_to_world = pose_to_transform((translation, quaternion))
    point_in_world_frame = camera_to_world @ np.array([*point_in_camera_frame, 1])
    point3d = point_in_world_frame[:3]
    return point3d


def get_pixel_from_world_point(camera: Camera, point: np.ndarray) -> np.ndarray:
    point = point.reshape(-1, 3)
    translation, quaternion = camera.get_world_pose()
    camera_to_world = pose_to_transform((translation, quaternion))
    world_to_camera = np.linalg.inv(camera_to_world)
    homogeneous_points = np.hstack([point, np.ones((point.shape[0], 1))])
    points_in_camera_frame = np.dot(homogeneous_points, world_to_camera.T)
    point_in_camera_frame = points_in_camera_frame[:, :3]
    add_rotation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    inv_rotation = np.linalg.inv(add_rotation)
    rotated_point = np.dot(point_in_camera_frame, inv_rotation.T)
    intrinsic = get_intrinsic_matrix(camera)
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    X, Y, Z = rotated_point[:, 0], rotated_point[:, 1], rotated_point[:, 2]
    x = (fx * X / Z) + cx
    y = (fy * Y / Z) + cy
    return np.column_stack((x, y))


def get_intrinsic_matrix(camera: Camera) -> np.ndarray:
    fx, fy = compute_fx_fy(
        camera, camera.get_resolution()[1], camera.get_resolution()[0]
    )
    cx, cy = camera.get_resolution()[0] / 2, camera.get_resolution()[1] / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def compute_fx_fy(camera: Camera, height: int, width: int) -> tuple[float, float]:
    focal_length = camera.get_focal_length()
    horiz_aperture = camera.get_horizontal_aperture()
    vert_aperture = camera.get_vertical_aperture()
    near, far = camera.get_clipping_range()
    fov = 2 * np.arctan(0.5 * horiz_aperture / focal_length)
    focal_x = height * focal_length / vert_aperture
    focal_y = width * focal_length / horiz_aperture
    return focal_x, focal_y


def set_camera_rational_polynomial(
    camera: Camera,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    pixel_size: float = 3,
    f_stop: float = 2.0,
    focus_distance: float = 0.3,
    D: np.ndarray | None = None,
) -> Camera:
    if D is None:
        D = np.zeros(8)
    camera.initialize()
    camera.set_resolution([width, height])
    camera.set_clipping_range(0.02, 5)
    horizontal_aperture = pixel_size * 1e-3 * width
    vertical_aperture = pixel_size * 1e-3 * height
    focal_length_x = fx * pixel_size * 1e-3
    focal_length_y = fy * pixel_size * 1e-3
    focal_length = (focal_length_x + focal_length_y) / 2  # in mm
    camera.set_focal_length(focal_length / 10.0)
    camera.set_focus_distance(focus_distance)
    camera.set_lens_aperture(f_stop * 100.0)
    camera.set_horizontal_aperture(horizontal_aperture / 10.0)
    camera.set_vertical_aperture(vertical_aperture / 10.0)
    camera.set_clipping_range(0.05, 1.0e5)
    diagonal = 2 * math.sqrt(max(cx, width - cx) ** 2 + max(cy, height - cy) ** 2)
    diagonal_fov = 2 * math.atan2(diagonal, fx + fy) * 180 / math.pi
    camera.set_projection_type("fisheyePolynomial")
    camera.set_rational_polynomial_properties(width, height, cx, cy, diagonal_fov, D)
    return camera


def set_camera_look_at(
    camera: Camera,
    target: XFormPrim | np.ndarray,
    distance: float = 0.4,
    elevation: float = 90.0,
    azimuth: float = 0.0,
) -> None:
    if isinstance(target, np.ndarray):
        target_position = target
    elif isinstance(target, XFormPrim):
        target_position, _ = target.get_world_pose()
    else:
        raise ValueError(f"Target must be a numpy array or XFormPrim: {type(target)}")
    elev_rad = math.radians(elevation)
    azim_rad = math.radians(azimuth)
    offset_x = distance * math.cos(elev_rad) * math.cos(azim_rad)
    offset_y = distance * math.cos(elev_rad) * math.sin(azim_rad)
    offset_z = distance * math.sin(elev_rad)
    camera_position = target_position + np.array([offset_x, offset_y, offset_z])
    rot = R.from_euler("xyz", [0, elevation, azimuth - 180], degrees=True)
    quaternion = rot.as_quat()
    quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    camera.set_world_pose(position=camera_position, orientation=quaternion)


def setup_camera(
    camera: Camera,
    camera_cfg: dict,
    only_depth_rep_for_camera: bool = False,
    only_color_rep_for_camera: bool = False,
) -> None:
    camera.initialize()

    # SimBox Style
    if "pixel_size" in camera_cfg:
        pixel_size = camera_cfg.get("pixel_size")  # Pixel size in microns
        if pixel_size is None:
            raise ValueError("Pixel size is not provided in SimBox style")
        f_number = camera_cfg.get("f_number")  # F-number
        if f_number is None:
            raise ValueError("F-number is not provided in SimBox style")
        focus_distance = camera_cfg.get("focus_distance")  # Focus distance in meters
        if focus_distance is None:
            raise ValueError("Focus distance is not provided in SimBox style")
        camera_params = camera_cfg.get("camera_params", None)
        if camera_params is not None:
            fx, fy, cx, cy = camera_params
        else:
            raise ValueError("Camera parameters are not provided in SimBox style")
        resolution = camera_cfg.get("resolution", None)
        if resolution is not None:
            width, height = resolution
        else:
            raise ValueError("Resolution is not provided in SimBox style")
        horizontal_aperture = pixel_size * 1e-3 * width
        vertical_aperture = pixel_size * 1e-3 * height
        focal_length_x = fx * pixel_size * 1e-3
        focal_length_y = fy * pixel_size * 1e-3
        focal_length = (focal_length_x + focal_length_y) / 2  # in mm

        # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
        camera.set_focal_length(focal_length / 10.0)
        camera.set_focus_distance(focus_distance)
        camera.set_lens_aperture(f_number * 100.0)
        camera.set_horizontal_aperture(horizontal_aperture / 10.0)
        camera.set_vertical_aperture(vertical_aperture / 10.0)
        camera.set_clipping_range(0.05, 1.0e5)
        camera.set_projection_type("pinhole")
        fx = width * camera.get_focal_length() / camera.get_horizontal_aperture()
        fy = height * camera.get_focal_length() / camera.get_vertical_aperture()
        camera.is_camera_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
        )
        camera.set_local_pose(
            translation=camera_cfg.get("position"),
            orientation=camera_cfg.get("orientation"),
            camera_axes=camera_cfg.get("camera_axes", "usd"),
        )
    # GenManip Style
    else:
        camera.set_focal_length(camera_cfg.get("focal_length", 4.5))
        camera.set_clipping_range(
            camera_cfg.get("clipping_range_min", 0.01),
            camera_cfg.get("clipping_range_max", 10000.0),
        )
        camera.set_vertical_aperture(camera_cfg.get("vertical_aperture", 5.625))
        camera.set_horizontal_aperture(camera_cfg.get("horizontal_aperture", 10.0))
        camera_params = camera_cfg.get("camera_params", None)
        if camera_params is not None:
            set_camera_rational_polynomial(camera, *camera_params)

    # add custom annotators
    if not only_color_rep_for_camera:
        if camera_cfg.get("with_distance", False):
            camera.add_distance_to_image_plane_to_frame()
        if not only_depth_rep_for_camera:
            if camera_cfg.get("with_semantic", False):
                camera.add_semantic_segmentation_to_frame()
            if camera_cfg.get("with_bbox2d", False):
                camera.add_bounding_box_2d_tight_to_frame()
                camera.add_bounding_box_2d_loose_to_frame()
            if camera_cfg.get("with_bbox3d", False):
                camera.add_bounding_box_3d_to_frame()
            if camera_cfg.get("with_motion_vector", False):
                camera.add_motion_vectors_to_frame()
