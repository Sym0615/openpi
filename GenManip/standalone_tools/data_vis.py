"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

"""
Professional Robotics Dataset Visualization Tool

This script provides comprehensive visualization capabilities for robotics datasets
stored in LMDB format, with support for various annotation types and both single
dataset and batch processing modes.

Features:
- Memory-efficient streaming processing for large datasets
- Selective data loading based on annotation requirements
- H.264 video encoding for optimal VSCode compatibility (with PyAV)
- Multi-threaded batch processing for multiple datasets
- Support for various annotation types (2D/3D bounding boxes, trajectories, etc.)

Usage Examples:

Single Dataset Mode:
    python data_vis.py \
        --data_path /path/to/single/dataset \
        --camera_name front_camera \
        --annotator_type st bb2d tcp2d \
        --output_path /path/to/output.mp4 \
        --output_format mp4 \
        --skip_existing

Batch Processing Mode:
    python data_vis.py \
        --batch_mode \
        --data_path /path/to/datasets/parent/folder \
        --camera_name front_camera \
        --annotator_type st bb2d tcp2d ins \
        --output_path /path/to/output/folder \
        --output_format mp4 \
        --num_workers 4 \
        --skip_existing

Annotation Types:
    st/step      - Frame step counter
    qp/qpos      - Joint positions  
    aa/arm_action - Arm action commands
    ep/ee_pose   - End-effector pose
    bb2d/bbox2d  - 2D bounding boxes
    bb3d/bbox3d  - 3D bounding boxes  
    tcp2d        - Current TCP positions
    atcp2d       - TCP trajectory (multi-frame)
    atcp2dd      - TCP trajectory with depth coloring
    sm/semantic_mask - Semantic segmentation overlay
    gp/grasp_point - Grasp point visualization
    ins/instruction - Task instruction text
    di/depth_image - Depth visualization
"""

from scipy.spatial.transform import Rotation as R
import lmdb
import pickle
import argparse
import cv2
import numpy as np
import numpy as np
from pathlib import Path
import os
import json
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial

# Optional dependency for better video encoding
try:
    import av

    HAS_AV = True
    print("PyAV detected - will use H.264 encoding for better VSCode compatibility")
except ImportError:
    HAS_AV = False
    print("PyAV not available - using OpenCV for video encoding")
    print("Install with: pip install av  # for better VSCode video compatibility")

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Constants for data processing
DEFAULT_RGB_SCALE_FACTOR = 256000.0
COLOR_MAP = []

# Generate random color map for visualization
for i in range(256):
    COLOR_MAP.append(
        (
            np.random.randint(0, 256),
            np.random.randint(0, 256),
            np.random.randint(0, 256),
        )
    )

# Thread-safe progress tracking
progress_lock = threading.Lock()
global_progress = {"completed": 0, "total": 0}


def parse_args():
    """Parse command line arguments for data visualization script.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - data_path: Path to the dataset directory (or parent directory in batch mode)
            - camera_name: Name of the camera sensor
            - annotator_type: List of annotation types to apply
            - output_path: Path for the output video/image file (or parent directory in batch mode)
            - output_format: Output video format (mp4 for VSCode compatibility, avi for wider support)
            - batch_mode: Enable batch processing of multiple datasets
            - num_workers: Number of worker threads for batch processing
            - skip_existing: Skip processing if output file already exists (useful for resuming interrupted batch jobs)
    """
    parser = argparse.ArgumentParser(
        description="Visualize robotics dataset with various annotation overlays"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the dataset directory (single mode) or parent directory containing multiple datasets (batch mode)",
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        required=True,
        help="Name of the camera sensor (e.g., 'front_camera', 'wrist_camera')",
    )
    parser.add_argument(
        "--annotator_type",
        nargs="+",
        type=str,
        required=False,
        default=[],
        help="List of annotation types: st/step, qp/qpos, aa/arm_action, ep/ee_pose, "
        "bb2d/bbox2d, bb3d/bbox3d, tcp2d, atcp2d, atcp2dd, sm/semantic_mask, "
        "gp/grasp_point, ins/instruction, di/depth_image",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the generated video/image file (single mode) or parent output directory (batch mode)",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["mp4", "avi"],
        default="mp4",
        help="Output video format (mp4 for VSCode compatibility, avi for wider support)",
    )
    parser.add_argument(
        "--batch_mode",
        action="store_true",
        help="Enable batch processing mode: process all subdirectories in data_path",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for batch processing (default: 4)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip processing if output file already exists (useful for resuming interrupted batch jobs)",
    )
    return parser.parse_args()


def visualize_3d_bbox(
    img,
    pixel_coordinates,
    point_color=(0, 0, 255),
    point_size=5,
    edge_color=(0, 255, 0),
    edge_thickness=1,
    draw_planes=False,
    plane_alpha=0.3,
):
    """Visualize 3D bounding box on 2D image.

    Renders a 3D bounding box by drawing corners, edges, and optionally filled planes
    on the input image using the projected 2D pixel coordinates.

    Args:
        img (np.ndarray): Input image to draw on
        pixel_coordinates (np.ndarray): 8x2 array of 2D pixel coordinates for bbox corners
        point_color (tuple): RGB color for corner points
        point_size (int): Radius of corner point circles
        edge_color (tuple): RGB color for bounding box edges
        edge_thickness (int): Thickness of edge lines
        draw_planes (bool): Whether to draw filled plane surfaces
        plane_alpha (float): Alpha transparency for plane surfaces (0.0-1.0)

    Returns:
        np.ndarray: Image with 3D bounding box visualization overlay
    """
    # Draw corner points
    for pixel_coordinate in pixel_coordinates:
        img = cv2.circle(
            img,
            (int(pixel_coordinate[0]), int(pixel_coordinate[1])),
            point_size,
            point_color,
            -1,
        )

    # Define edges connecting bbox corners (standard cuboid topology)
    edges = [
        (0, 1),
        (1, 3),
        (3, 2),
        (2, 0),  # Bottom face
        (4, 5),
        (5, 7),
        (7, 6),
        (6, 4),  # Top face
        (0, 4),
        (1, 5),
        (3, 7),
        (2, 6),  # Vertical edges
    ]

    # Draw edges
    for edge in edges:
        start_point = (
            int(pixel_coordinates[edge[0]][0]),
            int(pixel_coordinates[edge[0]][1]),
        )
        end_point = (
            int(pixel_coordinates[edge[1]][0]),
            int(pixel_coordinates[edge[1]][1]),
        )
        img = cv2.line(img, start_point, end_point, edge_color, edge_thickness)

    # Optionally draw filled planes for better 3D visualization
    if draw_planes:
        planes = [
            {"corners": [0, 1, 3, 2], "color": edge_color},  # Bottom
            {"corners": [4, 5, 7, 6], "color": edge_color},  # Top
            {"corners": [0, 2, 6, 4], "color": edge_color},  # Left
            {"corners": [1, 3, 7, 5], "color": edge_color},  # Right
            {"corners": [2, 3, 7, 6], "color": edge_color},  # Back
            {"corners": [0, 1, 5, 4], "color": edge_color},  # Front
        ]
        for plane in planes:
            contour = np.array(
                [
                    [int(pixel_coordinates[i][0]), int(pixel_coordinates[i][1])]
                    for i in plane["corners"]
                ],
                dtype=np.int32,
            )
            overlay = img.copy()
            cv2.fillPoly(overlay, [contour], plane["color"])
            img = cv2.addWeighted(overlay, plane_alpha, img, 1 - plane_alpha, 0)
    return img


def transform_to_pose(transform):
    """Convert transformation matrix to pose representation.

    Extracts translation and quaternion from a 4x4 homogeneous transformation matrix.

    Args:
        transform (np.ndarray): 4x4 transformation matrix

    Returns:
        tuple: (translation, quaternion) where:
            - translation (np.ndarray): 3D translation vector
            - quaternion (np.ndarray): Quaternion in [w,x,y,z] format
    """
    trans = transform[:3, 3]
    quat = R.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return trans, quat


def get_scalar_data_from_lmdb(data_path, key):
    """Retrieve scalar data from LMDB database.

    Loads and deserializes scalar data (e.g., joint positions, actions) from
    the LMDB storage format used in robotics datasets.

    Args:
        data_path (str): Path to dataset directory containing LMDB files
        key (bytes): LMDB key for the desired data

    Returns:
        Any: Deserialized data object (typically numpy arrays or lists)
    """
    meta_info = pickle.load(open(f"{data_path}/meta_info.pkl", "rb"))
    lmdb_env = lmdb.open(
        f"{data_path}/lmdb", readonly=True, lock=False, readahead=False, meminit=False
    )
    key_index = meta_info["keys"]["scalar_data"].index(key)
    key_key = meta_info["keys"]["scalar_data"][key_index]
    with lmdb_env.begin(write=False) as txn:
        data = pickle.loads(txn.get(key_key))
    return data


def get_json_data_from_lmdb(data_path):
    """Retrieve JSON metadata from LMDB database.

    Loads camera parameters and other metadata stored as JSON in the dataset.

    Args:
        data_path (str): Path to dataset directory containing LMDB files

    Returns:
        dict: Deserialized JSON data containing camera parameters and metadata
    """
    lmdb_env = lmdb.open(
        f"{data_path}/lmdb", readonly=True, lock=False, readahead=False, meminit=False
    )
    with lmdb_env.begin(write=False) as txn:
        data = pickle.loads(txn.get(b"json_data"))
    return data


def get_color_image_from_lmdb(data_path, key):
    """Load all color images from LMDB database (deprecated - memory intensive).

    Warning: This function loads all images into memory at once and should be
    avoided for large datasets. Use get_frame_data() instead.

    Args:
        data_path (str): Path to dataset directory
        key (str): Key pattern for color images

    Returns:
        list: List of color images as numpy arrays
    """
    meta_info = pickle.load(open(f"{data_path}/meta_info.pkl", "rb"))
    num_steps = meta_info["num_steps"]
    lmdb_env = lmdb.open(
        f"{data_path}/lmdb", readonly=True, lock=False, readahead=False, meminit=False
    )
    key_index = meta_info["keys"][key]
    color_image = []
    with lmdb_env.begin(write=False) as txn:
        for key in key_index:
            color_image.append(
                cv2.imdecode(pickle.loads(txn.get(key)), cv2.IMREAD_COLOR)
            )
    return color_image


def get_semantic_image_from_lmdb(data_path, key):
    """Load all semantic segmentation masks from LMDB database (deprecated).

    Warning: Memory intensive - use get_frame_data() for streaming access.

    Args:
        data_path (str): Path to dataset directory
        key (str): Key pattern for semantic masks

    Returns:
        list: List of grayscale semantic masks as numpy arrays
    """
    meta_info = pickle.load(open(f"{data_path}/meta_info.pkl", "rb"))
    num_steps = meta_info["num_steps"]
    lmdb_env = lmdb.open(
        f"{data_path}/lmdb", readonly=True, lock=False, readahead=False, meminit=False
    )
    key_index = meta_info["keys"][key]
    semantic_image = []
    with lmdb_env.begin(write=False) as txn:
        for key in key_index:
            semantic_image.append(
                cv2.imdecode(pickle.loads(txn.get(key)), cv2.IMREAD_GRAYSCALE)
            )
    return semantic_image


def uint16_array_to_float_array(uint16_array: np.ndarray) -> np.ndarray:
    """Convert uint16 depth values to float depth in meters.

    Converts depth image from uint16 format (scaled by 10000) back to
    floating point depth values in meters.

    Args:
        uint16_array (np.ndarray): Depth image in uint16 format

    Returns:
        np.ndarray: Depth image with float32 values in meters
    """
    float_array = uint16_array.astype(np.float32)
    float_array = float_array / 10000
    return float_array


def get_depth_image_from_lmdb(data_path, key):
    """Load all depth images from LMDB database (deprecated - memory intensive).

    Warning: This function loads all depth images into memory at once.
    Use get_frame_data() for memory-efficient streaming access.

    Args:
        data_path (str): Path to dataset directory
        key (str): Key pattern for depth images

    Returns:
        list: List of depth images as float32 numpy arrays (values in meters)
    """
    meta_info = pickle.load(open(f"{data_path}/meta_info.pkl", "rb"))
    num_steps = meta_info["num_steps"]
    lmdb_env = lmdb.open(
        f"{data_path}/lmdb", readonly=True, lock=False, readahead=False, meminit=False
    )
    key_index = meta_info["keys"][key]
    depth_image = []
    with lmdb_env.begin(write=False) as txn:
        for key in key_index:
            depth_image.append(
                uint16_array_to_float_array(
                    cv2.imdecode(pickle.loads(txn.get(key)), cv2.IMREAD_UNCHANGED)
                )
            )
    return depth_image


def get_meta_data(data_path, camera_name, annotator_type):
    """Selectively load metadata based on required annotation types.

    Optimized loading function that only retrieves data needed for the specified
    annotation types, significantly reducing memory usage for large datasets.

    Args:
        data_path (str): Path to dataset directory containing LMDB files
        camera_name (str): Name of the camera sensor to process
        annotator_type (list): List of annotation type strings to enable

    Returns:
        dict: Dictionary containing only the metadata required for specified annotations:
            - num_frames: Total number of frames in dataset
            - Additional fields based on annotator_type requirements
    """
    data = {}

    # Always load basic metadata
    meta_info = pickle.load(open(f"{data_path}/meta_info.pkl", "rb"))
    data["num_frames"] = meta_info["num_steps"]

    # Conditionally load data based on annotation requirements
    if "gp" in annotator_type:
        try:
            grasp_point = meta_info["task_data"]["grasp_point"]["0"][camera_name]
            data["grasp_point"] = grasp_point
        except (KeyError, TypeError, IndexError):
            print(f"Missing grasp_point for {camera_name}; continuing")
            data["grasp_point"] = None

    if "ins" in annotator_type:
        data["instruction"] = meta_info["task_data"]["instruction"]

    if "aa" in annotator_type:
        data["arm_action"] = get_scalar_data_from_lmdb(data_path, b"arm_action")

    if "qp" in annotator_type:
        data["qpos"] = get_scalar_data_from_lmdb(data_path, b"observation/robot/qpos")

    if "ep" in annotator_type:
        data["ee_pose"] = get_scalar_data_from_lmdb(
            data_path, b"observation/robot/ee_pose_state"
        )

    if "bb3d" in annotator_type:
        data["bounding_box_3d"] = get_scalar_data_from_lmdb(
            data_path, f"observation/{camera_name}/bbox3d".encode("utf-8")
        )
        data["bbox3d_id2labels"] = get_scalar_data_from_lmdb(
            data_path, f"observation/{camera_name}/bbox3d_id2labels".encode("utf-8")
        )
        # 3D bbox visualization requires camera parameters
        data["camera_intrinsic"] = get_json_data_from_lmdb(data_path)[
            f"observation/{camera_name}/camera_params"
        ]
        data["camera2env_pose"] = get_scalar_data_from_lmdb(
            data_path, f"observation/{camera_name}/camera2env_pose".encode("utf-8")
        )
        for i in range(len(data["camera2env_pose"])):
            data["camera2env_pose"][i] = transform_to_pose(data["camera2env_pose"][i])

    if any(x in annotator_type for x in ["tcp2d", "atcp2d", "atcp2dd"]):
        data["tcp_trace_2d"] = get_scalar_data_from_lmdb(
            data_path, f"observation/{camera_name}/tcp_2d_trace".encode("utf-8")
        )

    if "bb2d" in annotator_type:
        data["bounding_box_2d"] = get_scalar_data_from_lmdb(
            data_path, f"observation/{camera_name}/bbox2d_loose".encode("utf-8")
        )
        data["bounding_box_2d_id2labels"] = get_scalar_data_from_lmdb(
            data_path,
            f"observation/{camera_name}/bbox2d_loose_id2labels".encode("utf-8"),
        )
        data["bbox2d_tight_id2labels"] = get_scalar_data_from_lmdb(
            data_path,
            f"observation/{camera_name}/bbox2d_tight_id2labels".encode("utf-8"),
        )
        data["bbox2d_loose_id2labels"] = get_scalar_data_from_lmdb(
            data_path,
            f"observation/{camera_name}/bbox2d_loose_id2labels".encode("utf-8"),
        )

    if "sm" in annotator_type:
        data["semantic_mask_id2labels"] = get_scalar_data_from_lmdb(
            data_path,
            f"observation/{camera_name}/semantic_mask_id2labels".encode("utf-8"),
        )

    # Load depth statistics for depth-dependent annotations
    if any(x in annotator_type for x in ["di", "atcp2dd"]):
        try:
            all_depth_images = get_depth_image_from_lmdb(
                data_path, f"observation/{camera_name}/depth_image"
            )
            data["depth_range"] = [
                np.min(np.array(all_depth_images)),
                np.max(np.array(all_depth_images)),
            ]
            # Immediately release memory
            del all_depth_images
        except (FileNotFoundError, KeyError, ValueError, TypeError):
            print(f"No depth image found for {camera_name}; skipping depth_range")
            data["depth_range"] = None

    # Log loaded data types for debugging
    loaded_data_types = list(data.keys())
    print(f"Loaded metadata types: {loaded_data_types}")

    return data


def get_frame_data(data_path, camera_name, frame_idx, annotator_type):
    """Load single frame image data based on annotation requirements.

    Memory-efficient function that loads only the image data needed for the
    specified annotation types at the requested frame index.

    Args:
        data_path (str): Path to dataset directory
        camera_name (str): Name of the camera sensor
        frame_idx (int): Zero-based frame index to load
        annotator_type (list): List of annotation types to enable

    Returns:
        dict: Dictionary containing frame image data:
            - color_image: RGB color image (always loaded)
            - depth_image: Depth image in meters (if depth annotations enabled)
            - semantic_image: Semantic segmentation mask (if semantic annotations enabled)
    """
    meta_info = pickle.load(open(f"{data_path}/meta_info.pkl", "rb"))
    lmdb_env = lmdb.open(
        f"{data_path}/lmdb", readonly=True, lock=False, readahead=False, meminit=False
    )

    frame_data = {}

    # Color image is always required as the base canvas
    color_key_index = meta_info["keys"][f"observation/{camera_name}/color_image"]
    with lmdb_env.begin(write=False) as txn:
        frame_data["color_image"] = cv2.imdecode(
            pickle.loads(txn.get(color_key_index[frame_idx])), cv2.IMREAD_COLOR
        )

    # Conditionally load depth data for depth-dependent annotations
    if any(x in annotator_type for x in ["di", "atcp2dd"]):
        depth_key_index = meta_info["keys"][f"observation/{camera_name}/depth_image"]
        with lmdb_env.begin(write=False) as txn:
            frame_data["depth_image"] = uint16_array_to_float_array(
                cv2.imdecode(
                    pickle.loads(txn.get(depth_key_index[frame_idx])),
                    cv2.IMREAD_UNCHANGED,
                )
            )

    # Conditionally load semantic segmentation masks
    if "sm" in annotator_type:
        semantic_key_index = meta_info["keys"][
            f"observation/{camera_name}/semantic_mask"
        ]
        with lmdb_env.begin(write=False) as txn:
            frame_data["semantic_image"] = cv2.imdecode(
                pickle.loads(txn.get(semantic_key_index[frame_idx])),
                cv2.IMREAD_GRAYSCALE,
            )

    lmdb_env.close()
    return frame_data


def annotate_depth_image(data, frame_data, i):
    """Render depth image as colorized visualization.

    Converts depth values to a false-color representation using the JET colormap,
    with depth range normalized based on dataset statistics.

    Args:
        data (dict): Dataset metadata containing depth_range
        frame_data (dict): Current frame data with depth_image
        i (int): Current frame index (unused in this function)

    Returns:
        np.ndarray: Colorized depth image
    """
    depth_image = frame_data["depth_image"]
    depth_image = (depth_image - data["depth_range"][0]) / (
        data["depth_range"][1] - data["depth_range"][0]
    )
    depth_image = (depth_image * 255).astype(np.uint8)
    frame_data["color_image"] = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
    return frame_data["color_image"]


def annotate_grasp_point(data, frame_data, i):
    """Visualize grasp points on the image.

    Draws colored circles at the grasp point locations for robotic grasping tasks.
    Red circle for first grasp point, green circle for second grasp point.

    Args:
        data (dict): Dataset metadata containing grasp_point coordinates
        frame_data (dict): Current frame data with color_image
        i (int): Current frame index (unused in this function)

    Returns:
        np.ndarray: Image with grasp point annotations
    """
    frame_data["color_image"] = cv2.circle(
        frame_data["color_image"],
        (int(data["grasp_point"][0][0][0]), int(data["grasp_point"][0][0][1])),
        5,
        (0, 0, 255),  # Red circle
        -1,
    )
    frame_data["color_image"] = cv2.circle(
        frame_data["color_image"],
        (int(data["grasp_point"][1][0][0]), int(data["grasp_point"][1][0][1])),
        5,
        (0, 255, 0),  # Green circle
        -1,
    )
    return frame_data["color_image"]


def annotate_semantic_mask(data, frame_data, i):
    """Overlay semantic segmentation masks with color coding.

    Renders semantic segmentation masks as colored overlays on the original image,
    with each object class assigned a unique color from the global COLOR_MAP.
    Ground plane objects are excluded from visualization.

    Args:
        data (dict): Dataset metadata containing semantic_mask_id2labels
        frame_data (dict): Current frame data with semantic_image and color_image
        i (int): Current frame index for accessing frame-specific labels

    Returns:
        np.ndarray: Image with semantic segmentation overlay
    """
    semantic_mask = frame_data["semantic_image"]
    colored_mask = np.zeros_like(frame_data["color_image"])
    unique_labels = np.unique(semantic_mask)

    for label in unique_labels:
        if label == 0:  # Skip background
            continue
        if str(label) in data["semantic_mask_id2labels"][i]:
            if (
                data["semantic_mask_id2labels"][i][str(label)]["class"]
                == "defaultgroundplane"
            ):
                continue  # Skip ground plane
            mask = semantic_mask == label
            colored_mask[mask] = COLOR_MAP[label % len(COLOR_MAP)]

    frame_data["color_image"] = cv2.cvtColor(
        frame_data["color_image"], cv2.COLOR_RGB2BGR
    )
    frame_data["color_image"] = cv2.addWeighted(
        frame_data["color_image"], 0.7, colored_mask, 0.3, 0
    )
    return frame_data["color_image"]


def image_to_float_array(
    image: np.ndarray, scale_factor: float | None = None
) -> np.ndarray:
    """Convert RGB image to scaled float representation.

    Utility function for converting RGB images to float arrays with custom scaling.

    Args:
        image (np.ndarray): Input RGB image
        scale_factor (float, optional): Scaling factor, defaults to DEFAULT_RGB_SCALE_FACTOR

    Returns:
        np.ndarray: Scaled float array representation of the image
    """
    image_array = np.asarray(image)
    if scale_factor is None:
        scale_factor = DEFAULT_RGB_SCALE_FACTOR
    float_array = np.dot(image_array, [65536, 256, 1])
    return float_array / scale_factor


def annotate_all_tcp_trace_2d_depth(data, frame_data, i):
    """Visualize TCP (Tool Center Point) trajectory with depth-based coloring.

    Draws the robot's TCP trajectory over multiple future frames, with line colors
    modulated by depth values for 3D visualization effect. Lines become darker
    with increasing depth.

    Args:
        data (dict): Dataset metadata containing tcp_trace_2d and depth_range
        frame_data (dict): Current frame data with depth_image and color_image
        i (int): Current frame index for trajectory starting point

    Returns:
        np.ndarray: Image with depth-colored TCP trajectory overlay
    """
    for j in range(i, min(i + 20, len(data["tcp_trace_2d"]) - 1)):
        for k in range(len(data["tcp_trace_2d"][j])):
            x1 = int(data["tcp_trace_2d"][j][k][0])
            y1 = int(data["tcp_trace_2d"][j][k][1])
            depth_value1 = frame_data["depth_image"][y1, x1]  # Depth in meters
            x2 = int(data["tcp_trace_2d"][j + 1][k][0])
            y2 = int(data["tcp_trace_2d"][j + 1][k][1])
            depth_value2 = frame_data["depth_image"][y2, x2]  # Depth in meters
            depth_value = (depth_value1 + depth_value2) / 2

            base_color = COLOR_MAP[-k]
            scale_factor = 2
            # Modulate color intensity based on depth
            darker_color = tuple(
                int(
                    c
                    * (
                        1.0
                        - scale_factor
                        * (depth_value - data["depth_range"][0])
                        / (data["depth_range"][1] - data["depth_range"][0])
                    )
                )
                for c in base_color
            )
            frame_data["color_image"] = cv2.line(
                frame_data["color_image"],
                (x1, y1),
                (x2, y2),
                darker_color,
                2,
            )
    return frame_data["color_image"]


def annotate_all_tcp_trace_2d(data, frame_data, i):
    """Visualize TCP (Tool Center Point) trajectory over multiple frames.

    Draws the robot's TCP trajectory for the next 20 frames or until the end
    of the dataset, with different colors for each TCP component.

    Args:
        data (dict): Dataset metadata containing tcp_trace_2d
        frame_data (dict): Current frame data with color_image
        i (int): Current frame index for trajectory starting point

    Returns:
        np.ndarray: Image with TCP trajectory overlay
    """
    for j in range(i, min(i + 20, len(data["tcp_trace_2d"]) - 1)):
        for k in range(len(data["tcp_trace_2d"][j])):
            frame_data["color_image"] = cv2.line(
                frame_data["color_image"],
                (
                    int(data["tcp_trace_2d"][j][k][0]),
                    int(data["tcp_trace_2d"][j][k][1]),
                ),
                (
                    int(data["tcp_trace_2d"][j + 1][k][0]),
                    int(data["tcp_trace_2d"][j + 1][k][1]),
                ),
                COLOR_MAP[-k],
                2,
            )
    return frame_data["color_image"]


def annotate_tcp_trace_2d(data, frame_data, i):
    """Visualize current TCP (Tool Center Point) positions.

    Draws colored circles at the current TCP positions for each component
    of the robot's tool center point.

    Args:
        data (dict): Dataset metadata containing tcp_trace_2d
        frame_data (dict): Current frame data with color_image
        i (int): Current frame index

    Returns:
        np.ndarray: Image with TCP position markers
    """
    for idx, trace in enumerate(data["tcp_trace_2d"][i]):
        frame_data["color_image"] = cv2.circle(
            frame_data["color_image"],
            (int(trace[0]), int(trace[1])),
            3,
            COLOR_MAP[-idx],
            -1,
        )
    return frame_data["color_image"]


def annotate_qpos(data, frame_data, i):
    """Display robot joint positions as text overlay.

    Renders the current joint position values (qpos) as text on the image,
    with values rounded to 2 decimal places for readability.

    Args:
        data (dict): Dataset metadata containing qpos values
        frame_data (dict): Current frame data with color_image
        i (int): Current frame index

    Returns:
        np.ndarray: Image with joint position text overlay
    """
    cv2.putText(
        frame_data["color_image"],
        f"qpos: {[round(x, 2) for x in data['qpos'][i]]}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255, 255, 255),
        1,
    )
    return frame_data["color_image"]


def annotate_arm_action(data, frame_data, i):
    """Display robot arm action commands as text overlay.

    Renders the current arm action values as text on the image,
    with values rounded to 2 decimal places for readability.

    Args:
        data (dict): Dataset metadata containing arm_action values
        frame_data (dict): Current frame data with color_image
        i (int): Current frame index

    Returns:
        np.ndarray: Image with arm action text overlay
    """
    cv2.putText(
        frame_data["color_image"],
        f"arm_action: {[round(x, 2) for x in data['arm_action'][i]]}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255, 255, 255),
        1,
    )
    return frame_data["color_image"]


def annotate_ee_pose(data, frame_data, i):
    """Display end-effector pose as text overlay.

    Renders the current end-effector pose (position and orientation) as text,
    with values rounded to 2 decimal places for readability.

    Args:
        data (dict): Dataset metadata containing ee_pose values
        frame_data (dict): Current frame data with color_image
        i (int): Current frame index

    Returns:
        np.ndarray: Image with end-effector pose text overlay
    """
    cv2.putText(
        frame_data["color_image"],
        f"ee_pose: {[round(x, 2) for x in data['ee_pose'][i][0]]}, {[round(x, 2) for x in data['ee_pose'][i][1]]}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255, 255, 255),
        1,
    )
    return frame_data["color_image"]


def pose_to_transform(pose):
    """Convert pose representation to transformation matrix.

    Converts a pose tuple (translation, quaternion) back to a 4x4
    homogeneous transformation matrix.

    Args:
        pose (tuple): (translation, quaternion) where quaternion is [w,x,y,z]

    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    trans, quat = pose
    transform = np.eye(4)
    transform[:3, 3] = trans
    transform[:3, :3] = R.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    return transform


def get_world_point_from_pixel(point_2d, depth, intrinsic, translation, quaternion):
    """Project 2D pixel coordinates to 3D world coordinates.

    Performs inverse camera projection to convert 2D pixel coordinates and
    corresponding depth values to 3D world coordinates using camera parameters.

    Args:
        point_2d (array-like): 2D pixel coordinates [x, y]
        depth (np.ndarray): Depth image for depth lookup
        intrinsic (np.ndarray): 3x3 camera intrinsic matrix
        translation (np.ndarray): Camera translation in world coordinates
        quaternion (np.ndarray): Camera orientation quaternion [w,x,y,z]

    Returns:
        np.ndarray: 3D world coordinates [x, y, z]
    """
    intrinsic = np.array(intrinsic)
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    x, y = point_2d[0], point_2d[1]
    Z = depth[int(y)][int(x)]
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy

    point_in_camera_frame = np.array([X, Y, Z])
    # Apply coordinate system transformation
    add_rotation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    point_in_camera_frame = add_rotation @ point_in_camera_frame

    camera_to_world = pose_to_transform((translation, quaternion))
    point_in_world_frame = camera_to_world @ np.array([*point_in_camera_frame, 1])
    point3d = point_in_world_frame[:3]
    return point3d


def get_pixel_from_world_point(intrinsic, world_point, translation, quaternion):
    """Project 3D world coordinates to 2D pixel coordinates.

    Performs camera projection to convert 3D world coordinates to 2D pixel
    coordinates using camera intrinsic parameters and pose.

    Args:
        intrinsic (np.ndarray): 3x3 camera intrinsic matrix
        world_point (np.ndarray): 3D world coordinates (Nx3 or 3D point)
        translation (np.ndarray): Camera translation in world coordinates
        quaternion (np.ndarray): Camera orientation quaternion [w,x,y,z]

    Returns:
        np.ndarray: 2D pixel coordinates (Nx2 array)
    """
    world_point = world_point.reshape(-1, 3)
    camera_to_world = pose_to_transform((translation, quaternion))
    world_to_camera = np.linalg.inv(camera_to_world)

    homogeneous_points = np.hstack([world_point, np.ones((world_point.shape[0], 1))])
    points_in_camera_frame = np.dot(homogeneous_points, world_to_camera.T)
    point_in_camera_frame = points_in_camera_frame[:, :3]

    # Apply inverse coordinate system transformation
    add_rotation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    inv_rotation = np.linalg.inv(add_rotation)
    rotated_point = np.dot(point_in_camera_frame, inv_rotation.T)

    intrinsic = np.array(intrinsic)
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    X, Y, Z = rotated_point[:, 0], rotated_point[:, 1], rotated_point[:, 2]
    x = (fx * X / Z) + cx
    y = (fy * Y / Z) + cy
    return np.column_stack((x, y))


def annotate_bounding_box_3d(data, frame_data, i):
    """Visualize 3D bounding boxes projected onto the image.

    Projects 3D bounding box corners to 2D image coordinates and renders
    them using the visualize_3d_bbox function with color coding per object class.

    Args:
        data (dict): Dataset metadata containing bounding_box_3d, camera parameters
        frame_data (dict): Current frame data with color_image
        i (int): Current frame index

    Returns:
        np.ndarray: Image with 3D bounding box overlays
    """
    pixel_coordinates = []
    for box in data["bounding_box_3d"][i]:
        pixel_coordinates.append(box["corners"])
    pixel_coordinates = np.array(pixel_coordinates)
    pixel_coordinates = get_pixel_from_world_point(
        data["camera_intrinsic"], pixel_coordinates, *data["camera2env_pose"][i]
    )

    for bbox in data["bounding_box_3d"][i]:
        pixel_coordinates = get_pixel_from_world_point(
            data["camera_intrinsic"], bbox["corners"], *data["camera2env_pose"][i]
        )
        frame_data["color_image"] = visualize_3d_bbox(
            frame_data["color_image"],
            pixel_coordinates,
            edge_color=COLOR_MAP[bbox["class"]],
            draw_planes=True,
            point_size=1,
        )
    return frame_data["color_image"]


def annotate_step(data, frame_data, i):
    """Display current frame step number as text overlay.

    Shows the current frame index out of total frames in a formatted string,
    with zero-padding for consistent display width.

    Args:
        data (dict): Dataset metadata containing num_frames
        frame_data (dict): Current frame data with color_image
        i (int): Current frame index

    Returns:
        np.ndarray: Image with step counter text overlay
    """
    length = data["num_frames"]
    cv2.putText(
        frame_data["color_image"],
        f"step: {str(i).zfill(len(str(length)))}/{length}",
        (10, 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (255, 255, 255),
        1,
    )
    return frame_data["color_image"]


def annotate_bbox2d(data, frame_data, i):
    """Visualize 2D bounding boxes with colored overlays.

    Draws 2D bounding boxes as colored rectangles with semi-transparent fill
    and solid outline. Ground plane objects are excluded from visualization.

    Args:
        data (dict): Dataset metadata containing bounding_box_2d and labels
        frame_data (dict): Current frame data with color_image
        i (int): Current frame index

    Returns:
        np.ndarray: Image with 2D bounding box overlays
    """
    for bbox in data["bounding_box_2d"][i]:
        if (
            data["bounding_box_2d_id2labels"][i][str(bbox[0])]["class"]
            == "defaultgroundplane"
        ):
            continue

        overlay = np.zeros_like(frame_data["color_image"])
        overlay = cv2.rectangle(
            overlay,
            (bbox[1], bbox[2]),
            (bbox[3], bbox[4]),
            COLOR_MAP[bbox[0]],
            -1,
        )
        alpha = 0.3
        frame_data["color_image"] = cv2.addWeighted(
            frame_data["color_image"], 1, overlay, alpha, 0
        )
        frame_data["color_image"] = cv2.rectangle(
            frame_data["color_image"],
            (bbox[1], bbox[2]),
            (bbox[3], bbox[4]),
            COLOR_MAP[bbox[0]],
            1,
        )
    return frame_data["color_image"]


def annotate_bbox2d_id2labels(data, frame_data, i):
    """Display 2D bounding box class labels as text.

    Renders object class names as text overlays positioned at the top-left
    corners of their corresponding 2D bounding boxes.

    Args:
        data (dict): Dataset metadata containing bounding_box_2d and id2labels
        frame_data (dict): Current frame data with color_image
        i (int): Current frame index

    Returns:
        np.ndarray: Image with bounding box class label text overlays
    """
    for bbox in data["bounding_box_2d"][i]:
        if (
            data["bounding_box_2d_id2labels"][i][str(bbox[0])]["class"]
            == "defaultgroundplane"
        ):
            continue
        frame_data["color_image"] = cv2.putText(
            frame_data["color_image"],
            f"class: {data['bounding_box_2d_id2labels'][i][str(bbox[0])]['class'][:8]}",
            (bbox[1], bbox[2]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 0, 255),
            1,
        )
    return frame_data["color_image"]


def annotate_instruction(data, frame_data, i):
    """Display task instruction as text overlay at bottom of image.

    Renders the natural language instruction for the current task with a
    white background rectangle for improved readability.

    Args:
        data (dict): Dataset metadata containing instruction text
        frame_data (dict): Current frame data with color_image
        i (int): Current frame index (unused in this function)

    Returns:
        np.ndarray: Image with instruction text overlay
    """
    # Create white background rectangle for text readability
    cv2.rectangle(
        frame_data["color_image"],
        (10, frame_data["color_image"].shape[0] - 20),
        (
            frame_data["color_image"].shape[1] - 10,
            frame_data["color_image"].shape[0] - 5,
        ),
        (255, 255, 255),
        -1,
    )
    cv2.putText(
        frame_data["color_image"],
        data["instruction"],
        (10, frame_data["color_image"].shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (0, 0, 0),
        1,
    )
    return frame_data["color_image"]


def annotate_all_text(data, frame_data, i, args):
    """Apply all enabled text-based annotations to the frame.

    Conditionally applies various text overlays based on the annotation types
    specified in the command line arguments.

    Args:
        data (dict): Dataset metadata
        frame_data (dict): Current frame data with color_image
        i (int): Current frame index
        args (argparse.Namespace): Command line arguments with annotator_type

    Returns:
        np.ndarray: Image with all enabled text annotations
    """
    if "st" in args.annotator_type:
        annotate_step(data, frame_data, i)
    if "qp" in args.annotator_type:
        annotate_qpos(data, frame_data, i)
    if "aa" in args.annotator_type:
        annotate_arm_action(data, frame_data, i)
    if "ep" in args.annotator_type:
        annotate_ee_pose(data, frame_data, i)
    if "bb2d" in args.annotator_type:
        annotate_bbox2d_id2labels(data, frame_data, i)
    if "ins" in args.annotator_type:
        annotate_instruction(data, frame_data, i)
    return frame_data["color_image"]


def find_dataset_directories(parent_path):
    """Find all subdirectories containing valid LMDB datasets.

    Scans the parent directory for subdirectories that contain both
    'lmdb' folder and 'meta_info.pkl' file, indicating valid datasets.

    Args:
        parent_path (str): Path to parent directory containing multiple datasets

    Returns:
        list: List of valid dataset directory paths
    """
    dataset_dirs = []
    parent_path = Path(parent_path)

    if not parent_path.exists():
        raise FileNotFoundError(f"Parent directory not found: {parent_path}")

    for item in parent_path.iterdir():
        if item.is_dir():
            # Check if this directory contains required dataset files
            lmdb_path = item / "lmdb"
            meta_path = item / "meta_info.pkl"

            if lmdb_path.exists() and meta_path.exists():
                dataset_dirs.append(str(item))
                print(f"Found dataset: {item.name}")

    return sorted(dataset_dirs)


def process_single_dataset(
    data_path,
    camera_name,
    annotator_type,
    output_path,
    output_format,
    worker_id=0,
    skip_existing=False,
):
    """Process a single dataset and generate visualization video.

    This function encapsulates the entire processing pipeline for a single dataset,
    allowing it to be called from multiple threads in batch mode.

    Args:
        data_path (str): Path to the dataset directory
        camera_name (str): Name of the camera sensor
        annotator_type (list): List of annotation types to enable
        output_path (str): Output video file path
        output_format (str): Output video format ('mp4' or 'avi')
        worker_id (int): Worker thread ID for logging
        skip_existing (bool): Skip processing if output file already exists

    Returns:
        tuple: (success: bool, dataset_name: str, output_path: str, error_msg: str)
    """
    dataset_name = os.path.basename(data_path)

    try:
        # Check if output file already exists and skip_existing is enabled
        if skip_existing and os.path.exists(output_path):
            print(
                f"[Worker {worker_id}] {dataset_name}: Output file already exists, skipping..."
            )
            return True, dataset_name, output_path, "Skipped (file exists)"

        print(f"[Worker {worker_id}] Processing dataset: {dataset_name}")

        # Load metadata with selective loading based on annotation types
        data = get_meta_data(data_path, camera_name, annotator_type)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Single frame processing (PNG output)
        if data["num_frames"] == 1:
            print(
                f"[Worker {worker_id}] {dataset_name}: Single frame detected, processing as image..."
            )

            # For single frame, check PNG output path if skip_existing is enabled
            png_output = os.path.splitext(output_path)[0] + ".png"
            if skip_existing and os.path.exists(png_output):
                print(
                    f"[Worker {worker_id}] {dataset_name}: PNG output already exists, skipping..."
                )
                return True, dataset_name, png_output, "Skipped (PNG exists)"

            i = 0
            frame_data = get_frame_data(data_path, camera_name, i, annotator_type)
            frame_data["color_image"] = cv2.cvtColor(
                frame_data["color_image"], cv2.COLOR_RGB2BGR
            )

            # Apply all enabled annotations
            _apply_annotations(data, frame_data, i, annotator_type)

            # Save as PNG for single frame
            cv2.imwrite(png_output, frame_data["color_image"])
            print(
                f"[Worker {worker_id}] {dataset_name}: Single frame saved to {png_output}"
            )
            return True, dataset_name, png_output, ""

        # Multi-frame video processing
        print(
            f"[Worker {worker_id}] {dataset_name}: Processing {data['num_frames']} frames..."
        )

        # Get video dimensions from first frame
        first_frame_data = get_frame_data(data_path, camera_name, 0, annotator_type)
        height, width = first_frame_data["color_image"].shape[:2]

        # Configure output path with correct extension
        if output_format == "mp4" and not output_path.endswith(".mp4"):
            output_path = os.path.splitext(output_path)[0] + ".mp4"
        elif output_format == "avi" and not output_path.endswith(".avi"):
            output_path = os.path.splitext(output_path)[0] + ".avi"

        success = _process_video_frames(
            data,
            data_path,
            camera_name,
            annotator_type,
            output_path,
            output_format,
            width,
            height,
            worker_id,
            dataset_name,
        )

        if success:
            print(f"[Worker {worker_id}] {dataset_name}: Video saved to {output_path}")
            return True, dataset_name, output_path, ""
        else:
            return False, dataset_name, output_path, "Video processing failed"

    except (
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as e:
        error_msg = f"Error processing {dataset_name}: {str(e)}"
        print(f"[Worker {worker_id}] {error_msg}")
        return False, dataset_name, output_path, error_msg


def _apply_annotations(data, frame_data, i, annotator_type):
    """Apply all enabled annotations to a single frame."""
    if "di" in annotator_type:
        annotate_depth_image(data, frame_data, i)
    if "bb2d" in annotator_type:
        annotate_bbox2d(data, frame_data, i)
    if "bb3d" in annotator_type:
        annotate_bounding_box_3d(data, frame_data, i)
    if "tcp2d" in annotator_type:
        annotate_tcp_trace_2d(data, frame_data, i)
    if "atcp2d" in annotator_type:
        annotate_all_tcp_trace_2d(data, frame_data, i)
    if "atcp2dd" in annotator_type:
        annotate_all_tcp_trace_2d_depth(data, frame_data, i)
    if "sm" in annotator_type:
        annotate_semantic_mask(data, frame_data, i)
    if "gp" in annotator_type:
        annotate_grasp_point(data, frame_data, i)


def _process_video_frames(
    data,
    data_path,
    camera_name,
    annotator_type,
    output_path,
    output_format,
    width,
    height,
    worker_id,
    dataset_name,
):
    """Process video frames using either PyAV or OpenCV."""

    # Use PyAV for better video encoding if available
    if HAS_AV and output_format == "mp4":
        return _process_with_pyav(
            data,
            data_path,
            camera_name,
            annotator_type,
            output_path,
            width,
            height,
            worker_id,
            dataset_name,
        )
    else:
        return _process_with_opencv(
            data,
            data_path,
            camera_name,
            annotator_type,
            output_path,
            output_format,
            width,
            height,
            worker_id,
            dataset_name,
        )


def _process_with_pyav(
    data,
    data_path,
    camera_name,
    annotator_type,
    output_path,
    width,
    height,
    worker_id,
    dataset_name,
):
    """Process video using PyAV with H.264 encoding."""
    try:
        # Initialize PyAV container and stream
        output_container = av.open(output_path, "w")  # type: ignore[attr-defined]
        stream = output_container.add_stream("libx264", 30)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        # Process frames
        for i in range(data["num_frames"]):
            frame_data = get_frame_data(data_path, camera_name, i, annotator_type)
            frame_data["color_image"] = cv2.cvtColor(
                frame_data["color_image"], cv2.COLOR_RGB2BGR
            )

            _apply_annotations(data, frame_data, i, annotator_type)
            annotate_all_text(
                data,
                frame_data,
                i,
                type("Args", (), {"annotator_type": annotator_type})(),
            )

            # Convert to RGB for PyAV
            rgb_frame = cv2.cvtColor(frame_data["color_image"], cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")  # type: ignore[attr-defined]
            av_frame = av_frame.reformat(format="yuv420p")

            # Encode and write frame
            for packet in stream.encode(av_frame):
                output_container.mux(packet)

            del frame_data

        # Flush encoder
        for packet in stream.encode():
            output_container.mux(packet)

        output_container.close()
        return True

    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        cv2.error,
        RuntimeError,
    ) as e:
        print(f"[Worker {worker_id}] PyAV encoding failed for {dataset_name}: {e}")
        return False


def _process_with_opencv(
    data,
    data_path,
    camera_name,
    annotator_type,
    output_path,
    output_format,
    width,
    height,
    worker_id,
    dataset_name,
):
    """Process video using OpenCV VideoWriter."""
    try:
        # Configure video encoding
        if output_format == "mp4":
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        elif output_format == "avi":
            fourcc = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore[attr-defined]

        # Initialize video writer
        video_writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))  # type: ignore[attr-defined]

        if not video_writer.isOpened():
            # Fallback to Motion JPEG
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore[attr-defined]
            output_path = os.path.splitext(output_path)[0] + ".avi"
            video_writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

            if not video_writer.isOpened():
                return False

        # Process frames
        for i in range(data["num_frames"]):
            frame_data = get_frame_data(data_path, camera_name, i, annotator_type)
            frame_data["color_image"] = cv2.cvtColor(
                frame_data["color_image"], cv2.COLOR_RGB2BGR
            )

            _apply_annotations(data, frame_data, i, annotator_type)
            annotate_all_text(
                data,
                frame_data,
                i,
                type("Args", (), {"annotator_type": annotator_type})(),
            )

            video_writer.write(frame_data["color_image"])
            del frame_data

        video_writer.release()
        return True

    except (AttributeError, KeyError, cv2.error, TypeError, ValueError, OSError) as e:
        print(f"[Worker {worker_id}] OpenCV encoding failed for {dataset_name}: {e}")
        return False


def process_batch_datasets(args):
    """Process multiple datasets in batch mode using multi-threading.

    Args:
        args (argparse.Namespace): Command line arguments
    """
    print(f"Starting batch processing with {args.num_workers} workers...")

    # Find all dataset directories
    dataset_dirs = find_dataset_directories(args.data_path)

    if not dataset_dirs:
        print("ERROR: No valid datasets found in the specified directory!")
        print(
            "   Make sure each subdirectory contains 'lmdb' folder and 'meta_info.pkl' file."
        )
        return

    print(f"Found {len(dataset_dirs)} datasets to process")

    # Create output directory
    output_parent = Path(args.output_path)
    output_parent.mkdir(parents=True, exist_ok=True)

    # Prepare tasks for worker threads
    tasks = []
    for data_path in dataset_dirs:
        dataset_name = os.path.basename(data_path)
        output_file = output_parent / f"{dataset_name}.{args.output_format}"

        task = partial(
            process_single_dataset,
            data_path=data_path,
            camera_name=args.camera_name,
            annotator_type=args.annotator_type,
            output_path=str(output_file),
            output_format=args.output_format,
            skip_existing=args.skip_existing,
        )
        tasks.append((task, dataset_name))

    # Process datasets using thread pool
    successful = 0
    failed = 0
    results = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks
        future_to_dataset = {
            executor.submit(task, worker_id=i % args.num_workers): dataset_name
            for i, (task, dataset_name) in enumerate(tasks)
        }

        # Process completed tasks with progress tracking
        with tqdm(total=len(tasks), desc="Processing datasets", unit="dataset") as pbar:
            for future in as_completed(future_to_dataset):
                dataset_name = future_to_dataset[future]

                try:
                    success, name, output_path, error_msg = future.result()
                    if success:
                        successful += 1
                        results.append(f"SUCCESS: {name} -> {output_path}")
                    else:
                        failed += 1
                        results.append(f"FAILED: {name}: {error_msg}")
                except Exception as e:
                    failed += 1
                    results.append(f"FAILED: {dataset_name}: Unexpected error - {e}")

                pbar.update(1)

    # Print final results
    print(f"\nBatch processing completed!")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Output directory: {args.output_path}")

    if results:
        print(f"\nDetailed results:")
        for result in results[:10]:  # Show first 10 results
            print(f"   {result}")
        if len(results) > 10:
            print(f"   ... and {len(results) - 10} more")


if __name__ == "__main__":
    args = parse_args()

    # Check if batch mode is enabled
    if args.batch_mode:
        print("Batch processing mode enabled")
        process_batch_datasets(args)
        exit()

    # Single dataset processing mode
    print("Single dataset processing mode")

    # Check if output file already exists and skip_existing is enabled
    if args.skip_existing:
        # Determine the actual output path based on format and content
        temp_data = get_meta_data(args.data_path, args.camera_name, args.annotator_type)
        if temp_data["num_frames"] == 1:
            # Single frame will be saved as PNG
            check_path = os.path.splitext(args.output_path)[0] + ".png"
        else:
            # Multi-frame will be saved with specified format
            if args.output_format == "mp4" and not args.output_path.endswith(".mp4"):
                check_path = os.path.splitext(args.output_path)[0] + ".mp4"
            elif args.output_format == "avi" and not args.output_path.endswith(".avi"):
                check_path = os.path.splitext(args.output_path)[0] + ".avi"
            else:
                check_path = args.output_path

        if os.path.exists(check_path):
            print(f"Output file already exists: {check_path}")
            print("Skipping processing (use --skip_existing=False to force overwrite)")
            exit()

    # Optimized processing pipeline with selective data loading
    print("Loading metadata...")
    data = get_meta_data(args.data_path, args.camera_name, args.annotator_type)
    Path(os.path.dirname(args.output_path)).mkdir(parents=True, exist_ok=True)

    # Single frame processing (PNG output)
    if data["num_frames"] == 1:
        print("Detected single frame dataset, processing as image...")
        i = 0
        frame_data = get_frame_data(
            args.data_path, args.camera_name, i, args.annotator_type
        )
        frame_data["color_image"] = cv2.cvtColor(
            frame_data["color_image"], cv2.COLOR_RGB2BGR
        )

        # Apply all enabled annotations
        if "di" in args.annotator_type:
            annotate_depth_image(data, frame_data, i)
        if "bb2d" in args.annotator_type:
            annotate_bbox2d(data, frame_data, i)
        if "bb3d" in args.annotator_type:
            annotate_bounding_box_3d(data, frame_data, i)
        if "tcp2d" in args.annotator_type:
            annotate_tcp_trace_2d(data, frame_data, i)
        if "atcp2d" in args.annotator_type:
            annotate_all_tcp_trace_2d(data, frame_data, i)
        if "atcp2dd" in args.annotator_type:
            annotate_all_tcp_trace_2d_depth(data, frame_data, i)
        if "sm" in args.annotator_type:
            annotate_semantic_mask(data, frame_data, i)
        if "gp" in args.annotator_type:
            annotate_grasp_point(data, frame_data, i)
        annotate_all_text(data, frame_data, i, args)

        cv2.imwrite(args.output_path, frame_data["color_image"])
        print(f"Single frame saved to: {args.output_path}")
        exit()

    # Multi-frame processing with streaming (MP4 output)
    print(f"Processing video with {data['num_frames']} frames...")

    # Determine video dimensions from first frame
    first_frame_data = get_frame_data(
        args.data_path, args.camera_name, 0, args.annotator_type
    )
    height, width = first_frame_data["color_image"].shape[:2]

    # Configure output path
    output_path = args.output_path
    if args.output_format == "mp4" and not output_path.endswith(".mp4"):
        output_path = os.path.splitext(output_path)[0] + ".mp4"
    elif args.output_format == "avi" and not output_path.endswith(".avi"):
        output_path = os.path.splitext(output_path)[0] + ".avi"

    print(f"Video dimensions: {width}x{height}")
    print(f"Output path: {output_path}")

    # Use PyAV for better video encoding if available
    if HAS_AV and args.output_format == "mp4":
        print("Using PyAV with H.264 encoding for optimal VSCode compatibility")

        # Initialize PyAV container and stream
        output_container = av.open(output_path, "w")  # type: ignore[attr-defined]
        stream = output_container.add_stream("libx264", 30)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"  # Standard pixel format for maximum compatibility

        # Process frames with PyAV
        for i in tqdm(
            range(data["num_frames"]), desc="Processing frames", unit="frame"
        ):
            # Load frame data on-demand to minimize memory usage
            frame_data = get_frame_data(
                args.data_path, args.camera_name, i, args.annotator_type
            )
            # Convert to BGR for OpenCV annotation functions, will convert back to RGB for PyAV
            frame_data["color_image"] = cv2.cvtColor(
                frame_data["color_image"], cv2.COLOR_RGB2BGR
            )

            # Apply all enabled annotations
            if "di" in args.annotator_type:
                annotate_depth_image(data, frame_data, i)
            if "bb2d" in args.annotator_type:
                annotate_bbox2d(data, frame_data, i)
            if "bb3d" in args.annotator_type:
                annotate_bounding_box_3d(data, frame_data, i)
            if "tcp2d" in args.annotator_type:
                annotate_tcp_trace_2d(data, frame_data, i)
            if "atcp2d" in args.annotator_type:
                annotate_all_tcp_trace_2d(data, frame_data, i)
            if "atcp2dd" in args.annotator_type:
                annotate_all_tcp_trace_2d_depth(data, frame_data, i)
            if "sm" in args.annotator_type:
                annotate_semantic_mask(data, frame_data, i)
            if "gp" in args.annotator_type:
                annotate_grasp_point(data, frame_data, i)
            annotate_all_text(data, frame_data, i, args)

            # Convert back to RGB for PyAV
            rgb_frame = cv2.cvtColor(frame_data["color_image"], cv2.COLOR_BGR2RGB)

            # Convert frame to PyAV format
            av_frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")  # type: ignore[attr-defined]
            av_frame = av_frame.reformat(format="yuv420p")

            # Encode and write frame
            for packet in stream.encode(av_frame):
                output_container.mux(packet)

            # Explicit memory cleanup
            del frame_data

        # Flush encoder
        for packet in stream.encode():
            output_container.mux(packet)

        # Close container
        output_container.close()

        print("Video processing completed successfully!")
        print(f"Video saved as: {output_path}")
        print("H.264 encoded video should play perfectly in VSCode!")

    else:
        # Fallback to OpenCV VideoWriter
        if not HAS_AV and args.output_format == "mp4":
            print(
                "WARNING: PyAV not available - using OpenCV fallback (may have VSCode compatibility issues)"
            )
            print("TIP: Install PyAV for better compatibility: pip install av")

        # Configure video encoding based on output format
        if args.output_format == "mp4":
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
            print("Using MP4 format with MPEG-4 encoding")
        elif args.output_format == "avi":
            fourcc = cv2.VideoWriter_fourcc(*"XVID")  # type: ignore[attr-defined]
            print("Using AVI format with XVID encoding")

        # Initialize video writer
        video_writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))  # type: ignore[attr-defined]

        if not video_writer.isOpened():
            # Fallback to Motion JPEG if primary codec fails
            print("Primary codec failed, falling back to Motion JPEG...")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # type: ignore[attr-defined]
            output_path = os.path.splitext(output_path)[0] + ".avi"
            video_writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

            if not video_writer.isOpened():
                raise RuntimeError(
                    "Failed to initialize video writer with any supported codec"
                )

        # Process frames with OpenCV
        for i in tqdm(
            range(data["num_frames"]), desc="Processing frames", unit="frame"
        ):
            # Load frame data on-demand to minimize memory usage
            frame_data = get_frame_data(
                args.data_path, args.camera_name, i, args.annotator_type
            )
            frame_data["color_image"] = cv2.cvtColor(
                frame_data["color_image"], cv2.COLOR_RGB2BGR
            )

            # Apply all enabled annotations
            if "di" in args.annotator_type:
                annotate_depth_image(data, frame_data, i)
            if "bb2d" in args.annotator_type:
                annotate_bbox2d(data, frame_data, i)
            if "bb3d" in args.annotator_type:
                annotate_bounding_box_3d(data, frame_data, i)
            if "tcp2d" in args.annotator_type:
                annotate_tcp_trace_2d(data, frame_data, i)
            if "atcp2d" in args.annotator_type:
                annotate_all_tcp_trace_2d(data, frame_data, i)
            if "atcp2dd" in args.annotator_type:
                annotate_all_tcp_trace_2d_depth(data, frame_data, i)
            if "sm" in args.annotator_type:
                annotate_semantic_mask(data, frame_data, i)
            if "gp" in args.annotator_type:
                annotate_grasp_point(data, frame_data, i)
            annotate_all_text(data, frame_data, i, args)

            # Write frame to video
            video_writer.write(frame_data["color_image"])

            # Explicit memory cleanup for large datasets
            del frame_data

        video_writer.release()
        print("Video processing completed successfully!")
        print(f"Video saved as: {output_path}")

        # Provide format-specific playback recommendations
        if args.output_format == "mp4":
            print(
                "WARNING: OpenCV MP4 may have VSCode compatibility issues. Consider installing PyAV."
            )
        else:
            print(
                "TIP: AVI format has wider compatibility. Use VLC or other media players if VSCode cannot play it."
            )
