"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import os
from pathlib import Path
import re

import cv2
import numpy as np

try:
    import mediapy as media  # type: ignore
except ImportError:
    print("mediapy is not installed, please install it with 'pip install mediapy'")


def create_video_from_image_folder_with_mediapy(
    image_folder: str,
    output_video_path: str,
    fps: float = 30,
    frame_ending: str = ".png",
) -> None:
    # images: list of numpy arrays
    images = [img for img in os.listdir(image_folder) if img.endswith(frame_ending)]
    images.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
    images_npy = [
        cv2.cvtColor(
            cv2.imread(os.path.join(image_folder, img), cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB,
        )
        for img in images
    ]
    media.write_video(output_video_path, images_npy, fps=fps)  # type: ignore[attr-defined]


def create_video_from_image_list_with_mediapy(
    image_list: list[np.ndarray | bytes],
    output_video_path: str,
    fps: float = 30,
    frame_ending: str = ".png",
) -> None:
    if not image_list:
        return

    if isinstance(image_list[0], (bytes, bytearray, memoryview)):
        # Stream JPEG-encoded frames through mediapy.VideoWriter one at a time
        # so peak memory stays at a single decoded frame instead of the whole
        # episode. This keeps the H.264 codec (Rerun-compatible) without the
        # >30GB materialization the naive decoded-list approach caused.
        first = None
        start_idx = 0
        for i, raw in enumerate(image_list):
            first = decode_image_frame(raw)
            if first is not None:
                start_idx = i
                break
        if first is None:
            return
        height, width = first.shape[:2]
        with media.VideoWriter(  # type: ignore[attr-defined]
            output_video_path, shape=(height, width), fps=fps
        ) as writer:
            writer.add_image(first)
            for raw in image_list[start_idx + 1 :]:
                frame = decode_image_frame(raw)
                if frame is not None:
                    writer.add_image(frame)
        return

    media.write_video(output_video_path, image_list, fps=fps)  # type: ignore[attr-defined]


def save_image_with_description(
    image: np.ndarray,
    uuid: str,
    text: str,
    folder_path: str,
) -> None:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, font_thickness
    )
    image_height, image_width, _ = image.shape
    x = (image_width - text_width) // 2
    y = text_height + 10
    cv2.putText(image, text, (x, y), font, font_scale, (0, 0, 0), font_thickness)
    if not os.path.exists(folder_path):
        Path(folder_path).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(folder_path, f"{uuid}.png")
    cv2.imwrite(filepath, image)


def compress_to_jpeg_array(image_array: np.ndarray) -> np.ndarray:
    _, jpeg_array = cv2.imencode(".jpg", image_array)
    decoded_image = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
    return decoded_image


def decode_image_frame(frame: np.ndarray | bytes) -> np.ndarray | None:
    if isinstance(frame, np.ndarray):
        if frame.ndim == 3 and frame.size > 0:
            return frame
        return None
    if isinstance(frame, (bytes, bytearray, memoryview)):
        buffer = np.frombuffer(frame, dtype=np.uint8)
        frame_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if isinstance(frame_bgr, np.ndarray) and frame_bgr.size > 0:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return None


def create_video_from_image_folder(
    image_folder: str,
    output_video_path: str,
    fps: float = 30,
    frame_ending: str = ".png",
) -> None:
    images = [img for img in os.listdir(image_folder) if img.endswith(frame_ending)]
    images.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    try:
        for image in images:
            image_path = os.path.join(image_folder, image)
            frame = cv2.imread(image_path)
            video.write(frame)
    finally:
        video.release()


def create_video_from_image_list(
    image_list: list[np.ndarray | bytes],
    output_video_path: str,
    fps: float = 30,
) -> None:
    first_frame = None
    for image in image_list:
        first_frame = decode_image_frame(image)
        if first_frame is not None:
            break
    if first_frame is None:
        raise ValueError("image_list does not contain any decodable frame")
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    try:
        for image in image_list:
            frame = decode_image_frame(image)
            if frame is None:
                continue
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)
    finally:
        video.release()


def create_video_from_image_array(
    image_array: np.ndarray, output_video_path: str, fps: float = 30
) -> None:
    height, width, _ = image_array[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    try:
        for frame in image_array:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame)
    finally:
        video.release()


def save_image(image: np.ndarray, filepath: str) -> None:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, image_bgr)


def visualize_3d_bbox(
    img: np.ndarray,
    pixel_coordinates: list[tuple[int, int]],
    point_color: tuple[int, int, int] = (0, 0, 255),
    point_size: int = 5,
    edge_color: tuple[int, int, int] = (0, 255, 0),
    edge_thickness: int = 2,
    draw_planes: bool = False,
    plane_alpha: float = 0.3,
) -> np.ndarray:
    """
    Visualize 3D bounding box by drawing points and edges on the image.

    Args:
        img: Input image to draw on
        pixel_coordinates: 2D coordinates of 3D bounding box corners
        point_color: Color for corner points (B,G,R)
        point_size: Size of corner points
        edge_color: Color for edges (B,G,R)
        edge_thickness: Thickness of edge lines
        draw_planes: Whether to draw colored planes
        plane_alpha: Transparency of planes (0-1)

    Returns:
        Image with visualized bounding box
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

    # Define the edges of the 3D bounding box
    edges = [
        (0, 1),
        (1, 3),
        (3, 2),
        (2, 0),  # Back face
        (4, 5),
        (5, 7),
        (7, 6),
        (6, 4),  # Front face
        (0, 4),
        (1, 5),
        (3, 7),
        (2, 6),  # Connecting edges
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

    # Draw planes if requested
    if draw_planes:
        # Define planes with corner indices and colors
        planes = [
            {"corners": [0, 1, 3, 2], "color": edge_color},  # Back face (blue)
            {"corners": [4, 5, 7, 6], "color": edge_color},  # Front face (green)
            {"corners": [0, 2, 6, 4], "color": edge_color},  # Left face (red)
            {"corners": [1, 3, 7, 5], "color": edge_color},  # Right face (cyan)
            {"corners": [2, 3, 7, 6], "color": edge_color},  # Top face (magenta)
            {"corners": [0, 1, 5, 4], "color": edge_color},  # Bottom face (yellow)
        ]

        for plane in planes:
            # Create contour from corner points
            contour = np.array(
                [
                    [int(pixel_coordinates[i][0]), int(pixel_coordinates[i][1])]
                    for i in plane["corners"]
                ],
                dtype=np.int32,
            )

            # Create a copy of the image for overlay
            overlay = img.copy()
            # Fill the polygon
            cv2.fillPoly(overlay, [contour], plane["color"])
            # Blend with original image
            img = cv2.addWeighted(overlay, plane_alpha, img, 1 - plane_alpha, 0)

    return img
