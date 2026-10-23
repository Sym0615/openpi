"""Batched YAM grasp-site FK in the arm-base frame used by ABC-130K.

The model is the same FK-only MJCF used for ABC-130K consistency checks.
Evaluating its six hinge transforms in Torch avoids a simulation or rendering
dependency in training workers. Joint inputs are absolute angles in radians.
"""

import functools
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

MODEL_PATH = Path(__file__).parent / "robot_models/yam/yam_fk.xml"


def _vector(element: ET.Element, name: str, default: str) -> torch.Tensor:
    return torch.tensor([float(value) for value in element.get(name, default).split()], dtype=torch.float64)


def _rotation(element: ET.Element) -> torch.Tensor:
    quat = _vector(element, "quat", "1 0 0 0")
    w, x, y, z = quat / quat.norm()
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ]
    ).reshape(3, 3)


@functools.lru_cache(maxsize=1)
def _chain() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    body = ET.parse(MODEL_PATH).find("./worldbody/body[@name='arm']")
    if body is None:
        raise ValueError(f"YAM model {MODEL_PATH} has no arm base.")
    positions, rotations = [], []
    for index in range(1, 7):
        body = body.find(f"./body[@name='link_{index}']")
        if body is None:
            raise ValueError(f"YAM model is missing link_{index}.")
        joint = body.find(f"./joint[@name='joint{index}']")
        if (
            joint is None
            or joint.get("type", "hinge") != "hinge"
            or not torch.equal(_vector(joint, "axis", "0 0 1"), torch.tensor([0.0, 0.0, 1.0]))
            or _vector(joint, "pos", "0 0 0").count_nonzero() != 0
            or float(joint.get("ref", "0")) != 0
        ):
            raise ValueError(f"YAM joint{index} must be an origin-centered Z hinge with zero reference.")
        positions.append(_vector(body, "pos", "0 0 0"))
        rotations.append(_rotation(body))
    site = body.find("./site[@name='grasp_site']")
    if site is None:
        raise ValueError("YAM model is missing grasp_site.")
    return torch.stack(positions), torch.stack(rotations), _vector(site, "pos", "0 0 0"), _rotation(site)


def forward_kinematics(joints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return XYZ (metres) and rotation matrix for absolute ``(..., 6)`` joints.

    The TCP includes grasp_site's 0.1347 m offset and orientation. Both arms
    use their own base frame, so no world mounting translation is included.
    """
    if joints.shape[-1] != 6:
        raise ValueError(f"YAM FK requires six joint angles, got shape {tuple(joints.shape)}.")
    if not torch.is_floating_point(joints):
        joints = joints.to(torch.float32)
    if not torch.isfinite(joints).all():
        raise ValueError("YAM FK joint angles must be finite.")
    offsets, fixed_rotations, tcp_offset, tcp_rotation = (value.to(joints) for value in _chain())
    position = joints.new_zeros((*joints.shape[:-1], 3))
    rotation = torch.eye(3, dtype=joints.dtype, device=joints.device).expand(*joints.shape[:-1], 3, 3)
    for index in range(6):
        position = position + (rotation @ offsets[index].unsqueeze(-1)).squeeze(-1)
        rotation = rotation @ fixed_rotations[index]
        cosine, sine = joints[..., index].cos(), joints[..., index].sin()
        zero, one = torch.zeros_like(cosine), torch.ones_like(cosine)
        motion = torch.stack([cosine, -sine, zero, sine, cosine, zero, zero, zero, one], dim=-1)
        rotation = rotation @ motion.reshape(*joints.shape[:-1], 3, 3)
    return position + (rotation @ tcp_offset.unsqueeze(-1)).squeeze(-1), rotation @ tcp_rotation
