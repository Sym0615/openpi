import math
from typing import List, Literal, Tuple, Any
from pydantic import BaseModel, Field, field_validator

from pxr import Usd, UsdGeom, Gf
from omni.isaac.core.utils.stage import get_current_stage

from genmanip.core.metrics.base import BaseMetric
from genmanip.core.metrics.utils import MetricFactory


class OrientationConstraint(BaseModel):
    local_axis: Tuple[float, float, float] = Field(
        ..., description="Local axis vector or axis name (e.g. +X, -y, Z)"
    )
    world_axis: Tuple[float, float, float] = Field(
        ..., description="World axis vector or axis name (e.g. +X, -y, Z)"
    )
    angle_range: Tuple[int, int] = Field(
        ...,
        description="Allowed angle range in degrees: (min, max), 0 <= min < max < 180",
    )

    # ---------- axis parsing ----------

    @field_validator("local_axis", "world_axis", mode="before")
    @classmethod
    def parse_axis(cls, v):
        # case 1: string like "+X", "-y", "Z"
        if isinstance(v, str):
            s = v.strip()
            if not s:
                raise ValueError("axis string cannot be empty")

            sign = 1
            if s[0] in "+-":
                if s[0] == "-":
                    sign = -1
                s = s[1:]

            axis = s.upper()
            if axis not in {"X", "Y", "Z"}:
                raise ValueError(
                    "axis must be one of X, Y, Z (optionally prefixed with + or -)"
                )

            if axis == "X":
                return (sign * 1.0, 0.0, 0.0)
            if axis == "Y":
                return (0.0, sign * 1.0, 0.0)
            if axis == "Z":
                return (0.0, 0.0, sign * 1.0)

        # case 2: numeric sequence
        if isinstance(v, (list, tuple)):
            if len(v) != 3:
                raise ValueError("axis vector must have exactly 3 elements")
            if not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("axis vector elements must be numbers")
            return tuple(float(x) for x in v)

        raise TypeError(
            "axis must be a string like '+X', '-y', 'Z' or a sequence of 3 numbers"
        )

    # ---------- angle range ----------

    @field_validator("angle_range")
    @classmethod
    def validate_angle_range(cls, v):
        min_a, max_a = v
        if not (0 <= min_a < max_a < 180):
            raise ValueError("angle_range must satisfy 0 <= min < max < 180")
        return v


class OrientationConstraintSpec(BaseModel):
    type: Literal["AND", "OR"] = Field(
        "AND", description="Logical operator to combine constraints"
    )
    constraints: List[OrientationConstraint] = Field(
        ..., description="List of orientation constraints"
    )


class MatchLocalWorldOrientationConfig(BaseModel):
    obj_uid: str = Field(..., description="UID of the object")
    constraint_spec: OrientationConstraintSpec = Field(
        ..., description="Orientation matching constraints"
    )


@MetricFactory.register("manip/default/match_local_world_orientation")
class MatchLocalWorldOrientation(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)

        self.setting = MatchLocalWorldOrientationConfig(**sub_goal_setting)
        self.operator = self.setting.constraint_spec.type == "AND"

    def check_status(self, scene):
        obj_xform_prim = scene.object_list[self.setting.obj_uid]
        obj_xformable = UsdGeom.Xformable(obj_xform_prim.prim)

        time = Usd.TimeCode.Default()
        world_transform: Gf.Matrix4d = obj_xformable.ComputeLocalToWorldTransform(time)
        R = world_transform.ExtractRotationMatrix().GetTranspose()

        results = []

        for _c in self.setting.constraint_spec.constraints:
            _f = self.check_angle_constraint_usd(
                R, _c.local_axis, _c.world_axis, _c.angle_range
            )
            results.append(_f)

        return all(results) if self.operator else any(results)

    def check_angle_constraint_usd(
        self,
        R,
        local_axis,
        world_axis,
        angle_range,
    ):
        local_axis = MatchLocalWorldOrientation.normalize(local_axis)
        world_axis = MatchLocalWorldOrientation.normalize(world_axis)

        world_vec = self.rotate_vector_usd(R, local_axis)
        world_vec = MatchLocalWorldOrientation.normalize(world_vec)

        cos_theta = MatchLocalWorldOrientation.dot(world_vec, world_axis)

        if cos_theta > math.cos(math.radians(angle_range[0])):
            return False

        if cos_theta < math.cos(math.radians(angle_range[1])):
            return False

        return True

    def rotate_vector_usd(self, R: Gf.Matrix3d, v):
        vec = Gf.Vec3d(v[0], v[1], v[2])
        out = R * vec
        return (out[0], out[1], out[2])

    @staticmethod
    def normalize(v):
        l = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        return (v[0] / l, v[1] / l, v[2] / l)

    @staticmethod
    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
