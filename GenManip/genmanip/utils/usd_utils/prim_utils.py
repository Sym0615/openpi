"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from typing import Optional, Sequence  # type: ignore

import numpy as np
import open3d as o3d

import omni.usd  # type: ignore
from omni.isaac.core.prims import XFormPrim, RigidPrim  # type: ignore
from omni.isaac.core.utils.prims import get_prim_at_path  # type: ignore
from omni.isaac.core.utils.stage import add_reference_to_stage  # type: ignore
from pxr import Usd, UsdPhysics, UsdGeom, Gf  # type: ignore

from .collision_utils import set_colliders
from .mesh_utils import get_prim_bbox
from .rigid_utils import set_mass, set_rigid_body
from .semantic_utils import set_semantic_label
from .transform_utils import decompose_affine_transform, resolve_prim_local_transform
from genmanip.utils.standalone.pc_utils import compute_aabb_lwh, compute_mesh_bbox


def add_usd_to_world_reference(asset_path: str, prim_path: str, name: str):
    stage = omni.usd.get_context().get_stage()
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(asset_path)
    return XFormPrim(prim_path, name=name)


def add_usd_to_world(
    asset_path: str,
    prim_path: str,
    name: str,
    translation: Optional[Sequence[float]] = None,
    orientation: Optional[Sequence[float]] = None,
    scale: Optional[Sequence[float]] = None,
    add_rigid_body: bool = False,
    add_colliders: bool = False,
    collision_approximation: str = "convexDecomposition",
    mass: float | None = None,
    static_friction: float = 1.0,
    dynamic_friction: float = 1.0,
) -> XFormPrim:
    print(f"Adding USD to world: {asset_path} to {prim_path}")
    reference = add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)
    prim_path = str(reference.GetPrimPath())
    prim = XFormPrim(
        prim_path,
        name=name,
        translation=translation,
        orientation=orientation,
        scale=scale,
    )
    usd_prim = prim.prim
    if not usd_prim.IsValid():
        print(f"Prim at path {prim_path} is not valid.")
        return prim
    if add_colliders:
        set_colliders(prim_path, collision_approximation)
        print(f"CollisionAPI applied to {prim_path}")
    if add_rigid_body:
        set_rigid_body(
            prim_path,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
        )
        print(f"RigidBodyAPI applied to {prim_path}")
        if mass is not None:
            set_mass(prim_path, mass)
            print(f"MassAPI applied to {prim_path}")
    set_semantic_label(
        str(usd_prim.GetPath()), str(usd_prim.GetPath()).split("/")[-1][4:]
    )
    return prim


def resize_object_by_lwh(
    pre_object: XFormPrim,
    l: float,
    w: float,
    h: float,
    mesh: o3d.geometry.TriangleMesh | None = None,
) -> None:
    if mesh is None:
        aabb = get_prim_bbox(pre_object.prim)
    else:
        aabb = compute_mesh_bbox(mesh)
    x, y, z = compute_aabb_lwh(aabb)
    length_rate = l / x
    width_rate = w / y
    height_rate = h / z
    local_scale = pre_object.get_local_scale()
    local_scale[0] *= width_rate
    local_scale[1] *= length_rate
    local_scale[2] *= height_rate
    pre_object.set_local_scale(local_scale)


def resize_object(
    pre_object: XFormPrim,
    size: float,
    mesh: o3d.geometry.TriangleMesh | None = None,
) -> None:
    if mesh is None:
        aabb = get_prim_bbox(pre_object.prim)
    else:
        aabb = compute_mesh_bbox(mesh)
    x, y, z = compute_aabb_lwh(aabb)
    length_rate = size / x if size else 1.0
    width_rate = size / y if size else 1.0
    height_rate = size / z if size else 1.0
    local_scale = pre_object.get_local_scale()
    if x >= y and x >= z:
        pre_object.set_local_scale(local_scale * length_rate)
        return
    elif y >= x and y >= z:
        pre_object.set_local_scale(local_scale * width_rate)
        return
    elif z >= x and z >= y:
        pre_object.set_local_scale(local_scale * height_rate)
        return


def get_world_pose_by_prim_path(
    prim_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    prim = get_prim_at_path(prim_path)
    xformable = UsdGeom.Xformable(prim)
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world_transform = np.array(world_transform, dtype=float).T
    position, _, orientation, _ = decompose_affine_transform(world_transform)

    return position, orientation


def get_local_scale_by_prim_path(
    prim_path: str,
) -> np.ndarray:
    prim = get_prim_at_path(prim_path)
    xform = UsdGeom.Xformable(prim)
    local_transformation: Gf.Matrix4d = xform.GetLocalTransformation()
    scale: Gf.Vec3d = Gf.Vec3d(
        *(v.GetLength() for v in local_transformation.ExtractRotationMatrix())
    )
    return scale


def set_world_pose_by_prim_path(prim_path: str, world_pose: np.ndarray) -> None:
    xform_prim = XFormPrim(prim_path)
    xform_prim.set_world_pose(*world_pose)


def get_leaf_prims(prim: Usd.Prim) -> list[Usd.Prim]:
    # prim here include Xform and Mesh
    leaf_prims = set()

    def recurse_prim(current_prim):
        prim_type_name = current_prim.GetTypeName()
        if prim_type_name in ["Xform", "Mesh"]:
            leaf_prims.add(current_prim)
        if current_prim.GetChildren():
            for child in current_prim.GetChildren():
                recurse_prim(child)

    recurse_prim(prim)

    return list(leaf_prims)


def clean_prim_velocity(prim_path: str) -> Usd.Prim | None:
    prim = get_prim_at_path(prim_path)
    schema_list = prim.GetAppliedSchemas()
    if "PhysicsRigidBodyAPI" in schema_list:
        rigid_prim = RigidPrim(prim_path)
        rigid_prim.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
        rigid_prim.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        return prim
    else:
        return None


def get_prim_info(prim: Usd.Prim) -> dict:
    prim_info = {}
    prim_type = prim.GetTypeName()
    prim_path = str(prim.GetPath())

    local_transform = resolve_prim_local_transform(prim)
    translation = local_transform.translation
    orientation = local_transform.quat_wxyz
    scale = local_transform.scale

    mass_center = None
    if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.HasAPI(UsdPhysics.MassAPI):
        mass_center_gf = UsdPhysics.MassAPI(prim).GetCenterOfMassAttr().Get()
        mass_center = np.array(mass_center_gf).tolist()

    prim_info[prim_path] = {
        "translation": translation.tolist(),
        "orientation": orientation.tolist(),
        "scale": scale.tolist(),
        "mass_center": mass_center,
    }
    return prim_info


def set_prim_info(prim: Usd.Prim, prim_info: dict | None = None) -> Usd.Prim:
    if prim_info is None:
        print(f"prim info is None")
        return prim
    translation = np.array(prim_info["translation"])
    translation_gf = Gf.Vec3f(translation[0], translation[1], translation[2])
    orientation = np.array(prim_info["orientation"])
    # orientation_gf = Gf.Quatf(
    #     orientation[0], orientation[1], orientation[2], orientation[3]
    # )
    scale = np.array(prim_info["scale"])
    scale_gf = Gf.Vec3f(scale[0], scale[1], scale[2])

    prim_xform = UsdGeom.Xform(prim)

    # translation xform
    trans_attr = prim.GetAttribute("xformOp:translate")
    if not trans_attr:
        trans_attr = prim_xform.AddTranslateOp()
    trans_attr.Set(translation_gf)

    # orient xform
    orient_attr = prim.GetAttribute("xformOp:orient")
    if not orient_attr:
        orient_attr = prim_xform.AddOrientOp()
    orient_type = orient_attr.GetTypeName()
    if orient_type == "quatd":
        orientation_gf = Gf.Quatd(
            orientation[0], orientation[1], orientation[2], orientation[3]
        )
    else:  # Quatf
        orientation_gf = Gf.Quatf(
            orientation[0], orientation[1], orientation[2], orientation[3]
        )
    orient_attr.Set(orientation_gf)

    # scale xform
    scale_attr = prim.GetAttribute("xformOp:scale")
    if not scale_attr:
        scale_attr = prim_xform.AddScaleOp()
    scale_attr.Set(scale_gf)

    # prim.GetAttribute("xformOp:translate").Set(translation_gf)
    # prim.GetAttribute("xformOp:orient").Set(orientation_gf)
    # prim.GetAttribute("xformOp:scale").Set(scale_gf)

    if prim_info["mass_center"] is not None:
        if not prim.HasAPI(UsdPhysics.MassAPI):
            mass = UsdPhysics.MassAPI.Apply(prim)
            mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0, 0, 0))
        else:
            mass = UsdPhysics.MassAPI(prim)
        mass_center = np.array(prim_info["mass_center"])
        mass_center_gf = Gf.Vec3f(
            float(mass_center[0]), float(mass_center[1]), float(mass_center[2])
        )
        mass.GetCenterOfMassAttr().Set(mass_center_gf)
        # print(mass_center_gf)
    print(translation_gf, orientation_gf, scale_gf)
    return prim
