"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from omni.isaac.core.prims import GeometryPrim  # type: ignore
from omni.isaac.core.materials import PhysicsMaterial  # type: ignore
import omni.usd  # type: ignore
from pxr import UsdPhysics, PhysxSchema  # type: ignore


def setup_physics_scene(physics_scene_config) -> None:
    stage = omni.usd.get_context().get_stage()
    physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/physicsScene")

    for key, value in physics_scene_config.items():
        if isinstance(value, str) and "inf" in value.lower():
            value = float(value)

        method_name = f"Get{key}Attr"
        if hasattr(physxSceneAPI, method_name):
            attr = getattr(physxSceneAPI, method_name)()
        else:
            method_name = f"Create{key}Attr"
            attr = getattr(physxSceneAPI, method_name)()

        try:
            attr.Set(value)
            print(f"[OK] {method_name}().Set({value})")
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            print(f"[ERROR] {method_name}: {e}")


def add_physics_material(
    prim_path: str,
    static_friction: float = 1.0,
    dynamic_friction: float = 1.0,
) -> GeometryPrim:
    prim = str(prim_path)
    prim = GeometryPrim(prim_path)
    try:
        physics_material = PhysicsMaterial(
            prim_path=prim_path + "/physics_material",
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=0.1,
        )
        prim.apply_physics_material(physics_material)
    except (OSError, RuntimeError, TypeError, ValueError):
        print("Physics material already exists")
    return prim


def set_robot_physics_material(prim_path: str, setting: dict) -> None:
    stage = omni.usd.get_context().get_stage()
    usd_material_api = UsdPhysics.MaterialAPI.Get(stage, prim_path)
    physx_schema_api = PhysxSchema.PhysxMaterialAPI.Get(stage, prim_path)

    def _get_attr(method_name):
        if hasattr(usd_material_api, method_name):
            return getattr(usd_material_api, method_name)()

        if hasattr(physx_schema_api, method_name):
            return getattr(physx_schema_api, method_name)()

        return None

    for key, value in setting.items():
        method_name = f"Get{key}Attr"
        attr = _get_attr(method_name)

        if attr is None:
            method_name = f"Create{key}Attr"
            attr = _get_attr(method_name)

        if attr is None:
            print(f"[SKIP] {key} has no Get{key}Attr() or Create{key}Attr()")
            continue

        try:
            if not hasattr(attr, "Set"):
                print(f"[SKIP] {method_name}() has no Set()")
                continue

            attr.Set(value)
            print(f"[OK] {method_name}().Set({value})")
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            print(f"[ERROR] {method_name}: {e}")


def set_robot_contact_offset(prim_paths: list, offset_value: dict) -> None:
    stage = omni.usd.get_context().get_stage()

    for prim_path in prim_paths:
        physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Get(stage, prim_path)
        physxCollisionAPI.GetContactOffsetAttr().Set(offset_value)


def set_robot_rest_offset(prim_paths: list, offset_value: dict) -> None:
    stage = omni.usd.get_context().get_stage()

    for prim_path in prim_paths:
        physxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Get(stage, prim_path)
        physxCollisionAPI.GetRestOffsetAttr().Set(offset_value)
