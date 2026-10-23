"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from omni.isaac.core.utils.prims import get_prim_at_path  # type: ignore
from pxr import PhysxSchema, Sdf, Usd, UsdPhysics  # type: ignore


def remove_colliders(prim_path: str) -> None:
    prim = get_prim_at_path(prim_path)
    if not prim.IsValid():
        return
    schema_list = prim.GetAppliedSchemas()
    if "PhysicsCollisionAPI" in schema_list:
        prim.RemoveAPI(UsdPhysics.CollisionAPI)
    if "PhysicsMeshCollisionAPI" in schema_list:
        prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
    if "PhysxConvexHullCollisionAPI" in schema_list:
        prim.RemoveAPI(PhysxSchema.PhysxConvexHullCollisionAPI)
    if "PhysxConvexDecompositionCollisionAPI" in schema_list:
        prim.RemoveAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI)
    if "PhysxSDFMeshCollisionAPI" in schema_list:
        prim.RemoveAPI(PhysxSchema.PhysxSDFMeshCollisionAPI)
    if "PhysxTriangleMeshCollisionAPI" in schema_list:
        prim.RemoveAPI(PhysxSchema.PhysxTriangleMeshCollisionAPI)
    for child in prim.GetAllChildren():
        remove_colliders(str(child.GetPath()))


def set_rest_offset(prim_path: str, rest_offset: float) -> Usd.Prim:
    prim = get_prim_at_path(prim_path)
    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision_api.CreateRestOffsetAttr().Set(rest_offset)
    return prim


def set_contact_offset(prim_path: str, contact_offset: float) -> Usd.Prim:
    prim = get_prim_at_path(prim_path)
    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision_api.CreateContactOffsetAttr().Set(contact_offset)
    return prim


def remove_contact_offset(prim_path: str) -> Usd.Prim:
    prim = get_prim_at_path(prim_path)
    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision_api.CreateContactOffsetAttr().Set(float("-inf"))
    return prim


def set_rest_offset_recursively(prim_path: str, rest_offset: float) -> Usd.Prim:
    prim = get_prim_at_path(prim_path)
    for child in prim.GetAllChildren():
        set_rest_offset_recursively(str(child.GetPath()), rest_offset)
    schema_list = prim.GetAppliedSchemas()
    if "PhysicsCollisionAPI" in schema_list:
        set_rest_offset(prim_path, rest_offset)


def set_contact_offset_recursively(prim_path: str, contact_offset: float) -> Usd.Prim:
    prim = get_prim_at_path(prim_path)
    for child in prim.GetAllChildren():
        set_contact_offset_recursively(str(child.GetPath()), contact_offset)
    schema_list = prim.GetAppliedSchemas()
    if "PhysicsCollisionAPI" in schema_list:
        set_contact_offset(prim_path, contact_offset)
    return prim


def remove_contact_offset_recursively(prim: Usd.Prim) -> Usd.Prim:
    schema_list = prim.GetAppliedSchemas()
    if "PhysicsCollisionAPI" in schema_list:
        remove_contact_offset(str(prim.GetPath()))
    for child in prim.GetAllChildren():
        remove_contact_offset_recursively(child)
    return prim


def set_colliders(
    prim_path: str,
    collision_approximation: str = "convexDecomposition",
    convex_hulls: int | None = None,
):
    remove_colliders(prim_path)
    prim = set_colliders_by_prim_path(prim_path, collision_approximation, convex_hulls)
    return prim


def set_colliders_by_prim_path(
    prim_path: str,
    collision_approximation: str = "convexDecomposition",
    convex_hulls: int | None = None,
):
    prim = get_prim_at_path(prim_path)
    for child in prim.GetAllChildren():
        set_colliders_by_prim_path(
            str(child.GetPath()), collision_approximation, convex_hulls
        )
    if prim.GetTypeName() == "Mesh":
        collider = UsdPhysics.CollisionAPI.Apply(prim)
        mesh_collider = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collider.CreateApproximationAttr().Set(collision_approximation)
        collider.GetCollisionEnabledAttr().Set(True)
        if collision_approximation == "convexDecomposition":
            collision_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
            collision_api.CreateHullVertexLimitAttr().Set(64)
            if convex_hulls is not None:
                collision_api.CreateMaxConvexHullsAttr().Set(convex_hulls)
            else:
                collision_api.CreateMaxConvexHullsAttr().Set(16)
            collision_api.CreateMinThicknessAttr().Set(0.001)
            collision_api.CreateShrinkWrapAttr().Set(True)
            collision_api.CreateErrorPercentageAttr().Set(0.1)
            # collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            # collision_api.CreateContactOffsetAttr().Set(0.1)
        elif collision_approximation == "convexHull":
            collision_api = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
            collision_api.CreateHullVertexLimitAttr().Set(64)
            collision_api.CreateMinThicknessAttr().Set(0.00001)
        elif collision_approximation == "sdf":
            collision_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
            collision_api.CreateSdfResolutionAttr().Set(1024)
    return prim


def set_max_convex_hulls(prim_path: str, max_convex_hulls: int) -> Usd.Prim:
    prim = get_prim_at_path(prim_path)
    prim.CreateAttribute(
        "physxConvexDecompositionCollision:maxConvexHulls", Sdf.ValueTypeNames.Int
    ).Set(max_convex_hulls)
    return prim


def has_collision_api(prim: Usd.Prim) -> bool:
    for child in prim.GetAllChildren():
        if has_collision_api(child):
            return True
    schema_list = prim.GetAppliedSchemas()
    if "PhysicsCollisionAPI" in schema_list:
        return True
    return False
