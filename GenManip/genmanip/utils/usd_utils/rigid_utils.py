"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from omni.isaac.core.utils.prims import get_prim_at_path  # type: ignore
from pxr import PhysxSchema, UsdPhysics, Usd  # type: ignore

from .collision_utils import set_contact_offset_recursively, set_rest_offset_recursively
from .physics_utils import add_physics_material


def set_gravity(prim_path: str, gravity_enabled: bool) -> Usd.Prim:
    prim = get_prim_at_path(prim_path)
    rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rigid_body.CreateDisableGravityAttr().Set(not gravity_enabled)
    return prim


def set_rigid_body_CCD(prim_path: str, ccd_enabled: bool) -> Usd.Prim:
    prim = get_prim_at_path(prim_path)
    rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rigid_body.CreateEnableCCDAttr().Set(ccd_enabled)
    return prim


def set_rigid_body_max_contact_impulse(
    prim_path: str, max_contact_impulse: float = 5.0
) -> Usd.Prim:
    prim = get_prim_at_path(prim_path)
    rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rigid_body.CreateMaxContactImpulseAttr().Set(max_contact_impulse)
    return prim


def set_mass(prim_path: str, mass: float) -> Usd.Prim:
    prim = get_prim_at_path(prim_path)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(mass)
    return prim


def set_rigid_body(
    prim_path: str, static_friction: float = 1.0, dynamic_friction: float = 1.0
) -> Usd.Prim:
    prim = get_prim_at_path(prim_path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    set_rigid_body_CCD(prim_path, False)
    # set_colliders(prim_path)
    # coacd_prim = get_prim_at_path(f"{prim_path}/coacd")
    # if coacd_prim.IsValid() and coacd_prim.IsActive():
    #     coacd_prim.SetActive(False)
    set_contact_offset_recursively(prim_path, 0.02)
    set_rest_offset_recursively(prim_path, 0.0)
    add_physics_material(
        prim_path, static_friction=static_friction, dynamic_friction=dynamic_friction
    )
    set_rigid_body_solver_position_iteration_count(prim_path, 32)
    # set_rigid_body_solver_velocity_iteration_count(prim_path, 10)
    # set_rigid_body_linear_damping(prim_path, 10.0)
    # set_rigid_body_angular_damping(prim_path, 10.0)
    # set_rigid_body_contact_slop_coefficient(prim_path, 0.05)
    # set_rigid_body_max_contact_impulse(prim_path, 5.0)
    return prim


def set_rigid_body_linear_damping(prim_path: str, damping: float) -> Usd.Prim:
    prim = get_prim_at_path(prim_path)
    rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rigid_body.CreateLinearDampingAttr().Set(damping)
    return prim


def set_rigid_body_angular_damping(prim_path: str, damping: float) -> Usd.Prim:
    prim = get_prim_at_path(prim_path)
    rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rigid_body.CreateAngularDampingAttr().Set(damping)
    return prim


def set_rigid_body_contact_slop_coefficient(
    prim_path: str, slop_coefficient: float
) -> Usd.Prim:
    prim = get_prim_at_path(prim_path)
    rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rigid_body.CreateContactSlopCoefficientAttr().Set(slop_coefficient)
    return prim


def set_rigid_body_enable_speculative_ccd(prim_path: str, enable: bool) -> Usd.Prim:
    prim = get_prim_at_path(prim_path)
    rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rigid_body.CreateEnableSpeculativeCCDAttr().Set(enable)
    return prim


def set_rigid_body_retain_accelerations(prim_path: str, enable: bool) -> Usd.Prim:
    prim = get_prim_at_path(prim_path)
    rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rigid_body.CreateRetainAccelerationsAttr().Set(enable)
    return prim


def set_rigid_body_solver_position_iteration_count(
    prim_path: str, count: int
) -> Usd.Prim:
    prim = get_prim_at_path(prim_path)
    rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rigid_body.CreateSolverPositionIterationCountAttr().Set(count)
    return prim


def set_rigid_body_solver_velocity_iteration_count(
    prim_path: str, count: int
) -> Usd.Prim:
    prim = get_prim_at_path(prim_path)
    rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rigid_body.CreateSolverVelocityIterationCountAttr().Set(count)
    return prim
