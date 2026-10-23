"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from omni.isaac.core.prims import XFormPrim  # type: ignore
from omni.isaac.core.robots.robot import Robot  # type: ignore
from omni.isaac.core.utils.prims import get_prim_at_path  # type: ignore
import omni.usd  # type: ignore
from pxr import PhysxSchema, Usd, UsdPhysics  # type: ignore


def get_robot_all_joints(robot_prim: Usd.Prim, joint_dict: dict) -> dict:
    for child in robot_prim.GetChildren():
        if child.IsA(UsdPhysics.Joint):
            joint_dict[child.GetName()] = UsdPhysics.Joint(child)
        else:
            get_robot_all_joints(child, joint_dict)
    return joint_dict


def get_robot_all_links(robot_prim: Usd.Prim) -> dict:
    joint_dict = {}
    joint_dict = get_robot_all_joints(robot_prim, joint_dict)
    link_dict = {}
    for value in joint_dict.values():
        body0_rel = value.GetBody0Rel()
        body0_path = body0_rel.GetTargets()
        if len(body0_path) > 0:
            body0_prim = get_prim_at_path(str(body0_path[0]))
            if body0_prim.IsValid():
                link_dict[str(body0_prim.GetPath())] = body0_prim
        body1_rel = value.GetBody1Rel()
        body1_path = body1_rel.GetTargets()
        if len(body1_path) > 0:
            body1_prim = get_prim_at_path(str(body1_path[0]))
            if body1_prim.IsValid():
                link_dict[str(body1_prim.GetPath())] = body1_prim
    return link_dict


def get_all_joints(prim: Usd.Prim) -> list[UsdPhysics.Joint]:
    joint_list = []

    def recurse_prim(current_prim):
        for child in current_prim.GetChildren():
            if child.IsA(UsdPhysics.Joint):
                joint_type = child.GetTypeName()
                if joint_type == "PhysicsPrismaticJoint":
                    joint = UsdPhysics.PrismaticJoint(child)
                elif joint_type == "PhysicsRevoluteJoint":
                    joint = UsdPhysics.RevoluteJoint(child)
                else:
                    # joint = UsdPhysics.Joint(child)
                    continue
                joint_list.append(joint)
            recurse_prim(child)

    recurse_prim(prim)

    return joint_list


def set_angular_drive(
    joint_prim: UsdPhysics.Joint,
    stiffness: float = 0.0,
    target_position: float = 0.0,
    damping: float = 0.0,
    target_velocity: float = 0.0,
) -> None:
    angular_drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, UsdPhysics.Tokens.angular)
    angular_drive_api.CreateTypeAttr(UsdPhysics.Tokens.force)
    angular_drive_api.CreateStiffnessAttr(stiffness)
    angular_drive_api.CreateTargetPositionAttr(target_position)
    angular_drive_api.CreateDampingAttr(damping)
    angular_drive_api.CreateTargetVelocityAttr(target_velocity)


def set_drive_max_force(prim_path: str, max_force: float) -> Usd.Prim:
    prim = get_prim_at_path(prim_path)
    joint_type = prim.GetTypeName()
    if joint_type == "PhysicsPrismaticJoint":
        drive_api = UsdPhysics.DriveAPI.Apply(prim, UsdPhysics.Tokens.linear)
    elif joint_type == "PhysicsRevoluteJoint":
        drive_api = UsdPhysics.DriveAPI.Apply(prim, UsdPhysics.Tokens.angular)
    else:
        print(f"Joint type {joint_type} is not supported")
        return prim
    drive_api.CreateMaxForceAttr().Set(max_force)
    return prim


def set_drive_damping_and_stiffness(
    prim_path: str, damping: float = 0.1, stiffness: float = 10.0
) -> Usd.Prim:
    prim = get_prim_at_path(prim_path)
    if not prim.IsValid():
        print(f"Prim {prim_path} is not valid")
        return prim
    if not prim.IsA(UsdPhysics.Joint):
        print(f"Prim {prim_path} is not a joint")
        return prim
    joint_type = prim.GetTypeName()
    if joint_type == "PhysicsPrismaticJoint":
        drive_api = UsdPhysics.DriveAPI.Apply(prim, UsdPhysics.Tokens.linear)
    elif joint_type == "PhysicsRevoluteJoint":
        drive_api = UsdPhysics.DriveAPI.Apply(prim, UsdPhysics.Tokens.angular)
    else:
        print(f"Joint type {joint_type} is not supported")
        return prim
    drive_api.CreateDampingAttr().Set(damping)
    drive_api.CreateStiffnessAttr().Set(stiffness)
    return prim


def get_joint_info(joint: UsdPhysics.Joint) -> dict:
    path = str(joint.GetPath())
    axis = joint.GetAxisAttr().Get()
    lower_limit = joint.GetLowerLimitAttr().Get()
    upper_limit = joint.GetUpperLimitAttr().Get()
    joint_info = {}
    joint_info[path] = {
        "axis": axis,
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
    }
    return joint_info


def set_joint_info(
    joint: UsdPhysics.Joint, joint_info: dict | None = None
) -> UsdPhysics.Joint:
    if not joint_info:
        print(f"joint info is None")
        return joint
    joint.GetAxisAttr().Set(joint_info["axis"])
    joint.GetLowerLimitAttr().Set(joint_info["lower_limit"])
    joint.GetUpperLimitAttr().Set(joint_info["upper_limit"])
    return joint


def get_joint(
    prim_path: str, type: str = "PhysicsRevoluteJoint"
) -> UsdPhysics.Joint | None:
    stage = omni.usd.get_context().get_stage()
    if type == "PhysicsRevoluteJoint":
        return UsdPhysics.RevoluteJoint.Define(stage, prim_path)
    elif type == "PhysicsPrismaticJoint":
        return UsdPhysics.PrismaticJoint.Define(stage, prim_path)
    else:
        print(f"Joint type {type} is not supported")
        return None


def set_mimic_joint(
    target_joint: UsdPhysics.Joint,
    relative_joint: UsdPhysics.Joint,
    gearing: float = 1.0,
    offset: float = 0.0,
) -> UsdPhysics.Joint:
    mimic_api = PhysxSchema.PhysxMimicJointAPI.Apply(
        target_joint.GetPrim(), UsdPhysics.Tokens.rotX
    )
    mimic_api.GetReferenceJointRel().AddTarget(relative_joint.GetPath())
    mimic_api.GetGearingAttr().Set(gearing)
    mimic_api.GetOffsetAttr().Set(offset)
    return target_joint


def get_all_joints_simple(robot_prim: XFormPrim, joint_dict: dict) -> dict:
    for child in robot_prim.GetChildren():
        if child.IsA(UsdPhysics.Joint):
            joint_dict[child.GetName()] = UsdPhysics.Joint(child)
        else:
            get_all_joints_simple(child, joint_dict)
    return joint_dict


def get_all_body_from_joint(
    joint_dict: dict[str, UsdPhysics.Joint], body_dict: dict[str, XFormPrim]
) -> dict[str, XFormPrim]:
    for value in joint_dict.values():
        body0_rel = value.GetBody0Rel()
        body0_path = body0_rel.GetTargets()
        if len(body0_path) > 0:
            body0_prim = get_prim_at_path(str(body0_path[0]))
            if body0_prim.IsValid():
                body_dict[body0_prim.GetName()] = body0_prim
        body1_rel = value.GetBody1Rel()
        body1_path = body1_rel.GetTargets()
        if len(body1_path) > 0:
            body1_prim = get_prim_at_path(str(body1_path[0]))
            if body1_prim.IsValid():
                body_dict[body1_prim.GetName()] = body1_prim
    return body_dict


def create_joint_xform_list(robot: Robot) -> dict[str, XFormPrim]:
    robot_prim = get_prim_at_path(robot.prim_path)
    joint_prim_dict = get_robot_all_links(robot_prim)
    blacklist = ["Defeatured_2F_85_PAD_OPEN_basestep"]
    joint_xform_list = {
        joint_name: XFormPrim(str(joint_prim.GetPath()))
        for joint_name, joint_prim in joint_prim_dict.items()
        if all([black not in joint_name for black in blacklist])
    }
    return joint_xform_list


def create_tcp_xform_list(robot: Robot, tcp_config: list[dict]) -> list[XFormPrim]:
    tcp_xform_list = []
    for tcp_info in tcp_config:
        tcp = XFormPrim(
            f"{robot.prim_path}/{tcp_info['parent_prim_path']}/{tcp_info['name']}"
        )
        tcp.set_local_pose(tcp_info["position"], tcp_info["orientation"])
        tcp_xform_list.append(tcp)
    return tcp_xform_list
