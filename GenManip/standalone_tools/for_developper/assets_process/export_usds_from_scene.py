from isaacsim import SimulationApp  # type: ignore[import-untyped]

simulation_app = SimulationApp({"headless": True})  # False

from functools import partial
from pathlib import Path
from pxr import Sdf, Gf, Usd, UsdGeom, UsdUI  # type: ignore
from typing import List
import os
from omni.isaac.core.prims import XFormPrim  # type: ignore
import omni.usd  # type: ignore
import numpy as np
import open3d as o3d
import coacd
import trimesh
from scipy.spatial.transform import Rotation as R


def __set_xform_prim_transform(prim: UsdGeom.Xformable, transform: Gf.Matrix4d):
    prim = UsdGeom.Xformable(prim)
    _, _, scale, rot_mat, translation, _ = transform.Factor()
    angles = rot_mat.ExtractRotation().Decompose(
        Gf.Vec3d.ZAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.XAxis()
    )
    rotation = Gf.Vec3f(angles[2], angles[1], angles[0])

    for xform_op in prim.GetOrderedXformOps():
        attr = xform_op.GetAttr()
        prim.GetPrim().RemoveProperty(attr.GetName())
    prim.ClearXformOpOrder()

    UsdGeom.XformCommonAPI(prim).SetTranslate(translation)
    UsdGeom.XformCommonAPI(prim).SetRotate(rotation)
    UsdGeom.XformCommonAPI(prim).SetScale(Gf.Vec3f(scale[0], scale[1], scale[2]))


def export(path: str, prims: List[Usd.Prim]):
    """Export prim to external USD file"""
    filename = Path(path).stem

    # TODO: stage.Flatten() is extreamly slow
    stage = omni.usd.get_context().get_stage()
    source_layer = stage.Flatten()
    target_layer = Sdf.Layer.CreateNew(path)
    target_stage = Usd.Stage.Open(target_layer)
    axis = UsdGeom.GetStageUpAxis(stage)
    UsdGeom.SetStageUpAxis(target_stage, axis)

    # All prims will be put under /Root
    if len(prims) > 1:
        root_path = Sdf.Path.absoluteRootPath.AppendChild("Root")
        UsdGeom.Xform.Define(target_stage, root_path)
    else:
        root_path = Sdf.Path.absoluteRootPath

    keep_transforms = len(prims) > 1

    center_point = Gf.Vec3d(0.0)
    transforms = []
    if keep_transforms:
        bound_box = Gf.BBox3d()
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_]
        )
        for prim in prims:
            xformable = UsdGeom.Xformable(prim)
            if xformable:
                local_bound_box = bbox_cache.ComputeWorldBound(prim)
                transforms.append(
                    xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                )
                bound_box = Gf.BBox3d.Combine(bound_box, local_bound_box)
            else:
                transforms.append(None)

        center_point = bound_box.ComputeCentroid()
    else:
        transforms.append(Gf.Matrix4d(1.0))

    for i in range(len(transforms)):
        if transforms[i]:
            transforms[i] = transforms[i].SetTranslateOnly(
                transforms[i].ExtractTranslation() - center_point
            )

    # Set default prim name
    if len(prims) > 1:
        target_layer.defaultPrim = root_path.name
        # target_layer.SetDefaultPrim(prims[0].GetPath().name)

        for i in range(len(prims)):
            source_path = prims[i].GetPath()
            if len(prims) > 1 and transforms[i]:
                target_xform_path = root_path.AppendChild(source_path.name)
                target_xform_path = Sdf.Path(
                    omni.usd.get_stage_next_free_path(
                        target_stage, target_xform_path, False
                    )
                )
                target_xform = UsdGeom.Xform.Define(target_stage, target_xform_path)
                __set_xform_prim_transform(target_xform, transforms[i])
                target_path = target_xform_path.AppendChild(source_path.name)
            else:
                target_path = root_path.AppendChild(source_path.name)
                target_path = Sdf.Path(
                    omni.usd.get_stage_next_free_path(target_stage, target_path, False)
                )

            all_external_references = set([])

            def on_prim_spec_path(root_path, prim_spec_path):
                if prim_spec_path.IsPropertyPath():
                    return

                if prim_spec_path == Sdf.Path.absoluteRootPath:
                    return

                prim_spec = source_layer.GetPrimAtPath(prim_spec_path)
                if not prim_spec or not prim_spec.HasInfo(Sdf.PrimSpec.ReferencesKey):
                    return

                op = prim_spec.GetInfo(Sdf.PrimSpec.ReferencesKey)
                items = []
                items = op.ApplyOperations(items)

                for item in items:
                    if not item.primPath.HasPrefix(root_path):
                        all_external_references.add(item.primPath)

            # Traverse the source prim tree to find all references that are outside of the source tree.
            source_layer.Traverse(source_path, partial(on_prim_spec_path, source_path))

            # Copy dependencies
            for path in all_external_references:
                Sdf.CreatePrimInLayer(target_layer, path)
                Sdf.CopySpec(source_layer, path, target_layer, path)

            Sdf.CreatePrimInLayer(target_layer, target_path)
            Sdf.CopySpec(source_layer, source_path, target_layer, target_path)

            prim = target_stage.GetPrimAtPath(target_path)
            if transforms[i]:
                __set_xform_prim_transform(prim, Gf.Matrix4d(1.0))

            # Edit UI info of compound
            spec = target_layer.GetPrimAtPath(target_path)
            attributes = spec.attributes

            if UsdUI.Tokens.uiDisplayGroup not in attributes:
                attr = Sdf.AttributeSpec(
                    spec, UsdUI.Tokens.uiDisplayGroup, Sdf.ValueTypeNames.Token
                )
                attr.default = "Material Graphs"

            if UsdUI.Tokens.uiDisplayName not in attributes:
                attr = Sdf.AttributeSpec(
                    spec, UsdUI.Tokens.uiDisplayName, Sdf.ValueTypeNames.Token
                )
                attr.default = target_path.name

            if "ui:order" not in attributes:
                attr = Sdf.AttributeSpec(spec, "ui:order", Sdf.ValueTypeNames.Int)
                attr.default = 1024
    else:
        source_path = prims[0].GetPath()

        target_path = root_path.AppendChild(source_path.name)

        target_path = Sdf.Path(
            omni.usd.get_stage_next_free_path(target_stage, target_path, False)
        )
        target_layer.defaultPrim = source_path.name

        # all_external_references = set([])

        # def on_prim_spec_path(root_path, prim_spec_path):
        #     if prim_spec_path.IsPropertyPath():
        #         return

        #     if prim_spec_path == Sdf.Path.absoluteRootPath:
        #         return

        #     prim_spec = source_layer.GetPrimAtPath(prim_spec_path)
        #     if not prim_spec or not prim_spec.HasInfo(Sdf.PrimSpec.ReferencesKey):
        #         return

        #     op = prim_spec.GetInfo(Sdf.PrimSpec.ReferencesKey)
        #     items = []
        #     items = op.ApplyOperations(items)

        #     for item in items:
        #         if not item.primPath.HasPrefix(root_path):
        #             all_external_references.add(item.primPath)

        # # Traverse the source prim tree to find all references that are outside of the source tree.
        # source_layer.Traverse(source_path, partial(on_prim_spec_path, source_path))

        # Copy dependencies
        # for path in all_external_references:
        #     Sdf.CreatePrimInLayer(target_layer, path)
        #     Sdf.CopySpec(source_layer, path, target_layer, path)

        Sdf.CreatePrimInLayer(target_layer, target_path)
        Sdf.CopySpec(source_layer, source_path, target_layer, target_path)

        prim = target_stage.GetPrimAtPath(target_path)

        # Edit UI info of compound
        spec = target_layer.GetPrimAtPath(target_path)
        attributes = spec.attributes

        if UsdUI.Tokens.uiDisplayGroup not in attributes:
            attr = Sdf.AttributeSpec(
                spec, UsdUI.Tokens.uiDisplayGroup, Sdf.ValueTypeNames.Token
            )
            attr.default = "Material Graphs"

        if UsdUI.Tokens.uiDisplayName not in attributes:
            attr = Sdf.AttributeSpec(
                spec, UsdUI.Tokens.uiDisplayName, Sdf.ValueTypeNames.Token
            )
            attr.default = target_path.name

        if "ui:order" not in attributes:
            attr = Sdf.AttributeSpec(spec, "ui:order", Sdf.ValueTypeNames.Int)
            attr.default = 1024

    # Save
    target_layer.Save()


def move_prim(stage: Usd.Stage, old_path: str, new_path: str):
    old_path = Sdf.Path(old_path)
    new_path = Sdf.Path(new_path)
    old_prim = stage.GetPrimAtPath(old_path)
    if not old_prim or not old_prim.IsValid():
        print(f"[Error] Prim at {old_path} does not exist.")
        return
    if stage.GetPrimAtPath(new_path).IsValid():
        print(f"[Error] Prim at {new_path} already exists.")
        return
    layer = stage.GetEditTarget().GetLayer()
    if not Sdf.CopySpec(layer, old_path, layer, new_path):
        print(f"[Error] Failed to copy spec from {old_path} to {new_path}")
        return
    stage.RemovePrim(old_path)


def add_prim_from_asset(stage: Usd.Stage, prim_path: str, asset_path: str):
    prim_path = Sdf.Path(prim_path)
    if stage.GetPrimAtPath(prim_path).IsValid():
        print(f"[Error] Prim at {prim_path} already exists.")
        return
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(asset_path)


def save_stage_as(stage: Usd.Stage, new_path: str):
    new_layer = Sdf.Layer.CreateNew(new_path)
    if not new_layer:
        return
    stage.Export(new_path)


def delete_prim(stage: Usd.Stage, prim_path: str):
    prim_path = Sdf.Path(prim_path)
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(prim_path)


def print_prim_tree(prim, prefix="", is_last=True):
    RESET = "\033[0m"
    COLORS = {
        "Xform": "\033[94m",
        "Mesh": "\033[95m",
        "Camera": "\033[93m",
        "Light": "\033[96m",
        "Default": "\033[92m",
        "Type": "\033[91m",
    }
    connector = "└── " if is_last else "├── "
    prim_name = prim.GetName()
    prim_type = prim.GetTypeName()
    color = COLORS.get(prim_type, COLORS["Default"])
    color_type = COLORS["Type"]
    colored_name = f"{color}{prim_name}{RESET}"
    colored_type = f"{color_type} ({prim_type}){RESET}" if prim_type else ""
    print(prefix + connector + colored_name + colored_type)
    children = prim.GetChildren()
    count = len(children)
    for i, child in enumerate(children):
        is_last_child = i == count - 1
        extension = "    " if is_last else "│   "
        print_prim_tree(child, prefix + extension, is_last_child)


def _decompose_affine_transform(
    transform_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(transform_matrix, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 matrix, got shape {matrix.shape}")

    translation = matrix[:3, 3].copy()
    linear = matrix[:3, :3].copy()
    scale = np.linalg.norm(linear, axis=0)

    rotation_matrix = np.eye(3, dtype=float)
    non_zero = np.abs(scale) > 1e-12
    if np.any(non_zero):
        rotation_matrix[:, non_zero] = linear[:, non_zero] / scale[non_zero]

    if np.linalg.det(rotation_matrix) < 0 and np.any(non_zero):
        axis_idx = int(np.argmax(np.abs(scale)))
        rotation_matrix[:, axis_idx] *= -1.0
        scale[axis_idx] *= -1.0

    return translation, scale, rotation_matrix


def _resolve_local_transform(
    prim: Usd.Prim,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transform_attr = prim.GetAttribute("xformOp:transform").Get()
    if transform_attr is not None:
        matrix = np.array(transform_attr, dtype=float).T
        return _decompose_affine_transform(matrix)

    translation_attr = prim.GetAttribute("xformOp:translate").Get()
    translation = (
        np.zeros(3, dtype=float)
        if translation_attr is None
        else np.asarray(translation_attr, dtype=float)
    )

    scale_attr = prim.GetAttribute("xformOp:scale").Get()
    scale = (
        np.ones(3, dtype=float)
        if scale_attr is None
        else np.asarray(scale_attr, dtype=float)
    )

    scale_units_resolve = prim.GetAttribute("xformOp:scale:unitsResolve").Get()
    if scale_units_resolve is not None:
        scale = scale * np.asarray(scale_units_resolve, dtype=float)

    orient_attr = prim.GetAttribute("xformOp:orient").Get()
    rotate_xyz = prim.GetAttribute("xformOp:rotateXYZ").Get()
    rotate_zyx = prim.GetAttribute("xformOp:rotateZYX").Get()
    if orient_attr is not None:
        quat_wxyz = np.array(
            [orient_attr.GetReal(), *orient_attr.GetImaginary()], dtype=float
        )
    elif rotate_xyz is not None or rotate_zyx is not None:
        if rotate_xyz is not None:
            seq = "xyz"
            angles = np.asarray(rotate_xyz, dtype=float)
        else:
            seq = "zyx"
            angles = np.asarray(rotate_zyx, dtype=float)
        quat_xyzw = R.from_euler(seq, angles, degrees=True).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    else:
        quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    rotate_x_units_resolve = prim.GetAttribute("xformOp:rotateX:unitsResolve").Get()
    if rotate_x_units_resolve is not None:
        quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
        euler = R.from_quat(quat_xyzw).as_euler("xyz", degrees=True)
        euler[0] += float(rotate_x_units_resolve)
        quat_xyzw = R.from_euler("xyz", euler, degrees=True).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    rotation_matrix = R.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()
    return translation, scale, rotation_matrix


def _apply_local_transform(
    points: list[np.ndarray],
    translation: np.ndarray,
    scale: np.ndarray,
    rotation_matrix: np.ndarray,
) -> list[np.ndarray]:
    if len(points) == 0:
        return []
    points_array = np.asarray(points, dtype=float)
    transformed = (rotation_matrix @ (points_array * scale).T).T + translation
    return [p for p in transformed]


def recursive_parse(prim):
    translation, scale, rotation_matrix = _resolve_local_transform(prim)
    points_total = []
    faceuv_total = []
    normals_total = []
    faceVertexCounts_total = []
    faceVertexIndices_total = []
    mesh_total = []
    children = prim.GetChildren()
    for child in children:
        points, faceuv, normals, faceVertexCounts, faceVertexIndices, mesh_list = (
            recursive_parse(child)
        )
        base_num = len(points_total)
        for idx in faceVertexIndices:
            faceVertexIndices_total.append(base_num + idx)
        faceVertexCounts_total += faceVertexCounts
        faceuv_total += faceuv
        normals_total += normals
        points_total += points
        mesh_total += mesh_list
    if prim.IsA(UsdGeom.Mesh):
        mesh_path = str(prim.GetPath()).split("/")[-1]
        if not mesh_path == "SM_Dummy":
            mesh_total.append(mesh_path)
            points = prim.GetAttribute("points").Get()
            normals = prim.GetAttribute("normals").Get()
            faceVertexCounts = prim.GetAttribute("faceVertexCounts").Get()
            faceVertexIndices = prim.GetAttribute("faceVertexIndices").Get()
            faceuv = prim.GetAttribute("primvars:st").Get()
            if points is None:
                points = []
            if normals is None:
                normals = []
            if faceVertexCounts is None:
                faceVertexCounts = []
            if faceVertexIndices is None:
                faceVertexIndices = []
            if faceuv is None:
                faceuv = []
            normals = [_ for _ in normals]
            faceVertexCounts = [_ for _ in faceVertexCounts]
            faceVertexIndices = [_ for _ in faceVertexIndices]
            faceuv = [_ for _ in faceuv]
            ps = []
            for p in points:
                x, y, z = p
                p = np.array((x, y, z))
                ps.append(p)
            points = ps
            base_num = len(points_total)
            for idx in faceVertexIndices:
                faceVertexIndices_total.append(base_num + idx)
            faceVertexCounts_total += faceVertexCounts
            faceuv_total += faceuv
            normals_total += normals
            points_total += points
    if prim.IsA(UsdGeom.Cube):
        size_attr = prim.GetAttribute("size").Get()
        half_size = float(size_attr) / 2 if size_attr is not None else 0.5
        points = [
            np.array([-half_size, -half_size, -half_size]),
            np.array([half_size, -half_size, -half_size]),
            np.array([half_size, half_size, -half_size]),
            np.array([-half_size, half_size, -half_size]),
            np.array([-half_size, -half_size, half_size]),
            np.array([half_size, -half_size, half_size]),
            np.array([half_size, half_size, half_size]),
            np.array([-half_size, half_size, half_size]),
        ]
        face_vertex_counts = [4, 4, 4, 4, 4, 4]
        face_vertex_indices = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            0,
            4,
            7,
            3,
            1,
            5,
            6,
            2,
            3,
            2,
            6,
            7,
            0,
            1,
            5,
            4,
        ]
        points_total += points
        faceVertexCounts_total += face_vertex_counts
        faceVertexIndices_total += face_vertex_indices
        mesh_total.append(str(prim.GetPath()).split("/")[-1])

    new_points = _apply_local_transform(
        points_total, translation, scale, rotation_matrix
    )
    return (
        new_points,
        faceuv_total,
        normals_total,
        faceVertexCounts_total,
        faceVertexIndices_total,
        mesh_total,
    )


def get_mesh_from_points_and_faces(points, faceVertexCounts, faceVertexIndices):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    triangles = []
    idx = 0
    for count in faceVertexCounts:
        if count == 3:
            triangles.append(faceVertexIndices[idx : idx + 3])
        elif count == 4:
            face_indices = faceVertexIndices[idx : idx + 4]
            triangles.append([face_indices[0], face_indices[1], face_indices[2]])
            triangles.append([face_indices[0], face_indices[2], face_indices[3]])
        elif count > 4:
            face_indices = faceVertexIndices[idx : idx + count]
            for i in range(1, count - 1):
                triangles.append(
                    [face_indices[0], face_indices[i], face_indices[i + 1]]
                )
        idx += count
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh


def get_mesh_from_prim(prim):
    points, faceuv, normals, faceVertexCounts, faceVertexIndices, mesh_total = (
        recursive_parse(prim)
    )
    # points += np.array(prim.GetAttribute("xformOp:transform").Get()[3][0:3])
    mesh = get_mesh_from_points_and_faces(points, faceVertexCounts, faceVertexIndices)
    return mesh


def save_convex_hulls_as_obj(convex_hulls, output_path):
    with open(output_path, "w") as f:
        vertex_offset = 1
        for i, (vertices, faces) in enumerate(convex_hulls):
            f.write(f"o convex_part_{i}\n")
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                f.write(
                    f"f {face[0] + vertex_offset} {face[1] + vertex_offset} {face[2] + vertex_offset}\n"
                )
            vertex_offset += len(vertices)


def run_coacd(mesh_path, output_path):
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)  # type: ignore[attr-defined]
    parts = coacd.run_coacd(mesh)
    save_convex_hulls_as_obj(parts, output_path)
    return parts


def process_single_prim(prim: Usd.Prim, output_path: str):
    if (
        os.path.exists(os.path.join(output_path, prim.GetName() + ".usd"))
        and os.path.exists(os.path.join(output_path, prim.GetName() + ".obj"))
        and os.path.exists(os.path.join(output_path, prim.GetName() + "_coacd.obj"))
    ):
        return
    if (
        prim.GetPath() == "/World/franka"
        or prim.GetPath() == "/World/obj_defaultGroundPlane"
        or prim.GetPath() == "/World/Cube"
    ):
        return
    print(f"Processing {prim.GetPath()}")
    xform = XFormPrim(str(prim.GetPath()), name=prim.GetName())
    xform.set_world_pose([0, 0, 0], [1, 0, 0, 0])
    xform.set_local_scale([1, 1, 1])
    export(os.path.join(output_path, prim.GetName() + ".usd"), [prim])
    print(
        f"Exported {prim.GetName()} to {os.path.join(output_path, prim.GetName() + '.usd')}"
    )
    mesh = get_mesh_from_prim(prim)
    o3d.io.write_triangle_mesh(os.path.join(output_path, prim.GetName() + ".obj"), mesh)
    print(
        f"Exported {prim.GetName()} to {os.path.join(output_path, prim.GetName() + '.obj')}"
    )
    acd_mesh_path = os.path.join(output_path, prim.GetName() + "_coacd.obj")
    run_coacd(os.path.join(output_path, prim.GetName() + ".obj"), acd_mesh_path)
    print(f"Exported {prim.GetName()} to {acd_mesh_path}")
    os.remove(os.path.join(output_path, prim.GetName() + ".obj"))
    print(
        f"Deleted {prim.GetName()} from {os.path.join(output_path, prim.GetName() + '.obj')}"
    )


def load_world_xform_prim(scene_path, scene_prim_path="/World"):
    scene_path = os.path.abspath(scene_path)
    omni.usd.get_context().new_stage()
    omni.usd.get_context().open_stage(scene_path)
    scene_xform = XFormPrim(
        "/World",
        name="World",
    )
    uuid = str(scene_xform.prim.GetAllChildren()[0].GetPath()).split("/")[-1]
    return scene_xform, uuid


def export_usds_from_scene(scene_path: str, output_path: str):
    from omni.isaac.core import World  # type: ignore

    scene_xform, uuid = load_world_xform_prim(scene_path)
    world = World()
    world.reset()
    root = scene_xform.prim
    for child in root.GetChildren():
        process_single_prim(child, output_path)


if __name__ == "__main__":
    export_path = "scene_usds/scene_cup/"
    os.makedirs(export_path, exist_ok=True)
    export_usds_from_scene(
        "saved/assets/scene_usds/IROS_scenes_collected/scene_cup/base_2.usd",
        export_path,
    )
    export_path = "scene_usds/scene_drink/"
    os.makedirs(export_path, exist_ok=True)
    export_usds_from_scene(
        "saved/assets/scene_usds/IROS_scenes_collected/scene_drink/base.usd",
        export_path,
    )
    simulation_app.close()
