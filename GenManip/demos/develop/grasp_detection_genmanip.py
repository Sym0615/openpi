import argparse
import av
from filelock import SoftFileLock
import json
import os
from pathlib import Path
import random
import sys
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
args = parser.parse_args()

if args.debug:
    print("Debug mode activated.")

current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(current_dir)

from isaacsim import SimulationApp  # type: ignore

simulation_app = SimulationApp({"headless": True})  # False

from omni.isaac.core import World  # type: ignore
from omni.isaac.core.utils.prims import delete_prim  # type: ignore

from genmanip.core.robot.utils import RobotFactory
from genmanip.demogen.random_place.random_place import (
    place_object_to_object_by_relation,
)
from genmanip.extensions.skills.default.pick.utils import (
    prepare_grasp_motion_planning_payload,
)
from genmanip.extensions.skills.default.pick_and_place.utils import (
    adjust_grasp_by_embodiment,
)
from genmanip.utils.annotation.object_pool import ObjectPool
from genmanip.utils.anygrasp.anygrasp import get_init_grasp
from genmanip.utils.loader.scene import (
    create_camera_list,
    get_object_list,
    load_world_xform_prim,
)
from genmanip.utils.pointcloud.pointcloud import (
    get_current_meshList,
    get_mesh_info_by_load,
    objectList2meshList,
)
from genmanip.utils.standalone.file_utils import load_yaml, load_default_config
from genmanip.utils.standalone.pc_utils import compute_mesh_bbox, compute_aabb_lwh
from genmanip.utils.standalone.utils import setup_logger
from genmanip.utils.usd_utils import (
    add_usd_to_world,
    get_src,
    resize_object,
    resize_object_by_lwh,
    set_camera_look_at,
    set_colliders,
    setup_physics_scene,
)


UID_BLACKLIST = [
    "006d1922549f4f83a87158c46c8f8ea8",
    "266f3c4b79c94c1cad910c063495c187",
    "4129518d0a5d41e7bdab5b42ffd8faec",
    "72ac65ca62714856856370b95e6aa0ed",
    "042d2d9e0dde4a5cb580d87540d9d643",
    "29472731647d49eebccc6b5f8db95724",
    "44eef02ab604448a85423d810fda46f4",
    "04527c55da77419c8d23ef9720e52d4d",
    "2d012e643288403baa8f1c145779ba40",
    "4967c6aebe2f483caef9bc22ca17ef26",
    "056e0c896ce34f13b92dca2c39de44d0",
    "2f4a48252d8546da8a9738540e84aedf",
    "4b32a7bec1f84ac596ace30455bd48eb",
    "0c94ab9eda8d43079f3bfb773cf10bda",
    "331c237cc2044d87977cd3d3e9deabbd",
    "565c2da576404808b1d94dd7e8cbfce3",
    "1683e4db051847cb8a5a38786e071d2c",
    "3886f505abe5419fbb774f6ef2d24717",
    "5ffd0aca6eb4404a925235ad79becac1",
    "1ce07fffeaf74071a5097e6a075d9e0f",
    "3e64c0f124594553a9a50ada31fab4d6",
    "60fe872504de408ab7bf0f97f49ff96b",
    "9f8958d54d96451d911c75f65ff072da",
    "b2496a1228354ec3bfd86f83337bef26",
    "9d65ca34fa3d432b9372c9cdb60a9a00",
    "75a7a3a8146849dea698cfd04f73482b",
    "bcd322567e4447b383773e4465004786",
    "928428f75c764eaa93017161aa6900f2",
    "9d5ae79a89e74d29a362db4f6d55edf4",
    "7f526263e0ae48308447f5f82e1df9f8",
    "9d1ad8b8512040f5a200bc3f72577d8a",
    "b038c39e626a4d28b7ce90c11ebc01f5",
    "80f48376a8104639b87aea73786a9cb7",
]

simulation_app._carb_settings.set("/physics/cooking/ujitsoCollisionCooking", False)

logger = setup_logger()
default_config = load_default_config(
    current_dir=current_dir, config_name="__None__.json", anygrasp_mode="local"
)
ASSETS_DIR = default_config["ASSETS_DIR"]
TEST_USD_NAME = default_config["TEST_USD_NAME"]
TABLE_UID = "aa49db8a801d402dac6cf1579536502c"
camera_data = load_yaml("configs/cameras/fixed_camera_for_debug.yml")
scene_xform, uuid = load_world_xform_prim(
    os.path.join(ASSETS_DIR, "scene_usds/base_scenes/base.usda")
)
world = World()
setup_physics_scene()
camera_list = create_camera_list(camera_data, uuid)
robot_config = {
    "type": "franka",
    "config": {"gripper_type": "robotiq"},
    "default_joint_positions": [
        0.0,
        -0.785,
        0.0,
        -2.356,
        0.0,
        1.57079,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
}
robot_list = [
    RobotFactory.build(
        "manip/franka/panda_hand",
        scene_uid=uuid,
        default_config=default_config,
        robot_config={},
    )
]
for robot in robot_list:
    robot = world.scene.add(robot.robot)
object_list = get_object_list(uuid, scene_xform, TABLE_UID)
meshDict = objectList2meshList(object_list)
for obj in object_list.values():
    set_colliders(obj.prim_path, "convexDecomposition")
world.reset()
robot_list[0].initialize()
robot_list[0].set_planner(world, current_dir)
while get_src(camera_list["camera1"], "depth") is None:
    world.step()
default_joint_positions = robot_config["default_joint_positions"]
usd_list = os.listdir(os.path.join(ASSETS_DIR, "object_usds/objaverse_usd/"))
usd_list = [usd_path for usd_path in usd_list if usd_path.endswith(".usd")]
usd_list.sort()
print(f"processing {len(usd_list)} objects")
object_pool = ObjectPool(
    os.path.join(
        current_dir,
        "assets/objects/objaverse_annotation_refined_pnp.pickle",
    )
)
result_dir = Path(os.path.join(current_dir, "saved/filtered_objects"))
result_dir.mkdir(parents=True, exist_ok=True)
log_lock = SoftFileLock(str(result_dir / "log.lock"))
log_file = result_dir / "log.json"
with log_lock:
    if not log_file.exists():
        with open(log_file, "w") as f:
            json.dump({}, f)
for usd_path in usd_list:
    uid = usd_path.split(".")[0]
    try:
        process_lock = SoftFileLock(str(result_dir / f"{uid}.lock"), timeout=0)
        with process_lock:
            with log_lock:
                with open(log_file, "r") as f:
                    log_data = json.load(f)
            if uid in log_data:
                continue
            if uid in UID_BLACKLIST:
                print(f"Skipping {uid} because it is in the blacklist")
                with open(log_file, "r") as f:
                    log_data = json.load(f)
                log_data[uid] = {"is_success": "blacklisted", "scale": "blacklisted"}
                with open(log_file, "w") as f:
                    json.dump(log_data, f)
                continue
            object_info = object_pool.get_object_info(uid)
            if object_info is None:
                print(f"Skipping {uid} because it is not in the object pool")
                with open(log_file, "r") as f:
                    log_data = json.load(f)
                log_data[uid] = {
                    "is_success": "not_in_object_pool",
                    "scale": "not_in_object_pool",
                }
                with open(log_file, "w") as f:
                    json.dump(log_data, f)
                continue
            usd_path = os.path.join(ASSETS_DIR, "object_usds/objaverse_usd/", usd_path)
            object_list[uid] = add_usd_to_world(
                asset_path=usd_path,
                prim_path=f"/World/{uuid}/obj_{uid}",
                name=f"obj_{uid}",
                translation=None,
                orientation=[0.5, 0.5, 0.5, 0.5],
                scale=[0.01, 0.01, 0.01],  # [1.0, 1.0, 1.0],
                add_rigid_body=True,
                add_colliders=True,
                collision_approximation="convexDecomposition",
            )
            print(f"Finish adding object {uid}")
            meshDict = objectList2meshList(
                object_list,
                os.path.join(
                    default_config["ASSETS_DIR"],
                    "mesh_data",
                    "grasp_detection",
                ),
            )
            meshlist = get_current_meshList(object_list, meshDict)
            scale = random.uniform(
                object_info["scale"][0],
                object_info["scale"][1],
            )
            scale = np.clip(scale, 0.05, 0.15)
            resize_object(
                object_list[uid],
                scale,
                meshlist[uid],
            )
            mesh_info = get_mesh_info_by_load(
                object_list[uid],
                os.path.join(
                    default_config["ASSETS_DIR"],
                    "mesh_data",
                    "grasp_detection",
                    f"{uid}.obj",
                ),
            )
            if mesh_info is not None:
                meshDict[uid] = mesh_info
            robot_list[0].set_joint_positions(default_joint_positions)
            meshlist = get_current_meshList(object_list, meshDict)
            min_thickness = 0.06
            aabb = compute_mesh_bbox(meshlist[uid])
            if np.min(compute_aabb_lwh(aabb)) > min_thickness:
                l, w, h = compute_aabb_lwh(aabb)
                min_thickness_ratio = min_thickness / np.min([l, w])
                min_thickness_ratio = max(
                    min_thickness_ratio, min_thickness / np.min([l, w])
                )
                l *= min_thickness_ratio
                w *= min_thickness_ratio
                h *= min_thickness_ratio
                resize_object_by_lwh(
                    object_list[uid],
                    l=l,
                    w=w,
                    h=h,
                    mesh=meshlist[uid],
                )
                mesh_info = get_mesh_info_by_load(
                    object_list[uid],
                    os.path.join(
                        default_config["ASSETS_DIR"],
                        "mesh_data",
                        "grasp_detection",
                        f"{uid}.obj",
                    ),
                )
                if mesh_info is not None:
                    meshDict[uid] = mesh_info
            IS_OK = place_object_to_object_by_relation(
                uid,
                "00000000000000000000000000000000",
                object_list,
                meshDict,
                "on",
                platform_uid="00000000000000000000000000000000",
            )
            meshlist = get_current_meshList(object_list, meshDict)
            aabb = compute_mesh_bbox(meshlist[uid])
            center = aabb.get_max_bound() + aabb.get_max_bound()
            center = center / 2
            target_center = [0.0, 0.0]
            position = object_list[uid].get_world_pose()[0]
            tar_position = position
            tar_position[:2] = center[:2] - target_center - position[:2]
            object_list[uid].set_world_pose(position=tar_position)
            set_camera_look_at(
                camera_list["camera1"],
                object_list[uid],
                azimuth=180.0,
            )
            set_camera_look_at(
                camera_list["obs_camera"],
                [0.0, 0.0, 1.4],
                elevation=0.0,
                distance=1.0,
            )
            for _ in range(100):
                world.step(render=False)
            for _ in tqdm(range(15)):
                world.step(render=True)
            meshlist = get_current_meshList(object_list, meshDict)
            try:
                init_grasp = get_init_grasp(
                    camera_list["camera1"],
                    meshlist[uid],
                    address=default_config["ANYGRASP_ADDR"],
                )
            except (RuntimeError, ValueError, TypeError, KeyError, IndexError, AssertionError) as e:
                print(e)
                delete_prim(f"/World/{uuid}/obj_{uid}")
                object_list.pop(uid)
                meshDict.pop(uid)
                with log_lock:
                    with open(log_file, "r") as f:
                        log_data = json.load(f)
                    log_data[uid] = {"is_success": False, "scale": scale}
                    with open(log_file, "w") as f:
                        json.dump(log_data, f)
                continue

            init_grasp = adjust_grasp_by_embodiment(init_grasp, robot_list[0])
            action_list = prepare_grasp_motion_planning_payload(
                init_grasp, steps=30, padding=0.0
            )
            world = World()
            paths = []
            object_position = object_list[uid].get_world_pose()[0]
            image_list = []
            for idx, target in tqdm(enumerate(action_list)):
                results = robot_list[0].plan_pose(
                    (target["translation"], target["orientation"]),
                    robot_list[0].robot.get_joints_state(),
                )
                actions = []
                if results is not None:
                    for res in results:
                        if target["grasp"]:
                            actions.append(
                                np.concatenate(
                                    [
                                        res,
                                        robot_list[0].gripper_close,
                                    ]
                                ).tolist()
                            )
                        else:
                            actions.append(
                                np.concatenate(
                                    [res, robot_list[0].gripper_open]
                                ).tolist()
                            )
                    if idx == 1:
                        for _ in range(13):
                            actions.append(
                                np.concatenate(
                                    [
                                        actions[-1][:7],
                                        robot_list[0].gripper_close,
                                    ]
                                ).tolist()
                            )
                while actions:
                    action = actions.pop(0)
                    robot_list[0].robot_view.set_joint_position_targets(action)
                    world.step(render=True)
                    image_list.append(get_src(camera_list["obs_camera"], "rgb"))
            for _ in range(100):
                world.step(render=False)
            after_object_position = object_list[uid].get_world_pose()[0]
            is_success = False
            if (
                after_object_position[2] - object_position[2] > 0.1
                and abs(after_object_position[0] - object_position[0]) < 0.2
                and abs(after_object_position[1] - object_position[1]) < 0.2
            ):
                is_success = True
                print(f"Success {uid}")
            else:
                print(f"Failed {uid}")
            record_dir = os.path.join(current_dir, "tmp/record_grasp")
            Path(record_dir).mkdir(parents=True, exist_ok=True)
            if len(image_list) > 0:
                video_path = os.path.join(
                    record_dir, f"grasp_detection_genmanip_{uid}_{is_success}.mp4"
                )
                container = av.open(video_path, mode="w")
                stream = container.add_stream("h264", rate=30)
                stream.width = image_list[0].shape[1]
                stream.height = image_list[0].shape[0]
                stream.pix_fmt = "yuv420p"
                for image in image_list:
                    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
                print(
                    f"Saved video to {f'tmp/record_grasp/grasp_detection_genmanip_{uid}_{is_success}.mp4'}"
                )
            else:
                print("No image")
            with log_lock:
                with open(log_file, "r") as f:
                    log_data = json.load(f)
                log_data[uid] = {"is_success": is_success, "scale": scale}
                with open(log_file, "w") as f:
                    json.dump(log_data, f)
                delete_prim(f"/World/{uuid}/obj_{uid}")
                object_list.pop(uid)
                meshDict.pop(uid)
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, IndexError, AssertionError) as e:
        if uid in object_list and uid in meshDict:
            delete_prim(f"/World/{uuid}/obj_{uid}")
            object_list.pop(uid)
            meshDict.pop(uid)
        print(e)
        continue
simulation_app.close()
