"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from isaacsim import SimulationApp  # type: ignore
import os
from pathlib import Path
from PIL import Image
import sys
import numpy as np

current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(current_dir)

from genmanip.utils.standalone.file_utils import load_default_config, load_yaml
from genmanip.utils.standalone.utils import setup_logger

simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World  # type: ignore

from genmanip.utils.loader.scene import (
    create_camera_list,
    get_object_list,
    load_world_xform_prim,
)
from genmanip.core.robot.utils import RobotFactory
from genmanip.utils.pointcloud.pointcloud import objectList2meshList
from genmanip.utils.usd_utils.camera_utils import get_src
from genmanip.utils.usd_utils import set_colliders

# 1. setup logger and config
logger = setup_logger()
default_config = load_default_config(
    current_dir=current_dir, config_name="__None__.json", anygrasp_mode="local"
)
ASSETS_DIR = default_config["ASSETS_DIR"]
TEST_USD_NAME = default_config["TEST_USD_NAME"]
TABLE_UID = "aa49db8a801d402dac6cf1579536502c"
camera_data = load_yaml("configs/cameras/fixed_camera_for_debug.yml")

# 2. load scene
scene_xform, uuid = load_world_xform_prim(
    os.path.join(ASSETS_DIR, "scene_usds/base_scenes/base.usda")
)
world = World()
print(f"scene uuid: {uuid}")

# 3. create camera/robot/object/mesh list
embodiment_list = [
    RobotFactory.build(
        "manip/franka/panda_hand",
        scene_uid=uuid,
        default_config=default_config,
        robot_config={},
    )
]
camera_list = create_camera_list(camera_data, uuid)
object_list = get_object_list(uuid, scene_xform, TABLE_UID)
meshDict = objectList2meshList(object_list)
for obj in object_list.values():
    set_colliders(obj.prim_path, "convexDecomposition")

# 4. reset world
world.reset()

# 5. camera warmup
while any(
    camera._custom_annotators["distance_to_image_plane"] is not None
    and get_src(camera, "depth") is None
    for camera in camera_list.values()
):
    world.step()

# 6. save image
print("rendering before save image...")
for _ in range(10):
    world.step()
Path("tmp").mkdir(parents=True, exist_ok=True)
image = Image.fromarray(np.array(get_src(camera_list["obs_camera"], "rgb")))
image.save("tmp/test.png")
print("image saved to tmp/test.png")

# 7. close simulation app
simulation_app.close()
