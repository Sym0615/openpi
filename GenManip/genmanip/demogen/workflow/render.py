from datetime import datetime
from filelock import SoftFileLock
import gc
import logging
import json
import cv2
import numpy as np
import os
import random
import sys
from tqdm import tqdm
import traceback
import uuid

from omni.isaac.core.prims import XFormPrim  # type: ignore[import-untyped]

from genmanip.core.scene.scene import Scene
from genmanip.demogen.recoder.render_recorder import RenderRecorder
from genmanip.demogen.recoder.utils import parse_planning_result
from genmanip.utils.loader.domain_randomization import (
    random_texture,
    load_scene_as_background,
)
from genmanip.utils.loader.scene import (
    clear_scene,
    recovery_scene_render,
)
from genmanip.utils.pointcloud.pointcloud import (
    meshDict2pointCloudDict,
    get_current_pointCloutList,
    objectList2meshList,
)
from genmanip.utils.standalone.file_utils import (
    load_default_config,
    load_dict_from_pkl,
    load_task_config,
    make_dir,
)
from genmanip.utils.standalone.utils import (
    ColorFormatter,
    parse_demogen_config,
    setup_logger,
)
from genmanip.core.scene.scene_config import SceneConfig
from genmanip.utils.usd_utils import (
    create_joint_xform_list,
    remove_colliders,
    set_camera_look_at,
)
from genmanip.utils.usd_utils.camera_utils import get_src
from genmanip.utils.standalone.version_utils import process_archived_config


class RenderWorkflow:
    def __init__(self, args, simulation_app, current_dir):
        self.args = args
        self.simulation_app = simulation_app
        self.current_dir = current_dir
        self.logger = setup_logger()
        log_dir = os.path.join(current_dir, "logs", "render")
        mac_addr = uuid.getnode()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S_%f")
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, f"{mac_addr}_{timestamp}.log")

        logger_name = "RenderWorkflow"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            # ---- file handler (no color) ----
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            file_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            fh.setFormatter(file_fmt)
            self.logger.addHandler(fh)

            # ---- console handler (color) ----
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            color_fmt = ColorFormatter("%(asctime)s [%(levelname)s] %(message)s")
            ch.setFormatter(color_fmt)
            self.logger.addHandler(ch)

            self.logger.propagate = False

        self.logger.info("===== Render Workflow started =====")

        self.default_config = load_default_config(
            self.current_dir, "__None__.json", "local" if self.args.local else "default"
        )
        self.config = load_task_config(args.config)
        self.demogen_config_list = parse_demogen_config(self.config)
        if self.args.process_room_randomization:
            self.demogen_config_list = [
                self._process_room_randomization(demogen_config)
                for demogen_config in self.demogen_config_list
            ]
            self.render_limit = 30
        else:
            self.render_limit = None
        self.render_cnt = 0

    def run(self):
        for demogen_config in self.demogen_config_list:
            try:
                self._process_task(demogen_config)
            except (OSError, RuntimeError, ValueError, TypeError) as e:
                self.logger.error(
                    f"Critical error in task {demogen_config.get('task_name')}: {e}"
                )
                continue

        self.simulation_app.close()

    def _process_task(self, demogen_config):
        demogen_config = process_archived_config(demogen_config)
        scene_config = SceneConfig(**demogen_config)
        task_name = scene_config.task_name

        make_dir(
            os.path.join(self.default_config["DEMONSTRATION_DIR"], task_name, "render")
        )

        scene = self._initialize_scene(scene_config)

        for object in scene.object_list.values():
            remove_colliders(object.prim_path)

        traj_root = os.path.join(
            self.default_config["DEMONSTRATION_DIR"], task_name, "trajectory"
        )
        if not os.path.exists(traj_root):
            self.logger.warning(f"No trajectory directory found for {task_name}")
            return

        dir_list = os.listdir(traj_root)
        self.logger.info(f"rendering {len(dir_list)} trajectories for task {task_name}")

        for dir_name in dir_list:
            self._process_single_trajectory(scene, scene_config, dir_name)

        clear_scene(scene, scene_config, self.current_dir)

    def _initialize_scene(self, scene_config: SceneConfig):
        scene = Scene(scene_config)
        scene.initialize(
            self.default_config,
            physics_dt=1 / 600000.0,
            rendering_dt=1 / 600000.0,
            is_render=True,
            save_pointcloud=self.args.save_pointcloud,
        )
        scene.post_initialize()
        if scene_config.domain_randomization.random_environment.room_randomization:
            with open(
                os.path.join(
                    self.default_config["ASSETS_DIR"], "miscs/scene_list.json"
                ),
                "r",
            ) as f:
                scene_list = json.load(f)

            scene_info = random.choice(scene_list)
            load_scene_as_background(
                scene_info,
                self.default_config["ASSETS_DIR"],
                scene.uuid,
                scene_config.table_uid,
            )
        return scene

    def _process_single_trajectory(self, scene, scene_config: SceneConfig, dir_name):
        task_name = scene_config.task_name
        traj_dir = os.path.join(
            self.default_config["DEMONSTRATION_DIR"], task_name, "trajectory", dir_name
        )
        render_dir = os.path.join(
            self.default_config["DEMONSTRATION_DIR"], task_name, "render", dir_name
        )

        if not os.path.isdir(traj_dir):
            return
        if os.path.exists(render_dir):
            self.logger.info(f"skip {dir_name} because it is already rendered")
            return

        lock_file = os.path.join(
            self.default_config["DEMONSTRATION_DIR"],
            task_name,
            "render",
            f"render_{dir_name}_soft.lock",
        )
        lock = SoftFileLock(lock_file, timeout=0)

        try:
            executed = False
            with lock:
                if os.path.isdir(traj_dir) and not os.path.exists(render_dir):
                    make_dir(render_dir)
                else:
                    raise ValueError(
                        f"render directory {dir_name} already exists or traj missing"
                    )
                self._execute_rendering(
                    scene, scene_config, dir_name, traj_dir, render_dir
                )

                executed = True

                if executed and os.path.exists(lock_file):
                    os.remove(lock_file)
                    self.render_cnt += 1
                    if (
                        self.render_limit is not None
                        and self.render_cnt >= self.render_limit
                    ):
                        self.simulation_app.close()

        except (
            FileNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
        ) as e:
            self.logger.error(f"error in rendering {dir_name}: {e}")
            self.logger.error(traceback.format_exc())

    def _execute_rendering(
        self,
        scene: Scene,
        scene_config: SceneConfig,
        dir_name: str,
        traj_dir: str,
        render_dir: str,
    ):
        meta_info = load_dict_from_pkl(os.path.join(traj_dir, "meta_info.pkl"))
        input_camera_dict = self._setup_cameras(scene)
        recorder = RenderRecorder(
            input_camera_dict,
            scene.robot_list[0],
            meta_info["task_data"]["instruction"],
            log_dir=render_dir,
            task_data=meta_info["task_data"],
            tcp_config=scene.tcp_configs["franka"],
            logger=self.logger,
            record_depth=not self.args.without_depth,
        )
        recovery_scene_render(
            scene,
            meta_info["task_data"],
            self.default_config,
            remove_table=scene_config.domain_randomization.random_environment.room_randomization,
        )
        random_texture(
            scene, self.default_config, scene_config, table_without_collider=True
        )

        data_list = parse_planning_result(
            dir_name,
            self.default_config,
            scene_config.task_name,
            scene.object_list,
            scene.robot_list[0].embodiment_name,
        )
        if not data_list:
            raise ValueError(f"No planning data found for trajectory: {dir_name}")

        if self.args.downsample > 1:
            data_list = data_list[:: self.args.downsample]

        has_joint_world_pose = "joint_world_pose" in data_list[0]
        joint_xform_list = None
        if has_joint_world_pose:
            joint_xform_list = create_joint_xform_list(scene.robot_list[0].robot)

        def create_articulation_part_xform_list(data_frame: dict):
            articulation_part_xform_list = {}
            if not "articulation_info" in data_frame:
                return articulation_part_xform_list
            for articulation_id, articulation in data_frame[
                "articulation_info"
            ].items():
                articulation_part_xform_list[articulation_id] = XFormPrim(
                    articulation["prim_path"]
                )
                articulation_part_xform_list[articulation_id].set_world_pose(
                    articulation["position"], articulation["orientation"]
                )
                articulation_part_xform_list[articulation_id].set_local_scale(
                    articulation["scale"]
                )
            return articulation_part_xform_list

        articulation_part_xform_list = create_articulation_part_xform_list(data_list[0])

        self._warmup_simulation(
            scene,
            data_list[0],
            has_joint_world_pose,
            joint_xform_list,
            articulation_part_xform_list,
        )

        self.logger.info(
            f"rendering data with length: {len(data_list)} in {'render' if has_joint_world_pose else 'step'} mode"
        )

        pointDict, pointJointDict = None, None
        if self.args.save_pointcloud:
            scene.object_list["00000000000000000000000000000000"] = XFormPrim(
                prim_path=f"/World/{scene.uuid}/table",
            )
            if not scene.object_list[
                "00000000000000000000000000000000"
            ].prim.IsActive():
                scene.object_list["00000000000000000000000000000000"].prim.SetActive(
                    True
                )
            scene.cache_library.mesh_dict = objectList2meshList(scene.object_list)
            pointDict = meshDict2pointCloudDict(scene.cache_library.mesh_dict)
            if has_joint_world_pose:
                meshJointDict = objectList2meshList(joint_xform_list)  # type: ignore[attr-defined]
                pointJointDict = meshDict2pointCloudDict(meshJointDict)

        for idx, data in tqdm(enumerate(data_list)):
            if self.args.add_cycle_camera:
                self._set_cycle_camera(scene, idx * 1.0)
            self._render_single_frame(
                scene,
                data,
                has_joint_world_pose,
                joint_xform_list,
                articulation_part_xform_list,
                recorder,
                pointDict,
                pointJointDict,
            )
            if idx == 0:
                self._save_overhead_camera(render_dir, input_camera_dict)

            if self.args.render_first_frame:
                break

        recorder.save()
        recorder.release(trim=True)
        del recorder
        gc.collect()

    def _set_cycle_camera(self, scene, azimuth=0.0):
        set_camera_look_at(
            scene.camera_list["camera1"],
            np.array([0.0, 0.0, 1.1]),
            distance=1.0,
            elevation=30.0,
            azimuth=azimuth,
        )

    def _setup_cameras(self, scene):
        input_camera_dict = scene.camera_list.copy()
        if self.args.add_random_position_camera:
            random_azimuth = np.random.uniform(-150, 150)
            random_elevation = np.random.uniform(30, 50)
            distance = np.random.uniform(0.7, 1.2)
            set_camera_look_at(
                input_camera_dict["camera1"],
                np.array([0, 0, 1.1]),
                distance=distance,
                azimuth=random_azimuth,
                elevation=random_elevation,
            )
            if self.args.add_cycle_camera:
                self._set_cycle_camera(scene)
        else:
            if "camera1" in input_camera_dict:
                input_camera_dict.pop("camera1")
        return input_camera_dict

    def _warmup_simulation(
        self,
        scene,
        initial_data,
        has_joint_world_pose,
        joint_xform_list,
        articulation_part_xform_list,
    ):
        for _ in range(10):
            self._apply_state(
                scene,
                initial_data,
                has_joint_world_pose,
                joint_xform_list,
                articulation_part_xform_list,
            )
            if self.args.add_cycle_camera:
                self._set_cycle_camera(scene)
            if has_joint_world_pose:
                scene.world.step()
            else:
                scene.world.render()

    def _apply_state(
        self,
        scene,
        data,
        has_joint_world_pose,
        joint_xform_list,
        articulation_part_xform_list,
    ):
        # Robot state
        if has_joint_world_pose:
            for joint_name, joint_xform in joint_xform_list.items():
                joint_xform.set_world_pose(*data["joint_world_pose"][joint_name])
        else:
            scene.robot_list[0].robot.set_joint_positions(data["qpos"])

        # Articulation state
        for articulation_id, articulation in articulation_part_xform_list.items():
            articulation.set_world_pose(
                data["articulation_info"][articulation_id]["position"],
                data["articulation_info"][articulation_id]["orientation"],
            )
            articulation.set_local_scale(
                data["articulation_info"][articulation_id]["scale"]
            )

        # Object state
        for key in scene.object_list:
            if key == "00000000000000000000000000000000":
                continue
            if key not in data["obj_info"]:
                continue
            scene.object_list[key].set_world_pose(
                data["obj_info"][key]["position"],
                data["obj_info"][key]["orientation"],
            )
            scene.object_list[key].set_local_scale(data["obj_info"][key]["scale"])

    def _render_single_frame(
        self,
        scene,
        data,
        has_joint_world_pose,
        joint_xform_list,
        articulation_part_xform_list,
        recorder,
        pointDict,
        pointJointDict,
    ):
        self._apply_state(
            scene,
            data,
            has_joint_world_pose,
            joint_xform_list,
            articulation_part_xform_list,
        )
        if has_joint_world_pose:
            scene.world.render()
        else:
            scene.world.step()

        if self.args.high_quality:
            for _ in range(50):
                scene.world.render()
                scene.world.get_observations()

        pointcloud = None
        if self.args.save_pointcloud:
            pointcloud = get_current_pointCloutList(scene.object_list, pointDict)
            if has_joint_world_pose:
                pointcloud.update(
                    {
                        "robot": get_current_pointCloutList(
                            joint_xform_list, pointJointDict
                        )
                    }
                )

        recorder.load_dynamic_info(
            data["obj_info"],
            data["action"],
            data["qpos"],
            data["qvel"],
            data["gripper_close"],
            data["name"],
            data["base_motion"],
            pointcloud=pointcloud,
        )

    def _save_overhead_camera(
        self, render_dir: str, input_camera_dict: dict[str, object]
    ) -> None:
        preview_path = os.path.join(render_dir, "overhead_camera.jpg")
        if os.path.exists(preview_path):
            return

        preferred_cameras = ["overlook_camera", "obs_camera", "top_camera"]
        candidate_names = []
        for camera_name in preferred_cameras:
            if camera_name in input_camera_dict:
                candidate_names.append(camera_name)
        for camera_name in sorted(input_camera_dict.keys()):
            if camera_name not in candidate_names:
                candidate_names.append(camera_name)

        for camera_name in candidate_names:
            rgb = get_src(input_camera_dict[camera_name], "rgb")
            if not isinstance(rgb, np.ndarray) or rgb.size == 0:
                continue
            if rgb.ndim != 3 or rgb.shape[-1] < 3:
                continue
            # get_src returns RGB; cv2 expects BGR.
            bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(preview_path, bgr)
            if ok:
                return

        self.logger.warning("Failed to save preview image in %s", render_dir)

    def _process_room_randomization(self, demogen_config):
        demogen_config["domain_randomization"]["random_environment"]["has_wall"] = False
        demogen_config["domain_randomization"]["random_environment"][
            "robot_base_position"
        ] = False
        demogen_config["domain_randomization"]["random_environment"][
            "robot_eepose"
        ] = False
        demogen_config["domain_randomization"]["random_environment"][
            "table_texture"
        ] = False
        demogen_config["domain_randomization"]["random_environment"][
            "table_type"
        ] = False
        demogen_config["domain_randomization"]["random_environment"][
            "wall_texture"
        ] = False
        demogen_config["domain_randomization"]["random_environment"]["hdr"] = True
        demogen_config["domain_randomization"]["random_environment"][
            "room_randomization"
        ] = True
        return demogen_config
