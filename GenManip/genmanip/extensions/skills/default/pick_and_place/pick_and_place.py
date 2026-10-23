"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import os
import random
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field
from scipy.spatial.transform import Rotation as R
import torch
from tqdm import tqdm

from curobo.util.trajectory import get_smooth_trajectory

from genmanip.core.robot.base import BaseEmbodiment
from genmanip.core.robot.dualarm_manip import DualArmEmbodiment
from genmanip.core.scene.scene_config import SceneConfig
from genmanip.core.skill.base import BaseSkill, SkillConfig
from genmanip.core.skill.utils import SkillFactory
from genmanip.demogen.random_place.random_place import (
    place_object_to_object_by_relation,
)
from genmanip.demogen.recoder.planning_recorder import PlanningRecorder
from genmanip.extensions.metrics.default.sr_based_genmanip_relationship import (
    check_subgoal_finished_rigid,
)
from genmanip.extensions.skills.default.pick_and_place.utils import (
    adjust_grasp_by_embodiment,
    prepare_motion_planning_payload,
)
from genmanip.utils.anygrasp.anygrasp import get_init_grasp
from genmanip.utils.loader.utils import collect_world_pose_list
from genmanip.utils.pointcloud.pointcloud import get_current_meshList
from genmanip.utils.standalone.transform_utils import (
    adjust_translation_along_quaternion,
    compute_final_pose,
    rot_orientation_by_axis,
)
from genmanip.utils.usd_utils import set_camera_look_at, set_mass
from genmanip.utils.loader.utils import collect_world_pose_list, reset_object_xyz
from genmanip.utils.pointcloud.pointcloud import get_current_pcList_by_meshList
from genmanip.utils.standalone.transform_utils import (
    adjust_translation_along_quaternion,
)

if TYPE_CHECKING:
    from genmanip.core.scene.scene import Scene


class PickAndPlaceConfig(SkillConfig):
    name: str = "pick_and_place"
    obj1_uid: str = Field(..., description="UID of the object to pick")
    obj2_uid: str = Field(..., description="UID of the object to place")
    position: str = Field(..., description="Position of the object to place")
    another_obj2_uid: str | None = Field(
        default=None, description="UID of the another object to place"
    )
    arm: str = Field(
        default="default", description="Arm to use for the pick and place skill"
    )
    ignored_uid: list[str] = Field(
        default_factory=list, description="UIDs of the objects to ignore"
    )
    force_fixed_grasp: bool = Field(default=False, description="Force fixed grasp")
    allow_fixed_grasp: bool = Field(default=False, description="Allow fixed grasp")
    fixed_grasp_config: dict | None = Field(
        default=None, description="Fixed grasp config"
    )
    gripper_rot_config: dict | None = Field(
        default=None,
        description="Rotate grasp orientation after embodiment adjustment",
    )
    fixed_position: bool = Field(default=False, description="Fixed position")
    fixed_position_config: dict | list[dict] | None = Field(
        default=None, description="Fixed position config"
    )
    mesh_top_only: bool = Field(default=False, description="Mesh top only")
    without_platform: bool = Field(default=False, description="Without platform")
    update_planner: bool = Field(default=False, description="Update planner")
    plan_ignored_list: list[str] = Field(
        default_factory=list, description="Ignored list for planner"
    )
    motion_config: dict = Field(default={}, description="Motion config")
    base_motion_list: dict = Field(default={}, description="Robot base move config")
    grasp_idx: list[int] = Field(default=[0], description="Grasp index")


@SkillFactory.register("pick_and_place")
class PickAndPlaceSkill(BaseSkill):
    def __init__(self, config_dict: dict, scene_config: SceneConfig):
        self.config = PickAndPlaceConfig(**config_dict)
        self.scene_config = scene_config
        self.last_action = None

    def execute(
        self, scene: "Scene", recorder: PlanningRecorder, idx: str
    ) -> tuple[bool, str]:
        self._initialize(scene)
        action_meta_info = self._get_meta_info(scene)
        if (
            self.config.update_planner
            or self.scene_config.generation_config.update_planner
        ):
            self._update_planner(scene)
        _, arm = self._record_planning(
            scene,
            recorder,
            scene.robot_list[0],
            action_meta_info,
            idx_name=str(idx),
            smooth=self.scene_config.generation_config.smooth,
            reset_tcp=self.scene_config.generation_config.reset_tcp,
            base_motion_list=self.config.base_motion_list,
        )
        return self._is_done(scene), arm

    def _initialize(self, scene: "Scene"):
        set_mass(scene.object_list[self.config.obj1_uid].prim_path, 0.1)
        set_mass(scene.object_list[self.config.obj2_uid].prim_path, 10.0)

    def _axis_angle_from_obj1_orientation(
        self, obj1_orientation_wxyz: np.ndarray, axis: str
    ) -> float:
        rotation = R.from_quat(np.asarray(obj1_orientation_wxyz)[[1, 2, 3, 0]])
        euler_xyz = rotation.as_euler("xyz", degrees=True)
        axis_map = {"x": 0, "y": 1, "z": 2}
        return float(euler_xyz[axis_map[axis]])

    def _compute_gripper_rotation_angle(
        self, obj1_orientation_wxyz: np.ndarray | None = None
    ) -> tuple[str | None, float | None]:
        cfg = self.config.gripper_rot_config
        if cfg is None:
            return None, None
        axis = cfg.get("axis")
        angle = cfg.get("angle")
        if axis is None or angle is None:
            return None, None
        axis = str(axis).lower()
        angle = float(angle)
        if cfg.get("compensate_obj1_yaw", False):
            if obj1_orientation_wxyz is None:
                raise ValueError("obj1 orientation is required for yaw compensation")
            if axis != "z":
                raise ValueError("compensate_obj1_yaw only supports axis='z'")
            angle += self._axis_angle_from_obj1_orientation(obj1_orientation_wxyz, "z")
        return axis, angle

    def _apply_fixed_grasp_translation(
        self, scene: "Scene", grasp_position: np.ndarray
    ) -> np.ndarray:
        if self.config.fixed_grasp_config is None:
            return grasp_position
        translation = np.array(self.config.fixed_grasp_config["translation"])
        return grasp_position + translation

    def _set_final_object_pose(
        self, scene: "Scene", extra_erosion: float = 0.05
    ) -> int:
        if self.config.position == "top" or self.config.position == "in":
            IS_OK = place_object_to_object_by_relation(
                self.config.obj1_uid,
                self.config.obj2_uid,
                scene.object_list,
                scene.cache_library.mesh_dict,
                "on",
                platform_uid=(
                    "00000000000000000000000000000000"
                    if not self.config.without_platform
                    else None
                ),
                ignored_uid=self.config.ignored_uid,
                extra_erosion=extra_erosion,
                fixed_position=self.config.fixed_position,
                mesh_top_only=self.config.mesh_top_only,
            )
        elif self.config.position == "near":
            IS_OK = place_object_to_object_by_relation(
                self.config.obj1_uid,
                self.config.obj2_uid,
                scene.object_list,
                scene.cache_library.mesh_dict,
                "near",
                platform_uid="00000000000000000000000000000000",
                ignored_uid=self.config.ignored_uid,
                extra_erosion=extra_erosion,
            )
        else:
            if self.config.another_obj2_uid is not None:
                IS_OK = place_object_to_object_by_relation(
                    self.config.obj1_uid,
                    self.config.obj2_uid,
                    scene.object_list,
                    scene.cache_library.mesh_dict,
                    self.config.position,
                    platform_uid="00000000000000000000000000000000",
                    ignored_uid=self.config.ignored_uid,
                    extra_erosion=extra_erosion,
                    another_object2_uid=self.config.another_obj2_uid,
                )
            else:
                IS_OK = place_object_to_object_by_relation(
                    self.config.obj1_uid,
                    self.config.obj2_uid,
                    scene.object_list,
                    scene.cache_library.mesh_dict,
                    self.config.position,
                    platform_uid="00000000000000000000000000000000",
                    ignored_uid=self.config.ignored_uid,
                    extra_erosion=extra_erosion,
                )
        return IS_OK

    def _compute_final_object_pose(
        self, scene: "Scene", extra_erosion: float = 0.05
    ) -> dict | None:
        p_initial, q_initial = scene.object_list[self.config.obj1_uid].get_world_pose()
        if self._set_final_object_pose(scene, extra_erosion=extra_erosion) == -1:
            return None
        p_final, q_final = scene.object_list[self.config.obj1_uid].get_world_pose()
        scene.object_list[self.config.obj1_uid].set_world_pose(
            position=p_initial, orientation=q_initial
        )
        if self.config.fixed_position_config is not None:
            fixed_position_config = self.config.fixed_position_config
            if isinstance(fixed_position_config, list):
                fixed_position_config = random.choice(fixed_position_config)
            p_final = p_final + np.array(fixed_position_config["translation"])
            q_final = (
                R.from_quat(
                    np.array(fixed_position_config["orientation"])[[1, 2, 3, 0]]
                )
                * R.from_quat(q_final[[1, 2, 3, 0]])
            ).as_quat()[[3, 0, 1, 2]]
        return {"position": p_final, "orientation": q_final}

    def _get_initial_grasp(self, scene: "Scene") -> dict:
        set_camera_look_at(
            scene.camera_list["camera1"],
            scene.object_list[self.config.obj1_uid],
            azimuth=180.0,
        )
        current_pose_list = collect_world_pose_list(scene.object_list)
        current_joint_positions = scene.robot_list[0].robot.get_joint_positions()
        robot_world_pose = scene.robot_list[0].robot.get_world_pose()
        scene.robot_list[0].robot.set_world_pose(
            robot_world_pose[0] + np.array([1000.0, 0.0, 0.0]), robot_world_pose[1]
        )
        for _ in range(5):
            scene.world.step(render=True)

        meshlist = get_current_meshList(
            scene.object_list, scene.cache_library.mesh_dict
        )
        mesh = meshlist[self.config.obj1_uid]
        initial_grasp = get_init_grasp(
            scene.camera_list["camera1"],
            mesh,
            address=scene.default_config["ANYGRASP_ADDR"],
            allow_fixed_grasp=self.config.allow_fixed_grasp,
            force_fixed_grasp=self.config.force_fixed_grasp,
            idx=self.config.grasp_idx,
        )
        initial_grasp["position"] = initial_grasp["translation"]
        initial_grasp.pop("translation")

        for _ in range(5):
            if (
                current_pose_list is None
                or robot_world_pose is None
                or current_joint_positions is None
            ):
                raise ValueError(
                    "Current pose list, robot world pose, or current joint positions is not provided when force fixed grasp is not set"
                )
            reset_object_xyz(scene.object_list, current_pose_list)
            scene.robot_list[0].robot.set_joint_positions(current_joint_positions)
            scene.robot_list[0].robot.set_world_pose(*robot_world_pose)
            scene.world.step(render=True)

        if (
            self.config.allow_fixed_grasp
            and initial_grasp["position"][0] == 0.0
            and initial_grasp["position"][1] == 0.0
        ):
            initial_grasp["position"][:2] = scene.object_list[
                self.config.obj1_uid
            ].get_world_pose()[0][:2]
            initial_grasp["position"] = self._apply_fixed_grasp_translation(
                scene, initial_grasp["position"]
            )

        return initial_grasp

    def _get_initial_fixed_grasp(self, scene: "Scene") -> dict:
        meshlist = get_current_meshList(
            scene.object_list, scene.cache_library.mesh_dict
        )
        mesh = meshlist[self.config.obj1_uid]
        initial_grasp = get_init_grasp(
            scene.camera_list["camera1"],
            mesh,
            address=scene.default_config["ANYGRASP_ADDR"],
            allow_fixed_grasp=self.config.allow_fixed_grasp,
            force_fixed_grasp=self.config.force_fixed_grasp,
        )
        initial_grasp["position"] = initial_grasp["translation"]
        initial_grasp.pop("translation")
        initial_grasp["position"][:2] = scene.object_list[
            self.config.obj1_uid
        ].get_world_pose()[0][:2]
        initial_grasp["position"] = self._apply_fixed_grasp_translation(
            scene, initial_grasp["position"]
        )
        return initial_grasp

    def _get_meta_info(self, scene: "Scene") -> dict:
        action_meta_info = {}
        action_meta_info["final_object"] = self._compute_final_object_pose(
            scene, extra_erosion=0.05
        )
        if action_meta_info["final_object"] is None:
            raise Exception("can't create target position, retry......")
        if self.config.force_fixed_grasp:
            action_meta_info["initial_grasp"] = self._get_initial_fixed_grasp(scene)
        else:
            action_meta_info["initial_grasp"] = self._get_initial_grasp(scene)

        action_meta_info["initial_object"] = {}
        (
            action_meta_info["initial_object"]["position"],
            action_meta_info["initial_object"]["orientation"],
        ) = scene.object_list[self.config.obj1_uid].get_world_pose()

        if (
            self.config.gripper_rot_config is not None
            and self.config.gripper_rot_config.get("apply_in_relative_grasp", False)
        ):
            axis, angle = self._compute_gripper_rotation_angle(
                action_meta_info["initial_object"]["orientation"]
            )
            if axis is not None and angle is not None:
                action_meta_info["initial_grasp"]["orientation"] = (
                    rot_orientation_by_axis(
                        action_meta_info["initial_grasp"]["orientation"], axis, angle
                    )
                )

        action_meta_info["finial_grasp"] = {}
        (
            action_meta_info["finial_grasp"]["position"],
            action_meta_info["finial_grasp"]["orientation"],
        ) = compute_final_pose(
            action_meta_info["initial_object"]["position"],
            action_meta_info["initial_object"]["orientation"],
            action_meta_info["initial_grasp"]["position"],
            action_meta_info["initial_grasp"]["orientation"],
            action_meta_info["final_object"]["position"],
            action_meta_info["final_object"]["orientation"],
        )
        return action_meta_info

    def _update_planner(self, scene: "Scene") -> None:
        ignore_list = [
            f"obj_{self.config.obj1_uid}",
            f"obj_{self.scene_config.table_uid}",
        ]
        ignore_list.extend(self.config.plan_ignored_list)
        if scene.robot_list[0].planner is None:
            raise ValueError("Planner is not set")
        scene.robot_list[0].planner.update(ignore_list=ignore_list)

    def _record_planning(
        self,
        scene: "Scene",
        recorder: PlanningRecorder,
        embodiment: BaseEmbodiment,
        action_meta_info: dict,
        idx_name: str,
        smooth: bool = True,
        reset_tcp: tuple[list[float], list[float]] | float | bool | None = None,
        base_motion_list: dict = {},
    ) -> tuple[bool, str]:
        action_list = self._prepare_targets(action_meta_info, embodiment)

        self._auto_detect_arm(action_meta_info, embodiment)

        data_list = []
        sim_js = embodiment.robot.get_joints_state()

        start_action = None
        _target = action_list[0]
        with tqdm(total=len(action_list), desc=f"Planning {idx_name}") as tbar:
            for idx, target in enumerate(action_list):
                tbar.set_postfix_str(target["name"])
                if isinstance(base_motion_list, dict) and str(idx) in base_motion_list:
                    self._handle_complex_delta_move(
                        scene,
                        recorder,
                        embodiment,
                        data_list,
                        idx_name,
                        target,
                        base_motion_list[str(idx)],
                    )

                # list of joint position on the planned trajectory point
                trajectory_points = self._plan_step(embodiment, target, sim_js, smooth)

                current_step_actions = []
                for point in trajectory_points:
                    action_data = self._create_action_data(
                        embodiment, point, target, target["grasp"]
                    )
                    current_step_actions.append(action_data)
                    sim_js.positions = embodiment.convert_action_to_joint_state(
                        action_data["action"], self.config.arm
                    )

                if idx == 0 and len(data_list) == 0:
                    start_action = current_step_actions[0]
                data_list.extend(current_step_actions)

                if self._is_grasp_changing(idx, action_list):
                    transition_actions = self._generate_gripper_transition(
                        embodiment,
                        data_list[-1]["action"],
                        target["name"],
                        action_list[idx + 1]["grasp"],
                    )
                    data_list.extend(transition_actions)
                tbar.update()
                _target = target
            if (
                isinstance(base_motion_list, dict)
                and str(len(action_list)) in base_motion_list
            ):
                self._handle_complex_delta_move(
                    scene,
                    recorder,
                    embodiment,
                    data_list,
                    idx_name,
                    _target,
                    base_motion_list[str(len(action_list))],
                )

        if reset_tcp:
            if start_action is None:
                raise ValueError("start_action is not set")
            self._append_reset_trajectory(data_list, start_action, reset_tcp)

        self._excute_and_record(scene, recorder, embodiment, data_list, idx_name)

        return True, self.config.arm

    def _prepare_targets(
        self, action_meta_info: dict, embodiment: BaseEmbodiment
    ) -> list[dict]:
        action_list = prepare_motion_planning_payload(
            action_meta_info,
            aug_distance=self.scene_config.generation_config.aug_distance,
            **self.config.motion_config,
        )
        for target in action_list:
            adjust_grasp_by_embodiment(target, embodiment)
            # Backward-compatible path: keep old behavior unless relative-grasp mode is enabled.
            if (
                self.config.gripper_rot_config is not None
                and not self.config.gripper_rot_config.get(
                    "apply_in_relative_grasp", False
                )
            ):
                axis, angle = self._compute_gripper_rotation_angle(
                    action_meta_info["initial_object"]["orientation"]
                )
                if axis is not None and angle is not None:
                    target["orientation"] = rot_orientation_by_axis(
                        target["orientation"], axis, angle
                    )
        return action_list

    def _auto_detect_arm(self, action_meta_info: dict, embodiment: BaseEmbodiment):
        if self.config.arm == "auto":
            self.config.arm = embodiment.reference_arm_type(
                action_meta_info["initial_grasp"]["position"]
            )

    def _randomize_move(self, op: dict) -> dict:
        if isinstance(op["x"], list):
            if len(op["x"]) != 2:
                raise ValueError(f"If x is a list, it must be a list of two elements")
            op["x"] = random.uniform(op["x"][0], op["x"][1])
        if isinstance(op["y"], list):
            if len(op["y"]) != 2:
                raise ValueError(f"If y is a list, it must be a list of two elements")
            op["y"] = random.uniform(op["y"][0], op["y"][1])
        if isinstance(op["yaw"], list):
            if len(op["yaw"]) != 2:
                raise ValueError(f"If yaw is a list, it must be a list of two elements")
            op["yaw"] = random.uniform(op["yaw"][0], op["yaw"][1])
        return op

    def _handle_complex_delta_move(
        self,
        scene: "Scene",
        recorder: PlanningRecorder,
        embodiment: BaseEmbodiment,
        data_list: list[dict],
        idx_name: str,
        target: dict,
        ops: list[dict],
    ):
        if not isinstance(embodiment, DualArmEmbodiment):
            return

        self._excute_and_record(scene, recorder, embodiment, data_list, idx_name)

        _ops = ops.copy()
        for op in _ops:
            op = self._randomize_move(op)
            op_type = op.get("type", "align")
            if op_type == "align":
                pose = embodiment._transform_goal_pose(
                    (target["translation"], target["orientation"]), self.config.arm
                )
                if op["x"] is not None:
                    x_diff = pose[0][0] - op["x"]
                else:
                    x_diff = 0.0
                if op["y"] is not None:
                    y_diff = pose[0][1] - op["y"]
                else:
                    y_diff = 0.0
                self._handle_dual_arm_base_move(
                    scene,
                    recorder,
                    embodiment,
                    idx_name,
                    {"x": -x_diff, "y": -y_diff, "yaw": None},
                )
            elif op_type == "delta":
                self._handle_dual_arm_base_move(
                    scene, recorder, embodiment, idx_name, op
                )
            else:
                raise ValueError(f"Unknown operation type: {op_type}")

    def _handle_dual_arm_base_move(
        self,
        scene: "Scene",
        recorder: PlanningRecorder,
        embodiment: DualArmEmbodiment,
        idx_name: str,
        delta_move_config: dict = {"x": 0.0, "y": 0.0, "yaw": 0.0},
    ) -> None:
        max_xy = max(abs(delta_move_config["x"]), abs(delta_move_config["y"]))
        x_step = delta_move_config["x"] / max_xy * 0.01
        y_step = delta_move_config["y"] / max_xy * 0.01
        x_remaining = delta_move_config["x"]
        y_remaining = delta_move_config["y"]
        motion_list = []

        while x_remaining != 0.0 or y_remaining != 0.0:
            if abs(x_step) > abs(x_remaining):
                x_step = x_remaining
            if abs(y_step) > abs(y_remaining):
                y_step = y_remaining
            x_remaining -= x_step
            y_remaining -= y_step
            motion_list.append((x_step, y_step, 0.0))

        for motion in tqdm(motion_list, desc=f"Base motion executing {idx_name}"):
            grasp_val = None
            if self.last_action is not None:
                embodiment.robot_view.set_joint_position_targets(
                    self.last_action["action"],
                    joint_indices=embodiment.default_dof_indices,
                )
                grasp_val = 1.0 if self.last_action["grasp"] else -1.0
            embodiment.delta_move_to(*motion)
            recorder.load_dynamic_info(
                self.last_action["action"] if self.last_action is not None else None,
                grasp_val,
                arm=self.config.arm,
                base_motion=np.array([*motion]),
                name=f"{idx_name}/move",
            )
            scene.step(render=False)
            if os.environ.get("GENMANIP_DEBUG", "0") == "1":
                scene.world.render()

    def _plan_step(self, embodiment, target, sim_js, smooth: bool) -> list[list[float]]:
        results = embodiment.plan_pose(
            (target["translation"], target["orientation"]),
            sim_js,
            arm=self.config.arm,
        )
        if results is None:
            eepose = embodiment._transform_goal_pose(
                (target["translation"], target["orientation"]), self.config.arm
            )
            raise RuntimeError(
                f"Motion planning failed for target: {target['name']}, target pose might be out of reach: {eepose}"
            )

        if smooth:
            results = get_smooth_trajectory(
                torch.from_numpy(np.array(results)), 5
            ).numpy()

        return [np.asarray(res).tolist() for res in results]

    def _create_action_data(
        self, embodiment, joint_pos, target_info, grasp_state
    ) -> dict:
        return {
            "action": embodiment.convert_curobo_result_to_action(
                joint_pos, grasp_state, self.config.arm
            ).tolist(),
            "name": target_info["name"],
            "grasp": grasp_state,
        }

    def _is_grasp_changing(self, current_idx: int, action_list: list) -> bool:
        if current_idx == len(action_list) - 1:
            return False
        return (
            action_list[current_idx]["grasp"] != action_list[current_idx + 1]["grasp"]
        )

    def _generate_gripper_transition(
        self, embodiment, last_action, target_name, next_grasp_state
    ) -> list[dict]:
        transition_list = []
        for _ in range(13):
            new_action = embodiment.convert_curobo_result_to_action(
                last_action, next_grasp_state, self.config.arm
            ).tolist()

            transition_list.append(
                {
                    "action": new_action,
                    "name": target_name,
                    "grasp": next_grasp_state,
                }
            )
        return transition_list

    def _append_reset_trajectory(
        self, data_list: list[dict], start_action: dict, reset_tcp_config
    ):
        if not data_list:
            return

        if isinstance(reset_tcp_config, bool) and reset_tcp_config:
            reset_val = 0.0
        elif isinstance(reset_tcp_config, (float, int)) and reset_tcp_config != -1:
            reset_val = float(reset_tcp_config)
        else:
            return

        original_final_joints = np.array(data_list[-1]["action"])
        start_joints = np.array(start_action["action"])

        ratio = np.random.uniform(0.0, reset_val)
        target_reset_joints = start_joints * (1 - ratio) + original_final_joints * ratio

        delta_joints = (target_reset_joints - original_final_joints) / 100

        for i in range(100):
            next_joints = original_final_joints + delta_joints * i
            data_list.append(
                {
                    "action": next_joints.tolist(),
                    "name": "reset_tcp",
                    "grasp": False,
                }
            )

    def _excute_and_record(
        self,
        scene: "Scene",
        recorder: PlanningRecorder,
        embodiment: BaseEmbodiment,
        data_list: list[dict],
        idx_name: str,
    ) -> None:
        for action in tqdm(data_list, desc=f"Arm action executing {idx_name}"):
            embodiment.robot_view.set_joint_position_targets(
                action["action"], joint_indices=embodiment.default_dof_indices
            )
            grasp_val = 1.0 if action["grasp"] else -1.0
            recorder.load_dynamic_info(
                action["action"],
                grasp_val,
                arm=self.config.arm,
                name=f"{idx_name}/{action['name']}",
            )
            self.last_action = action
            scene.step(render=False)
            if os.environ.get("GENMANIP_DEBUG", "0") == "1":
                scene.world.render()
        data_list.clear()

    def _is_done(self, scene: "Scene") -> bool:
        pclist = get_current_pcList_by_meshList(
            scene.object_list, scene.cache_library.mesh_dict
        )
        is_success = check_subgoal_finished_rigid(
            self.config.position,
            pclist[self.config.obj1_uid],
            pclist[self.config.obj2_uid],
        )
        return is_success or (self.config.fixed_position and self.config.mesh_top_only)
