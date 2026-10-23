"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import argparse
import asyncio
import os
from pathlib import Path
import pickle
import shutil
import json
from tqdm import tqdm
import huggingface_hub

import yaml

from isaacsim import SimulationApp  # type: ignore[import-untyped]

kit = SimulationApp({"headless": True})
from omni.isaac.core.utils.extensions import enable_extension  # type: ignore

enable_extension("omni.kit.usd.collect")

from omni.kit.usd.collect import Collector, CollectorStatus  # type: ignore

VERSION = "0.1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_path", type=str, nargs="+", required=True)
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--upload_to_huggingface", action="store_true", default=False)
    parser.add_argument("--huggingface_username", type=str, required=False)
    return parser.parse_args()


def collect_pickle_files(asset_path: str) -> list[str]:
    pickle_files = []
    for root, dirs, files in os.walk(asset_path):
        for file in files:
            if file.endswith("meta_info.pkl"):
                pickle_files.append(os.path.join(root, file))
    return pickle_files


def check_task_rename(pickle_path: str) -> bool:
    folder_name = str(os.path.dirname(pickle_path)).split("/")[-1]
    if folder_name.isdigit() and len(folder_name) == 3:
        return True
    else:
        return False


def rename_task_folder(pickle_path: str) -> None:
    folder_name = os.path.dirname(os.path.dirname(pickle_path))
    cnt = 0
    for log_dir in os.listdir(folder_name):
        if os.path.isdir(os.path.join(folder_name, log_dir)):
            os.rename(
                os.path.join(folder_name, log_dir),
                os.path.join(folder_name, f"{str(cnt).zfill(3)}"),
            )
            cnt += 1


def parse_pickle_file(pickle_file: str) -> dict:
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    yaml_dir = str(pickle_file).replace("meta_info.pkl", "config.yaml")
    with open(yaml_dir, "r") as f:
        yaml_data = yaml.load(f, yaml.FullLoader)
        yaml_data = json.loads(json.dumps(yaml_data))
    data["scene_usd_path"] = []
    for v in yaml_data["evaluation_configs"]:
        data["scene_usd_path"].append(v["usd_name"])
    return data


def collect_usd(usd_path: str, target_dir: str) -> bool:
    collector = Collector(
        usd_path, target_dir, usd_only=False, flat_collection=False, skip_existing=True
    )
    asyncio.ensure_future(collector.collect())
    while (
        collector.get_status() == CollectorStatus.IN_PROGRESS
        or collector.get_status() == CollectorStatus.NOT_STARTED
    ):
        kit.update()
    if collector.get_status() == CollectorStatus.FINISHED:
        return True
    elif collector.get_status() == CollectorStatus.CANCELLED:
        print(f"Collection cancelled or failed: {usd_path}")
        return False
    else:
        print(f"Unknown collector state: {collector.get_status()}")
        return False


usda_template = """#usda 1.0
(
    customLayerData = {{
        dictionary omni_layer = {{
            dictionary locked = {{
            }}
            dictionary muteness = {{
            }}
        }}
        dictionary renderSettings = {{
            float3 "rtx:debugView:pixelDebug:textColor" = (0, 1e18, 0)
            float3 "rtx:fog:fogColor" = (0.75, 0.75, 0.75)
            float3 "rtx:index:backgroundColor" = (0, 0, 0)
            float3 "rtx:index:regionOfInterestMax" = (0, 0, 0)
            float3 "rtx:index:regionOfInterestMin" = (0, 0, 0)
            float3 "rtx:post:backgroundZeroAlpha:backgroundDefaultColor" = (0, 0, 0)
            float3 "rtx:post:colorcorr:contrast" = (1, 1, 1)
            float3 "rtx:post:colorcorr:gain" = (1, 1, 1)
            float3 "rtx:post:colorcorr:gamma" = (1, 1, 1)
            float3 "rtx:post:colorcorr:offset" = (0, 0, 0)
            float3 "rtx:post:colorcorr:saturation" = (1, 1, 1)
            float3 "rtx:post:colorgrad:blackpoint" = (0, 0, 0)
            float3 "rtx:post:colorgrad:contrast" = (1, 1, 1)
            float3 "rtx:post:colorgrad:gain" = (1, 1, 1)
            float3 "rtx:post:colorgrad:gamma" = (1, 1, 1)
            float3 "rtx:post:colorgrad:lift" = (0, 0, 0)
            float3 "rtx:post:colorgrad:multiply" = (1, 1, 1)
            float3 "rtx:post:colorgrad:offset" = (0, 0, 0)
            float3 "rtx:post:colorgrad:whitepoint" = (1, 1, 1)
            float3 "rtx:post:lensDistortion:lensFocalLengthArray" = (10, 30, 50)
            float3 "rtx:post:lensFlares:anisoFlareFalloffX" = (450, 475, 500)
            float3 "rtx:post:lensFlares:anisoFlareFalloffY" = (10, 10, 10)
            float3 "rtx:post:lensFlares:cutoffPoint" = (2, 2, 2)
            float3 "rtx:post:lensFlares:haloFlareFalloff" = (10, 10, 10)
            float3 "rtx:post:lensFlares:haloFlareRadius" = (75, 75, 75)
            float3 "rtx:post:lensFlares:isotropicFlareFalloff" = (50, 50, 50)
            float3 "rtx:post:tonemap:whitepoint" = (1, 1, 1)
            float3 "rtx:raytracing:indexdirect:svoBrickSize" = (32, 32, 32)
            float3 "rtx:raytracing:inscattering:singleScatteringAlbedo" = (0.9, 0.9, 0.9)
            float3 "rtx:raytracing:inscattering:transmittanceColor" = (0.5, 0.5, 0.5)
            float3 "rtx:sceneDb:ambientLightColor" = (0.1, 0.1, 0.1)
            double "rtx:translucency:worldEps" = 0.005
        }}
    }}
    defaultPrim = "World"
    endTimeCode = 1000000
    metersPerUnit = 1.0
    startTimeCode = 0
    timeCodesPerSecond = 60
    upAxis = "Z"
)

over "Render" (
    hide_in_stage_window = true
)
{{
}}

def Xform "World"
{{
    def "_{uid}" (
        prepend payload = @./{absolute_usd_path}@
    )
    {{
        float3 xformOp:rotateXYZ = (0, 0, 0)
        float3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }}
}}

def Xform "Environment"
{{
    double3 xformOp:rotateXYZ = (0, 0, 0)
    double3 xformOp:scale = (1, 1, 1)
    double3 xformOp:translate = (0, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
}}
"""


def usda_gen(root_path: str) -> None:
    file_list = []

    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".usd"):
                file_list.append((root, file))

    pbar = tqdm(file_list, desc="Generating usda files")
    for root, file in pbar:
        pbar.set_description(f"Generating usda files: {file.split('/')[-1]}")
        uid = file[:-4]
        usda_content = usda_template.format(uid=uid, absolute_usd_path=file)
        usda_filename = f"{uid}.usda"
        usda_filepath = os.path.join(root, usda_filename)
        with open(usda_filepath, "w", encoding="utf-8") as usda_file:
            usda_file.write(usda_content)


def print_info(
    file_projection: dict[str, str],
    scene_raw_path_list: list[str],
    asset_raw_path_list: list[str],
    pickle_infos: dict[str, dict],
    upload_to_huggingface: bool,
    repo_id: str | None,
) -> None:
    print("-" * 50)
    print(f"Collected assets successfully")
    print("-" * 50)

    for k, v in file_projection.items():
        print(f"{k} -> {v}/instance.usd")
    print("-" * 50)
    print(f"Collect {len(scene_raw_path_list)} scenes")
    print(f"Collect {len(asset_raw_path_list)} assets")
    print(f"Processed {len(pickle_infos)} pickle files")
    print("-" * 50)
    if upload_to_huggingface:
        if repo_id is not None:
            print(f"Upload to Hugging Face {repo_id}")
        else:
            print("Upload to Hugging Face cancelled")
        print("-" * 50)


def collect_assets(
    pickle_infos: dict[str, dict], base_path: str, dataset_id: str
) -> tuple[dict[str, str], list[str], list[str]]:
    file_projection: dict[str, str] = {}
    scene_raw_path_list: list[str] = []
    asset_raw_path_list: list[str] = []

    for pickle_info in pickle_infos.values():
        for base_usd_path in pickle_info["scene_usd_path"]:
            if base_usd_path not in scene_raw_path_list:
                scene_raw_path_list.append(base_usd_path)
        for usd_info in pickle_info["task_data"]["initial_layout"].values():
            if (
                "path" in usd_info
                and usd_info["path"] != ""
                and usd_info["path"] not in asset_raw_path_list
            ):
                asset_raw_path_list.append(usd_info["path"])

    pbar = tqdm(scene_raw_path_list, desc="Collecting scenes")
    for base_usd_path in pbar:
        pbar.set_description(f"Collecting scenes: {base_usd_path.split('/')[-1]}")
        usd_raw_path = os.path.join(base_path, f"{base_usd_path}.usd")
        if not os.path.exists(usd_raw_path):
            continue
        target_raw_dir = (
            str(os.path.join(base_path, base_usd_path))
            .replace(
                "saved/assets",
                f"saved/assets/collected_packages/GenManip-Package-{dataset_id}/scenes",
            )
            .replace(".usd", "")
        )
        Path(target_raw_dir).mkdir(parents=True, exist_ok=True)
        collect_usd(usd_raw_path, target_raw_dir)
        file_projection[usd_raw_path] = target_raw_dir

    pbar = tqdm(
        asset_raw_path_list,
        desc=f"Collecting objects",
    )
    for asset_raw_path in pbar:
        usd_raw_path = os.path.join(base_path, asset_raw_path)
        if not os.path.exists(usd_raw_path):
            continue
        pbar.set_description(f"Collecting objects: {asset_raw_path.split('/')[-1]}")
        target_raw_dir = (
            str(os.path.join(base_path, asset_raw_path))
            .replace(
                "saved/assets",
                f"saved/assets/collected_packages/GenManip-Package-{dataset_id}/assets",
            )
            .replace(".usd", "")
        )
        Path(target_raw_dir).mkdir(parents=True, exist_ok=True)
        collect_usd(os.path.join(base_path, asset_raw_path), target_raw_dir)
        file_projection[usd_raw_path] = target_raw_dir

    return file_projection, scene_raw_path_list, asset_raw_path_list


def rename_assets(file_projection: dict[str, str]) -> None:
    pbar = tqdm(file_projection.items(), desc="Renaming assets")
    for original_path, target_path in pbar:
        pbar.set_description(f"Renaming assets: {original_path.split('/')[-1]}")
        if os.path.exists(os.path.join(target_path, str(original_path).split("/")[-1])):
            os.rename(
                os.path.join(target_path, str(original_path).split("/")[-1]),
                os.path.join(target_path, "instance.usd"),
            )


def rewrite_pickle_info(
    pickle_infos: dict[str, dict],
    file_projection: dict[str, str],
    base_path: str,
    dataset_id: str,
) -> None:
    total_config_list = []
    pbar = tqdm(pickle_infos.items(), desc="Rewriting pickle info")
    for pickle_file, pickle_info in pbar:
        pbar.set_description(f"Rewriting pickle info: {pickle_file.split('/')[-1]}")
        for v in pickle_info["task_data"]["initial_layout"].values():
            if "path" in v and v["path"] != "":
                usd_raw_path = os.path.join(base_path, v["path"])
                v["path"] = str(
                    os.path.join(file_projection[usd_raw_path], "instance.usd")
                ).replace("saved/assets/", "")
        case_dir = os.path.dirname(
            os.path.join(os.path.dirname(os.path.dirname(base_path)), pickle_file)
        )
        target_case_dir = str(case_dir).replace(
            f"saved/tasks/GenManip-Package-{dataset_id}",
            f"saved/assets/collected_packages/GenManip-Package-{dataset_id}/tasks",
        )
        Path(os.path.dirname(target_case_dir)).mkdir(parents=True, exist_ok=True)
        shutil.copytree(case_dir, target_case_dir, dirs_exist_ok=True)
        with open(os.path.join(target_case_dir, "meta_info.pkl"), "wb") as f:
            pickle.dump(pickle_info, f)
        with open(os.path.join(target_case_dir, "config.yaml"), "r") as f:
            config_data = yaml.load(f, yaml.FullLoader)
            config_data = json.loads(json.dumps(config_data))
            config_data["demonstration_configs"] = []
            for v in config_data["evaluation_configs"]:
                v["usd_name"] = (
                    str(
                        file_projection[os.path.join(base_path, f"{v['usd_name']}.usd")]
                    ).replace(base_path + "/", "")
                ) + "/instance"
            if config_data not in total_config_list:
                total_config_list.append(config_data)
            with open(os.path.join(target_case_dir, "config.yaml"), "w") as f:
                yaml.dump(config_data, f)
    total_config = {"evaluation_configs": [], "demonstration_configs": []}
    for cfg in total_config_list:
        total_config["evaluation_configs"].extend(cfg["evaluation_configs"])
    with open(
        os.path.join(
            base_path,
            "collected_packages",
            f"GenManip-Package-{dataset_id}",
            "tasks",
            "config.yaml",
        ),
        "w",
    ) as f:
        yaml.dump(total_config, f)


def upload_to_huggingface(
    base_path: str, dataset_id: str, args: argparse.Namespace
) -> str | None:
    api = huggingface_hub.HfApi()

    if args.huggingface_username:
        whoami = args.huggingface_username
    else:
        whoami = api.whoami()["name"]
    repo_id = f"{whoami}/GenManip-Package-{dataset_id}"
    ans = input(f"Upload to Hugging Face {repo_id} ? (y/n): ")

    if ans == "y":
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=False,
                exist_ok=True,
            )
        except (
            OSError,
            huggingface_hub.errors.HfHubHTTPError,
            huggingface_hub.errors.ValidationError,
        ) as e:
            print(f"Repo creation warning: {e}")

        huggingface_hub.upload_large_folder(
            repo_id=repo_id,
            folder_path=os.path.join(
                base_path,
                "collected_packages",
                f"GenManip-Package-{dataset_id}",
            ),
            repo_type="dataset",
        )
        return repo_id
    else:
        print("Upload to Hugging Face cancelled")
        return None


def preprocess_asset_path(asset_path: list[str], dataset_id) -> str:
    camera_config_path = (
        f"saved/assets/collected_packages/GenManip-Package-{dataset_id}/cameras"
    )
    Path(camera_config_path).mkdir(parents=True, exist_ok=True)
    task_path = "saved/tasks"
    for path in asset_path:
        rel_path = os.path.relpath(path, task_path)
        Path(os.path.join(task_path, f"GenManip-Package-{dataset_id}")).mkdir(
            parents=True, exist_ok=True
        )
        shutil.copytree(
            path, os.path.join(task_path, f"GenManip-Package-{dataset_id}", rel_path)
        )
        for root, dirs, files in os.walk(
            os.path.join(task_path, f"GenManip-Package-{dataset_id}", rel_path)
        ):
            if "config.yaml" in files:
                with open(os.path.join(root, "config.yaml"), "r") as f:
                    config_data = yaml.load(f, yaml.FullLoader)
                    config_data = json.loads(json.dumps(config_data))
                config_data["demonstration_configs"] = []
                for v in config_data["evaluation_configs"]:
                    camera_dst_path = os.path.join(
                        camera_config_path,
                        v["domain_randomization"]["cameras"]["config_path"].replace(
                            "configs/cameras/", ""
                        ),
                    )
                    if not os.path.exists(camera_dst_path):
                        shutil.copyfile(
                            v["domain_randomization"]["cameras"]["config_path"],
                            camera_dst_path,
                        )
                    v["domain_randomization"]["cameras"]["config_path"] = v[
                        "domain_randomization"
                    ]["cameras"]["config_path"].replace(
                        "configs/cameras/",
                        f"saved/assets/collected_packages/GenManip-Package-{dataset_id}/cameras/",
                    )
                with open(os.path.join(root, "config.yaml"), "w") as f:
                    yaml.dump(config_data, f)
    return str(os.path.join(task_path, f"GenManip-Package-{dataset_id}"))


def main() -> None:
    args = parse_args()

    # Create meta info for collected assets
    base_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "saved", "assets")
    )
    asset_path = args.asset_path
    dataset_id = args.dataset_id
    asset_path = preprocess_asset_path(asset_path, dataset_id)
    collect_path = os.path.join(
        base_path, "collected_packages", f"GenManip-Package-{dataset_id}"
    )
    Path(collect_path).mkdir(parents=True, exist_ok=True)
    pickle_files = collect_pickle_files(asset_path)
    pickle_files_available = [
        check_task_rename(pickle_path=pickle_file) for pickle_file in pickle_files
    ]
    while not all(pickle_files_available):
        rename_task_folder(pickle_files[pickle_files_available.index(False)])
        pickle_files = collect_pickle_files(asset_path)
        pickle_files_available = [
            check_task_rename(pickle_path=pickle_file) for pickle_file in pickle_files
        ]
    pickle_infos = {
        pickle_file: parse_pickle_file(pickle_file) for pickle_file in pickle_files
    }

    # Collect assets and scenes
    file_projection, scene_raw_path_list, asset_raw_path_list = collect_assets(
        pickle_infos=pickle_infos, base_path=base_path, dataset_id=dataset_id
    )

    # Rename assets
    rename_assets(file_projection=file_projection)

    # Rewrite Pickle Info
    rewrite_pickle_info(
        pickle_infos=pickle_infos,
        file_projection=file_projection,
        base_path=base_path,
        dataset_id=dataset_id,
    )

    # Generate USDA file
    usda_gen(
        os.path.join(
            base_path,
            "collected_packages",
            f"GenManip-Package-{dataset_id}",
            "scenes",
        )
    )

    with open(
        f"saved/assets/collected_packages/GenManip-Package-{dataset_id}/.version", "w"
    ) as f:
        f.write(VERSION)

    if args.upload_to_huggingface:
        repo_id = upload_to_huggingface(
            base_path=base_path, dataset_id=dataset_id, args=args
        )

    print_info(
        file_projection=file_projection,
        scene_raw_path_list=scene_raw_path_list,
        asset_raw_path_list=asset_raw_path_list,
        pickle_infos=pickle_infos,
        upload_to_huggingface=args.upload_to_huggingface,
        repo_id=repo_id if args.upload_to_huggingface else None,  # type: ignore
    )

    kit.close()


if __name__ == "__main__":
    main()
