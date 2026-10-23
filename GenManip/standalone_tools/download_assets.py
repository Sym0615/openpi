"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import argparse
import os
import pathlib
import shutil
import tarfile
from huggingface_hub import snapshot_download


def find_filetree_depth(path, cnt=0) -> int:
    path_list = os.listdir(path)
    if "config.yaml" in path_list:
        return cnt
    for item in path_list:
        if os.path.isdir(os.path.join(path, item)):
            cnt = find_filetree_depth(os.path.join(path, item), cnt + 1)
    return cnt


def remove_huggingface_info(path):
    if os.path.exists(f"{path}/.collect.mapping.json"):
        os.remove(f"{path}/.collect.mapping.json")
    if os.path.exists(f"{path}/.gitattributes"):
        os.remove(f"{path}/.gitattributes")
    if os.path.exists(f"{path}/README.md"):
        os.remove(f"{path}/README.md")


def download_basic_dataset():
    repo_id = "Axi404/GenManip-Basic-Assets"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir="saved/assets/scene_usds/base_scenes",
    )
    remove_huggingface_info("saved/assets/scene_usds/base_scenes")


def download_banana_dataset():
    repo_id = "Axi404/GenManip-Banana-Assets"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir="saved/assets/scene_usds/debug_scenes/banana_plate_scenes",
    )
    remove_huggingface_info("saved/assets/scene_usds/debug_scenes/banana_plate_scenes")


def download_banana_layout_dataset():
    repo_id = "Axi404/GenManip-Banana-Layouts"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir="saved/tasks/Minimal_Banana",
    )
    remove_huggingface_info("saved/tasks/Minimal_Banana")


def download_robot_franka_robotiq_dataset():
    repo_id = "Axi404/GenManip-Robot-FrankaRobotiq-Assets"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir="saved/assets/robot_usds/robotiq",
    )
    remove_huggingface_info("saved/assets/robot_usds/robotiq")


def download_robot_split_aloha_mid_360_dataset():
    repo_id = "Axi404/GenManip-Robot-AlohaSplit-Assets"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir="saved/assets/robot_usds/split_aloha_mid_360",
    )
    remove_huggingface_info("saved/assets/robot_usds/split_aloha_mid_360")
    repo_id = "Axi404/GenManip-Piper-cuRobo-Assets"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir="saved/assets/miscs/curobo/piper100",
    )
    remove_huggingface_info("saved/assets/miscs/curobo/piper100")


def download_robot_lift2_dataset():
    repo_id = "Axi404/GenManip-Robot-Lift2-Assets"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir="saved/assets/robot_usds/lift2",
    )
    remove_huggingface_info("saved/assets/robot_usds/lift2")
    repo_id = "Axi404/GenManip-R5a-cuRobo-Assets"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir="saved/assets/miscs/curobo/R5a",
    )
    remove_huggingface_info("saved/assets/miscs/curobo/R5a")


def download_objaverse_subset_dataset():
    repo_id = "Axi404/GenManip-Assets-ObjaverseSubset"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir="saved/assets/object_usds/objaverse_usd",
    )
    remove_huggingface_info("saved/assets/object_usds/objaverse_usd")
    parts_dir = "saved/assets/object_usds/objaverse_usd/textures_tar"
    output_tar = "saved/assets/object_usds/objaverse_usd/textures.tar"
    extract_dir = "saved/assets/object_usds/objaverse_usd/textures"
    parts = sorted(
        f for f in os.listdir(parts_dir) if f.startswith("textures.tar.part")
    )
    with open(output_tar, "wb") as out:
        for p in parts:
            with open(os.path.join(parts_dir, p), "rb") as f:
                out.write(f.read())
    with tarfile.open(output_tar, "r") as tar:
        tar.extractall(os.path.dirname(extract_dir))
    shutil.rmtree(parts_dir)
    os.remove(output_tar)


def download_all_robot_datasets():
    download_robot_franka_robotiq_dataset()
    download_robot_split_aloha_mid_360_dataset()
    download_robot_lift2_dataset()


def download_custom_benchmark_dataset(repo_id: str):
    pathlib.Path("saved/assets/collected_packages").mkdir(parents=True, exist_ok=True)
    pathlib.Path("saved/tasks").mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=f"saved/assets/collected_packages/{repo_id.split('/')[-1]}",
    )
    remove_huggingface_info(f"saved/assets/collected_packages/{repo_id.split('/')[-1]}")

    if os.path.exists(
        f"saved/assets/collected_packages/{repo_id.split('/')[-1]}/.version"
    ):
        return

    src = f"saved/assets/collected_packages/{repo_id.split('/')[-1]}/tasks"
    dst = f"saved/tasks"
    for item in os.listdir(src):
        if os.path.isdir(os.path.join(src, item)):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            shutil.move(s, d)

    shutil.move(
        os.path.join(src, "config.yaml"),
        os.path.join("configs/tasks", f"{repo_id.split('/')[-1]}.yml"),
    )

    camera_src = f"saved/assets/collected_packages/{repo_id.split('/')[-1]}/cameras"
    camera_dst = f"configs/cameras/{repo_id.split('/')[-1]}"
    pathlib.Path(camera_dst).mkdir(parents=True, exist_ok=True)
    for item in os.listdir(camera_src):
        shutil.copyfile(os.path.join(camera_src, item), os.path.join(camera_dst, item))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    path = pathlib.Path("saved/assets")
    path.mkdir(parents=True, exist_ok=True)

    if args.dataset == "basic":
        download_basic_dataset()
    elif args.dataset == "banana":
        download_banana_dataset()
    elif args.dataset == "banana-layout":
        download_banana_layout_dataset()
    elif args.dataset == "all-robot":
        download_all_robot_datasets()
    elif args.dataset == "franka-robotiq":
        download_robot_franka_robotiq_dataset()
    elif args.dataset == "split-aloha-mid-360":
        download_robot_split_aloha_mid_360_dataset()
    elif args.dataset == "lift2":
        download_robot_lift2_dataset()
    elif args.dataset == "objaverse-subset":
        download_objaverse_subset_dataset()
    else:
        download_custom_benchmark_dataset(args.dataset)
