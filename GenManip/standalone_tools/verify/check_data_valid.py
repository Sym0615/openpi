import argparse
import os
import shutil
import tqdm
import pickle
import lmdb
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_scalar_data_from_lmdb(data_path: str, key: str | bytes) -> list[Any]:
    meta_info = pickle.load(open(f"{data_path}/meta_info.pkl", "rb"))
    lmdb_env = lmdb.open(
        f"{data_path}/lmdb", readonly=True, lock=False, readahead=False, meminit=False
    )
    key_index = meta_info["keys"]["scalar_data"].index(key)
    key_key = meta_info["keys"]["scalar_data"][key_index]
    with lmdb_env.begin(write=False) as txn:
        data = pickle.loads(txn.get(key_key))
    return data


def check_single_data(data_path: str, type: str = "trajectory"):
    """
    .
    |-- config.yaml
    |-- lmdb
    |   |-- data.mdb
    |   |-- info.json
    |   `-- lock.mdb
    `-- meta_info.pkl
    """
    if (
        not os.path.exists(os.path.join(data_path, "config.yaml"))
        and type == "trajectory"
    ):
        return False
    if not os.path.exists(os.path.join(data_path, "lmdb")):
        return False
    if not os.path.exists(os.path.join(data_path, "lmdb", "data.mdb")):
        return False
    if not os.path.exists(os.path.join(data_path, "lmdb", "info.json")):
        return False
    if not os.path.exists(os.path.join(data_path, "lmdb", "lock.mdb")):
        return False
    if not os.path.exists(os.path.join(data_path, "meta_info.pkl")):
        return False
    try:
        if type == "trajectory":
            qpos_data = get_scalar_data_from_lmdb(data_path, b"observation/robot/qpos")
            qvel_data = get_scalar_data_from_lmdb(data_path, b"observation/robot/qvel")
            arm_action_data = get_scalar_data_from_lmdb(data_path, b"arm_action")
            gripper_action_data = get_scalar_data_from_lmdb(
                data_path, b"gripper_action"
            )
            gripper_close_data = get_scalar_data_from_lmdb(data_path, b"gripper_close")
            name_data = get_scalar_data_from_lmdb(data_path, b"name")
        elif type == "render":
            qpos_data = get_scalar_data_from_lmdb(data_path, b"observation/robot/qpos")
            qvel_data = get_scalar_data_from_lmdb(data_path, b"observation/robot/qvel")
            arm_action_data = get_scalar_data_from_lmdb(data_path, b"arm_action")
            gripper_action_data = get_scalar_data_from_lmdb(
                data_path, b"gripper_action"
            )
            gripper_close_data = get_scalar_data_from_lmdb(data_path, b"gripper_close")
            name_data = get_scalar_data_from_lmdb(data_path, b"name")
    except (
        KeyError,
        lmdb.Error,
        OSError,
        ValueError,
        EOFError,
        TypeError,
        pickle.PickleError,
    ) as exc:
        print(f"Invalid data for {data_path}: {exc}")
        return False
    with open(os.path.join(data_path, "meta_info.pkl"), "rb") as f:
        meta_info = pickle.load(f)
    if "task_data" not in meta_info:
        return False
    if "goal" not in meta_info["task_data"]:
        return False
    if "initial_layout" not in meta_info["task_data"]:
        return False
    return True


def process_single_data(
    data_path: str, type: str = "trajectory", delete_invalid: bool = True
):
    is_valid = check_single_data(data_path, type)
    if not is_valid:
        if delete_invalid:
            shutil.rmtree(data_path)
        print(f"Invalid data: {data_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--type", type=str, default="trajectory")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--delete_invalid", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = os.listdir(args.data_path)
    data_paths = [
        os.path.join(args.data_path, path)
        for path in paths
        if os.path.isdir(os.path.join(args.data_path, path))
    ]
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                process_single_data, data_path, args.type, args.delete_invalid
            )
            for data_path in data_paths
        ]
        for future in tqdm.tqdm(
            as_completed(futures), total=len(futures), desc="Processing data"
        ):
            future.result()


if __name__ == "__main__":
    main()
