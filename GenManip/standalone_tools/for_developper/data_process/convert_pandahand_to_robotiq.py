import os
import pickle
import numpy as np
import lmdb
from tqdm import tqdm
from multiprocessing import Pool
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        required=True,
        help="path to the data folder, this folder should contain trajectory folder",
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=10, help="number of workers"
    )
    return parser.parse_args()


def fix_qpos_to_robotiq(data_path):
    np.random.seed(hash(data_path) % (2**32))
    max_size = int(1 * 1024**4)
    lmdb_env = lmdb.open(f"{data_path}/lmdb", map_size=max_size)
    meta_info = pickle.load(open(f"{data_path}/meta_info.pkl", "rb"))
    if "franka_robot" in meta_info["task_data"]["initial_layout"].keys():
        franka_data = meta_info["task_data"]["initial_layout"]["franka_robot"]
        franka_data["joint_positions"] = np.concatenate(
            [franka_data["joint_positions"][:7], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        franka_data["position"] += np.array(
            [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), 0]
        )
        meta_info["task_data"]["initial_layout"]["franka_robotiq"] = franka_data
        meta_info["task_data"]["initial_layout"].pop("franka_robot")
    qpos_index = meta_info["keys"]["scalar_data"].index(b"observation/robot/qpos")
    qpos_key = meta_info["keys"]["scalar_data"][qpos_index]
    with lmdb_env.begin(write=False) as txn:
        qpos = pickle.loads(txn.get(qpos_key))
    new_qpos_list = []
    for i in range(len(qpos)):
        new_qpos = np.concatenate([qpos[i][:7], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        new_qpos_list.append(new_qpos)
    key = "observation/robot/qpos"
    meta_info["keys"]["scalar_data"].append(key.encode("utf-8"))
    pickle.dump(meta_info, open(f"{data_path}/meta_info.pkl", "wb"))
    with lmdb_env.begin(write=True) as txn:
        txn.put(key.encode("utf-8"), pickle.dumps(new_qpos_list))


if __name__ == "__main__":
    args = parse_args()
    data_path = args.folder
    print("\033[91m\033[1m", end="")
    print("-" * 126)
    print(
        "| [Warning] this script is used to convert panda hand to robotiq, make sure your data is only used to render the first frame |"
    )
    print("-" * 126)
    print("\033[0m", end="")
    ret = input("\033[92m\033[1mY(es)\033[0m/\033[91m\033[1mN(o)\033[0m:")
    if ret.lower() != "y" and ret.lower() != "yes":
        exit()
    data_list = os.listdir(f"{data_path}/trajectory")
    data_list = [
        f"{data_path}/trajectory/{path}"
        for path in data_list
        if os.path.isdir(f"{data_path}/trajectory/{path}")
    ]
    with Pool(processes=args.workers) as pool:
        list(tqdm(pool.imap(fix_qpos_to_robotiq, data_list), total=len(data_list)))
    print("Done, welcome to robotiq")
