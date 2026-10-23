import argparse
import lmdb, pickle, os
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation as R


def joint_position_to_end_effector_pose(joint_position, panda=None):
    if panda is None:
        panda = rtb.models.Panda()
    hand_pose = panda.fkine(q=joint_position, end="panda_hand").A  # type: ignore[attr-defined]
    position = hand_pose[:3, 3]
    rotation = hand_pose[:3, :3]
    orientation = R.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    return position, orientation


def add_ee_pose_state(log_dir: str, franka: rtb.models.Panda):
    lmdb_path = os.path.join(log_dir, "lmdb")
    env = lmdb.open(lmdb_path, map_size=0)  # map_size 不变
    with env.begin() as rtxn:
        # 读出所有 qpos
        qpos_bytes = rtxn.get(b"observation/robot/qpos")
        qpos_list = pickle.loads(qpos_bytes)
    # 计算 ee_pose_state
    ee_states = [
        joint_position_to_end_effector_pose(qpos, franka) for qpos in qpos_list
    ]

    # 写回 LMDB
    with env.begin(write=True) as wtxn:
        wtxn.put(b"observation/robot/ee_pose_state", pickle.dumps(ee_states))

    env.close()
    print(f"Added {len(ee_states)} ee_pose_state entries to {lmdb_path}")


def add_ee_pose_action(log_dir: str, franka: rtb.models.Panda):
    lmdb_path = os.path.join(log_dir, "lmdb")
    env = lmdb.open(lmdb_path, map_size=0)  # map_size 不变
    with env.begin() as rtxn:
        # 读出所有 qpos
        qpos_bytes = rtxn.get(b"arm_action")
        qpos_list = pickle.loads(qpos_bytes)
    ee_poses = [joint_position_to_end_effector_pose(qpos, franka) for qpos in qpos_list]
    with env.begin(write=True) as wtxn:
        wtxn.put(b"ee_pose_action", pickle.dumps(ee_poses))
    env.close()
    print(f"Added {len(ee_poses)} ee_pose_action entries to {lmdb_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_dir", type=str, required=True)
    args = parser.parse_args()
    franka = rtb.models.Panda()
    for log_dir in os.listdir(args.log_dir):
        add_ee_pose_state(os.path.join(args.log_dir, log_dir), franka)
        add_ee_pose_action(os.path.join(args.log_dir, log_dir), franka)
