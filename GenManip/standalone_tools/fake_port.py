"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import argparse
import pickle
import struct
import socket
import time
import numpy as np
from typing import Any


def send_message(send_socket: socket.socket, data: dict) -> None:
    """Send a message to the socket."""
    serialized_data = pickle.dumps(data)
    message_size = struct.pack("Q", len(serialized_data))
    send_socket.sendall(message_size + serialized_data)


def _recv_all(conn: socket.socket, size: int) -> bytes:
    """Receive all the data from the socket."""
    data = bytearray()
    while len(data) < size:
        packet = conn.recv(size - len(data))
        if not packet:
            raise ConnectionError("Socket connection closed unexpectedly")
        data.extend(packet)
    return bytes(data)


def wait_message(conn: socket.socket):
    """Wait for a message from the socket."""
    payload_size = struct.calcsize("Q")
    packed_size = _recv_all(conn, payload_size)
    msg_size = struct.unpack("Q", packed_size)[0]
    frame_data = _recv_all(conn, msg_size)
    return pickle.loads(frame_data)


def create_send_port_and_wait(port: int) -> socket.socket:
    """Create a send port and wait for a connection."""
    serial = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # set socket options, reuse address
    serial.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serial.bind(("localhost", port))
    serial.listen(1)
    print("Waiting for a connection...")
    conn, addr = serial.accept()
    print("Connected by", addr)
    return conn


def create_receive_port_and_attach(port: int) -> socket.socket:
    """Create a receive port and attach to the socket."""
    serial = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serial.connect(("localhost", port))
    print("connected port ", port)
    return serial


def generate_action(data: dict, agent: Any) -> tuple[np.ndarray, float]:
    """Generate an action from the data."""
    obs = {}
    obs["robot"] = {"qpos": data["joint_position_state"]}
    obs["obs_camera"] = {"color_image": data["obs_camera_rgb"]}
    obs["realsense"] = {"color_image": data["realsense_rgb"]}
    goal = data["instruction"]
    timestep = data["timestep"]
    reset = data["reset"]
    if reset:
        agent.reset()
    output, gripper, _ = agent.forward(obs, goal, timestep)
    result = np.array(output)
    return result, gripper


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--receive_port", type=int, default=10001)
parser.add_argument("-s", "--send_port", type=int, default=10000)
parser.add_argument("-a", "--arm_type", type=str, default="franka")
parser.add_argument("-g", "--gripper_type", type=str, default="panda_hand")
parser.add_argument("-c", "--control_type", type=str, default="joint_position")
args = parser.parse_args()

if __name__ == "__main__":
    # 1. create a send port to send action to GenManip and wait for GenManip to connect
    send_socket = create_send_port_and_wait(port=args.send_port)
    time.sleep(1)

    # 2. create a receive port to receive data from GenManip
    receive_socket = create_receive_port_and_attach(port=args.receive_port)

    # 3. while True, start the model inference loop
    while True:
        # 4. Receive data from GenManip
        data = wait_message(receive_socket)
        print("=" * 20, data["timestep"], "=" * 20)
        print("current instruction:", data["instruction"])
        print("current joint position:", data["joint_position_state"])
        print("current ee pose:", data["ee_pose_state"])

        # 5. Generate action, for model inference, replace with your own model inference code
        if args.arm_type == "franka":
            if args.gripper_type == "panda_hand":
                if args.control_type == "joint_position":
                    actions = {"action": [0.0] * 9}
                elif args.control_type == "ee_pose":
                    actions = {
                        "action": (
                            [0.001, 0.001, 0.001],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.04, 0.04],
                        )
                    }
                else:
                    raise ValueError("Invalid control type")
            elif args.gripper_type == "robotiq":
                if args.control_type == "joint_position":
                    actions = {"action": [0] * 13}
                elif args.control_type == "ee_pose":
                    actions = {
                        "action": (
                            [0.001, 0.001, 0.001],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.7853, 0.7853, -0.7853, -0.7853, -0.7853, -0.7853],
                        )
                    }
                else:
                    raise ValueError("Invalid control type")
            else:
                raise ValueError("Invalid gripper type")
        elif args.arm_type == "aloha":
            if args.gripper_type == "piper":
                if args.control_type == "joint_position":
                    actions = {"action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05] * 2}
                elif args.control_type == "ee_pose":
                    actions = {
                        "action": (
                            (
                                [0.001, 0.001, 0.001],
                                [1.0, 0.0, 0.0, 0.0],
                                [0.05, 0.05],
                            ),
                            (
                                [0.001, 0.001, 0.001],
                                [1.0, 0.0, 0.0, 0.0],
                                [0.05, 0.05],
                            ),
                        )
                    }
                else:
                    raise ValueError("Invalid control type")
            else:
                raise ValueError("Invalid gripper type")
        else:
            raise ValueError("Invalid arm type")
        print("action:", actions)

        # 6. Send action to GenManip
        send_message(send_socket, actions)
