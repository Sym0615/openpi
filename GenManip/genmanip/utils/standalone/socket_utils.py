"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import pickle
import socket
import struct


def send_message(send_socket: socket.socket, data: dict) -> None:
    serialized_data = pickle.dumps(data)
    message_size = struct.pack("Q", len(serialized_data))
    send_socket.sendall(message_size + serialized_data)


def _recv_all(conn: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        packet = conn.recv(size - len(data))
        if not packet:
            raise ConnectionError("Socket connection closed unexpectedly")
        data.extend(packet)
    return bytes(data)


def wait_message(conn: socket.socket) -> dict:
    payload_size = struct.calcsize("Q")
    packed_size = _recv_all(conn, payload_size)
    msg_size = struct.unpack("Q", packed_size)[0]
    frame_data = _recv_all(conn, msg_size)
    return pickle.loads(frame_data)


def create_send_port_and_wait(port: int) -> socket.socket:
    serial = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serial.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serial.bind(("localhost", port))
    serial.listen(1)
    print("Waiting for a connection...")
    conn, addr = serial.accept()
    print("Connected by", addr)
    return conn


def create_receive_port_and_attach(port: int) -> socket.socket:
    serial = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serial.connect(("localhost", port))
    print("connected port ", port)
    return serial
