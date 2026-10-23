"""Observation/action adapters and a local NPZ pipe protocol for ABC and OpenPI."""

import io
import json
import struct

import numpy as np

CAMERA_MAP = {"top": "base_0_rgb", "left": "left_wrist_0_rgb", "right": "right_wrist_0_rgb"}
STATS_GROUP = "dataset/abc_sim_bottles"
MAX_PACKET_BYTES = 16 * 1024 * 1024


def observation_to_openpi(obs, *, canonical_adapter=None, stats_group=STATS_GROUP):
    state = np.asarray(obs["state"], dtype=np.float32)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("ABC state must be a finite (14,) policy-space vector.")
    images = {}
    for source, target in CAMERA_MAP.items():
        rgb = np.asarray(obs["images"][source])
        if rgb.ndim != 3 or rgb.shape[0] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"ABC camera {source} must be CHW uint8 RGB, got {rgb.shape}/{rgb.dtype}.")
        if not bool(obs["masks"][source]):
            raise ValueError(f"ABC camera {source} is unavailable.")
        images[target] = np.ascontiguousarray(rgb.transpose(1, 2, 0))
    mapped_state = np.pad(state, (0, 18))
    padding = np.arange(32) >= 14
    if canonical_adapter is not None:
        # Use the training reader's projection, including FK and dimension masks.
        import torch

        from openpi.training.petrel_lerobot_reader import PetrelLeRobotDataset

        parquet = {"observation.state": torch.from_numpy(state.copy()).unsqueeze(0)}
        for side, joint_slice, gripper_index in (("left", slice(0, 6), 6), ("right", slice(7, 13), 13)):
            parquet[f"states.{side}_joint.position"] = torch.from_numpy(state[joint_slice].copy()).unsqueeze(0)
            parquet[f"states.{side}_gripper.position"] = torch.tensor([[state[gripper_index]]])
        dataset = object.__new__(PetrelLeRobotDataset)
        dataset.name = "abc_live_observation"
        dataset.use_canonical_adapter = True
        dataset.canonical_dim = 32
        dataset.canonical_adapter = canonical_adapter
        mapped, padding = dataset._canonical_project(parquet, [0], "state")  # noqa: SLF001
        mapped_state = mapped[0].numpy()
        padding = padding.numpy()
    return {
        "state": mapped_state,
        "state_dim_is_pad": padding,
        "stats_group": stats_group,
        "stats_group_fallbacks": (),
        "prompt": str(obs["prompt"]),
        "image": images,
        "image_mask": dict.fromkeys(images, True),
    }


def actions_to_abc(actions, *, reference_state=None, delta_indices=()):
    """Restore joint delta labels to absolute targets using the chunk's initial state."""
    actions = np.asarray(actions)
    if actions.shape != (30, 32) or not np.isfinite(actions).all():
        raise ValueError("OpenPI actions must be a finite (30, 32) array after unnormalization.")
    active = actions[:, :14].astype(np.float32, copy=True)
    if delta_indices:
        reference = np.asarray(reference_state, dtype=np.float32)
        if reference.shape != (14,) or not np.isfinite(reference).all():
            raise ValueError("Joint delta actions require a finite (14,) reference_state.")
        indices = np.asarray(delta_indices, dtype=np.int64)
        if not np.isin(indices, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]).all():
            raise ValueError("delta_indices must select ABC joint dimensions.")
        active[:, indices] += reference[indices]
    grippers = active[:, [6, 13]].copy()
    active[:, [6, 13]] = np.clip(grippers, 0.0, 1.0)
    clipped = int(np.count_nonzero(active[:, [6, 13]] != grippers))
    return active, clipped


def send_packet(stream, metadata, arrays=None):
    buffer = io.BytesIO()
    np.savez(buffer, message_json=np.array(json.dumps(metadata)), **(arrays or {}))
    payload = buffer.getvalue()
    if len(payload) > MAX_PACKET_BYTES:
        raise ValueError("Pipe packet exceeds maximum size.")
    stream.write(struct.pack("!I", len(payload)))
    stream.write(payload)
    stream.flush()


def _read_exact(stream, size):
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("OpenPI/ABC pipe closed before the packet was complete.")
        chunks.extend(chunk)
    return chunks


def receive_packet(stream):
    size = struct.unpack("!I", _read_exact(stream, 4))[0]
    if size > MAX_PACKET_BYTES:
        raise ValueError("Pipe packet exceeds maximum size.")
    with np.load(io.BytesIO(_read_exact(stream, size)), allow_pickle=False) as packet:
        metadata = json.loads(str(packet["message_json"].item()))
        arrays = {key: packet[key].copy() for key in packet.files if key != "message_json"}
    return metadata, arrays
