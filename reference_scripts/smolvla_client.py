"""
Client for smolvla_server.py. Runs inside Isaac Sim's python; needs only numpy + stdlib.
Replaces smolvla_isaac_glue.py, same interface:

    from smolvla_client import SmolVLARunner
    runner = SmolVLARunner(instruction="pick up the cube")

    target = runner.act(
        joint_pos_rad=simulation.arm.get_joint_positions(),
        wrist_rgb=simulation.camera_wrist.get_rgba()[:, :, :3],
        scene_rgb=simulation.camera_scene.get_rgba()[:, :, :3],
    )
    simulation.arm.apply_action(ArticulationAction(joint_positions=target))
"""

import io
import socket
import struct

import numpy as np

# Index of each SO101 joint inside Isaac's dof ordering.
# Print `simulation.arm.dof_names` and edit so that position i here is the sim index of:
# shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
SIM_IDX_FOR_LEROBOT_JOINT = [0, 1, 2, 3, 4, 5]


# ---------------- wire helpers ----------------
def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("SmolVLA server closed the connection")
        buf.extend(chunk)
    return bytes(buf)


def _recv_msg(sock):
    n = struct.unpack(">Q", _recv_exact(sock, 8))[0]
    return np.load(io.BytesIO(_recv_exact(sock, n)), allow_pickle=False)


def _send_msg(sock, **arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    data = buf.getvalue()
    sock.sendall(struct.pack(">Q", len(data)) + data)


def _to_uint8(img) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    arr = arr.astype(np.float32)
    if arr.max() <= 1.5:
        arr = arr * 255.0
    return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))


# ---------------- runner ----------------
class SmolVLARunner:
    def __init__(self, instruction: str, lower, upper, host: str = "127.0.0.1", port: int = 5555, timeout: float = 60.0):
        self.lower = np.asarray(lower, dtype=np.float32)
        self.upper = np.asarray(upper, dtype=np.float32)
        self.instruction = instruction
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"Connected to SmolVLA server at {host}:{port}")

    # ----- conversions (edit these to match your checkpoint's calibration) -----

    def to_policy_state(self, q):
        frac = (np.asarray(q, np.float32) - self.lower) / (self.upper - self.lower)
        out = frac * 200.0 - 100.0
        out[5] = frac[5] * 100.0               # gripper is 0..100
        return out.astype(np.float32)

    def to_sim(self, a, current_sim_rad):
        a = np.asarray(a, np.float32)
        frac = (a + 100.0) / 200.0
        frac[5] = a[5] / 100.0
        return self.lower + np.clip(frac, 0.0, 1.0) * (self.upper - self.lower)

    # ----- API -----
    def act(self, joint_pos_rad, wrist_rgb, scene_rgb) -> np.ndarray:
        _send_msg(
            self.sock,
            cmd=np.array("act"),
            state=self.to_policy_state(joint_pos_rad),
            scene=_to_uint8(scene_rgb),
            wrist=_to_uint8(wrist_rgb),
            task=np.array(self.instruction),
        )
        reply = _recv_msg(self.sock)
        print("raw action:", np.round(reply["action"], 1))
        if "error" in reply.files:
            raise RuntimeError(f"SmolVLA server error: {reply['error']}")
        return self.to_sim(reply["action"], joint_pos_rad)

    def reset(self):
        """Call when you reset the episode (clears the server's action queue)."""
        _send_msg(self.sock, cmd=np.array("reset"))
        reply = _recv_msg(self.sock)
        if "error" in reply.files:
            raise RuntimeError(f"SmolVLA server error: {reply['error']}")

    def close(self):
        self.sock.close()