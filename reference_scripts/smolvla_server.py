"""
SmolVLA policy server. Run this in your `lerobot` conda env (NOT in Isaac Sim's python):

    conda activate lerobot
    python smolvla_server.py

It listens on 127.0.0.1:5555, receives camera images + joint state + instruction,
and returns the next action. The Isaac Sim script talks to it via smolvla_client.py.

Wire format: 8-byte big-endian length, then an .npz payload (no pickle).
"""

import argparse
import io
import socket
import struct
import traceback

import numpy as np
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors  # check name/path for your LeRobot version


# ---------------- wire helpers ----------------
def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("client disconnected")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock):
    n = struct.unpack(">Q", recv_exact(sock, 8))[0]
    return np.load(io.BytesIO(recv_exact(sock, n)), allow_pickle=False)


def send_msg(sock, **arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    data = buf.getvalue()
    sock.sendall(struct.pack(">Q", len(data)) + data)


# ---------------- policy wrapper ----------------
class PolicyServer:
    def __init__(self, model_id: str, device: str):
        self.device = device
        self.policy = SmolVLAPolicy.from_pretrained(model_id).to(device).eval()
        self.policy.reset()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            model_id,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        feats = self.policy.config.input_features
        print("Policy input features:", feats)
        # Image keys the checkpoint wants, e.g. observation.images.camera1..3
        self.image_keys = sorted(k for k in feats if k.startswith("observation.images."))
        print("Image keys:", self.image_keys)

    @staticmethod
    def _img(arr_uint8: np.ndarray) -> torch.Tensor:
        """HxWx3 uint8 -> 1x3xHxW float32 in [0,1]."""
        t = torch.from_numpy(arr_uint8.astype(np.float32) / 255.0)
        return t.permute(2, 0, 1).unsqueeze(0)

    @torch.inference_mode()
    def act(self, state, scene, wrist, task):
        obs = {"observation.state": torch.from_numpy(state.astype(np.float32)).unsqueeze(0)}

        # First image key gets the scene view, the rest get the wrist view.
        # Swap or edit this if your checkpoint expects a different assignment.
        for i, key in enumerate(self.image_keys):
            obs[key] = self._img(scene if i == 0 else wrist)
        obs["task"] = task

        obs = self.pre(obs)
        action = self.policy.select_action(obs)
        action = self.post(action)
        return action.squeeze(0).float().cpu().numpy()


def serve(server: PolicyServer, host: str, port: int):
    lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lsock.bind((host, port))
    lsock.listen(1)
    print(f"SmolVLA server ready on {host}:{port}")

    while True:
        conn, addr = lsock.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("Client connected:", addr)
        try:
            while True:
                msg = recv_msg(conn)
                cmd = str(msg["cmd"])
                try:
                    if cmd == "reset":
                        server.policy.reset()
                        send_msg(conn, ok=np.array(1))
                    elif cmd == "act":
                        action = server.act(
                            state=msg["state"],
                            scene=msg["scene"],
                            wrist=msg["wrist"],
                            task=str(msg["task"]),
                        )
                        send_msg(conn, action=action)
                    else:
                        send_msg(conn, error=np.array(f"unknown cmd {cmd}"))
                except Exception as e:  # keep the server alive on bad requests
                    traceback.print_exc()
                    send_msg(conn, error=np.array(str(e)))
        except ConnectionError:
            print("Client disconnected, waiting for a new one...")
        finally:
            conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lerobot/smolvla_base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    args = ap.parse_args()

    serve(PolicyServer(args.model, args.device), args.host, args.port)