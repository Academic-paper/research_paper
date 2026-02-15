#imports
import os
import io
import numpy as np
import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
import socket
import pickle
import struct
import threading
import re


pending_bwd = {}  # job_id -> payload (if BWD arrives before TRAIN_JOB finishes)
pending_lock = threading.Lock()

#socket helper functions
def send_msg(sock, obj):
    data = pickle.dumps(obj)
    length = struct.pack("!I", len(data))
    sock.sendall(length + data)

def recv_msg(sock):
    try:
        raw_len = sock.recv(4)
        if raw_len == b'':
            return "__DISCONNECT__"
    except socket.error:
        return "__ERROR__"

    msg_len = struct.unpack("!I", raw_len)[0]

    data = b""
    while len(data) < msg_len:
        chunk = sock.recv(msg_len - len(data))
        if chunk == b'':
            return "__DISCONNECT__"
        data += chunk

    return pickle.loads(data)

def serialize_tensor(tensor):
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()

def deserialize_tensor(byte_data):
    buffer = io.BytesIO(byte_data)
    return torch.load(buffer, map_location="cpu")


client_id_assigned = None
server_host = "server"
server_port = 5000

def establish_connection(server_host, server_port):
    global client_id_assigned
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    while True:
        try:
            sock.connect((server_host, server_port))
            msg = recv_msg(sock)
            if msg == "__DISCONNECT__" or msg == "__ERROR__":
                raise RuntimeError("Disconnected before ID assignment")

            if msg["cmd"] != "ASSIGN_ID":
                raise RuntimeError(f"Expected ASSIGN_ID, got {msg}")

            client_id_assigned = msg["payload"]["client_id"]
            break
        except ConnectionRefusedError:
            print("[Client] Waiting for server")
            time.sleep(2)

    return sock

sock = establish_connection(server_host, server_port)


model = None
optimizer = None
train_ds = None

head_opts = {}          # split_layer -> optimizer
head_opts_lock = threading.Lock()


class P3SLModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, 10),
        ])

    def forward_from(self, x, split_layer):
        """
        Continue forward pass from split_layer+1 to end
        """
        for i in range(split_layer + 1, len(self.layers)):
            x = self.layers[i](x)
        return x

    def forward_upto(self, x, split_layer):
        """
        Forward pass from start up to split_layer
        """
        for i in range(0, split_layer + 1):
            x = self.layers[i](x)
        return x

def command_loop(sock):
    while True:
        msg = recv_msg(sock)

        if msg in ("__DISCONNECT__", "__ERROR__"):
            print("[CLIENT] Server disconnected")
            break

        cmd = msg["cmd"]
        payload = msg.get("payload")

        if cmd == "SET_MODEL":
            handle_set_model(payload)
        elif cmd == "RESET":
            handle_reset(payload)
        elif cmd == "TRAIN_JOB":
            handle_train(payload)
        elif cmd == "STOP":
            break
        elif cmd == "LOAD_DATASET":
            load_dataset_on_client(payload)
        elif cmd == "BWD":
            # if BWD arrives outside training (rare but possible), buffer it
            pl = payload
            with pending_lock:
                pending_bwd[pl["job_id"]] = pl
        elif cmd == "GET_HEAD_WEIGHTS":
            handle_get_head_weights(payload)
        else:
            print(f"[CLIENT] Unknown command: {cmd}")

def handle_set_model(payload):
    print("[CLIENT] Setting model config")

def handle_reset(payload):
    global model, optimizer, head_opts
    model = P3SLModel()

    # IMPORTANT: new model => old optimizers invalid
    with head_opts_lock:
        head_opts = {}

    split_layer = payload["split_layer"]
    optimizer = get_head_optimizer(split_layer)
    send_msg(sock, {"cmd": "RESET_OK", "payload": {"client_id": client_id_assigned}})


def load_dataset_on_client(payload):
    global train_ds
    # payload root should be "/data/FashionMNIST"
    root = payload["dir"]
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.FashionMNIST(root, train=True, download=False, transform=tfm)
    print(f"[Client] Loaded FashionMNIST with N={len(train_ds)}", flush=True)

def get_x_batch(indices):
    """
    indices: list[int]
    returns: Tensor [B,1,28,28]
    """
    xs = []
    for idx in indices:
        x, _ = train_ds[idx]     # label ignored (server owns labels)
        xs.append(x)
    return torch.stack(xs, dim=0)

def head_parameters_for_split(split_layer):
    params = []
    for i in range(0, split_layer + 1):
        params += list(model.layers[i].parameters())
    return params

def get_head_optimizer(split_layer, lr=0.01, momentum=0.9):
    global head_opts
    with head_opts_lock:
        if split_layer not in head_opts:
            params = head_parameters_for_split(split_layer)
            head_opts[split_layer] = optim.SGD(params, lr=lr, momentum=momentum)
        return head_opts[split_layer]



def handle_train(payload):
    """
    payload = {"job_id": str, "indices": list[int], "split_layer": int}
    """
    global model

    job_id = payload["job_id"]
    indices = payload["indices"]
    split_layer = payload["split_layer"]

    if train_ds is None:
        raise RuntimeError("train_ds not loaded. Call RESET first or implement LOAD_DATA.")

    # 1) build x batch
    x = get_x_batch(indices)

    # 2) forward client side to split layer
    model.train()
    optimizer = get_head_optimizer(split_layer)
    optimizer.zero_grad()

    # IMPORTANT: IR must require grad for boundary backprop
    ir = model.forward_upto(x, split_layer)
    if not torch.isfinite(ir).all():
        print("IR has NaN/Inf");
        return

    # 3) send IR to server
    send_msg(sock, {
        "cmd": "IR",
        "payload": {
            "job_id": job_id,
            "ir": serialize_tensor(ir.detach())
        }
    })

    # 4) wait for matching BWD
    bwd_payload = None

    # if BWD already arrived early, consume it
    with pending_lock:
        if job_id in pending_bwd:
            bwd_payload = pending_bwd.pop(job_id)

    # otherwise block until we get it
    while bwd_payload is None:
        msg = recv_msg(sock)
        if msg in ("__DISCONNECT__", "__ERROR__"):
            print("[CLIENT] Server disconnected during train", flush=True)
            return

        cmd = msg.get("cmd")
        pl = msg.get("payload")

        if cmd == "BWD":
            if pl["job_id"] == job_id:
                bwd_payload = pl
            else:
                # store for later (another job)
                with pending_lock:
                    pending_bwd[pl["job_id"]] = pl
        else:
            # other commands arriving while training; you can handle or ignore
            print(f"[CLIENT] got {cmd} while waiting BWD", flush=True)

    # 5) boundary backprop into client layers
    grad = deserialize_tensor(bwd_payload["grad"])
    ir.backward(grad)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    p = next(model.parameters())
    gn = float(p.grad.norm().item()) if p.grad is not None else 0.0
    print("client grad norm:", gn, flush=True)

    # optional ack so server can print per-step progress
    send_msg(sock, {
        "cmd": "STEP_OK",
        "payload": {"job_id": job_id}
    })

    print(f"[CLIENT {client_id_assigned}] STEP_OK job={job_id} loss={bwd_payload.get('loss')}", flush=True)

_layer_key_re = re.compile(r"^layers\.(\d+)\.")

def layer_index_from_key(k: str):
    m = _layer_key_re.match(k)
    return int(m.group(1)) if m else None

def is_key_upto_layer(k: str, upto: int) -> bool:
    idx = layer_index_from_key(k)
    return (idx is not None) and (idx <= upto)

def handle_get_head_weights(payload):
    smax = payload["smax"]
    split_layer = payload["split_layer"]
    upto = min(split_layer, smax)

    sd = model.state_dict()
    head_sd = {k: v.cpu() for k, v in sd.items() if is_key_upto_layer(k, upto)}

    send_msg(sock, {
        "cmd": "HEAD_WEIGHTS",
        "payload": {
            "client_id": client_id_assigned,
            "split_layer": split_layer,
            "state_dict": head_sd
        }
    })


#potencial functions to implement in script:
def handle_set_global_head(payload):
    smax = payload["smax"]
    head_sd = payload["state_dict"]

    sd = model.state_dict()
    # overwrite keys we received
    for k, v in head_sd.items():
        sd[k] = v
    model.load_state_dict(sd)

    # IMPORTANT: optimizers are now stale -> clear cache
    global head_opts
    with head_opts_lock:
        head_opts = {}

    send_msg(sock, {"cmd": "SET_GLOBAL_HEAD_OK", "payload": {"client_id": client_id_assigned}})



if __name__ == '__main__':
    command_loop(sock)





