# server has the command
# server tells each clients to set their models
# sends maximum split layer allowed
# training starts
# server each one by one, asks each client to train with them
# training ends,weighted model aggregation happens
import random

print("[SERVER] server.py started", flush=True)


#imports
import os
import io
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim
from torch.autograd import Variable
import threading, time, queue, struct, pickle, socket, time

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
    return torch.load(buffer)


#Deciding orchestration logic
#             ┌────────────┐
#             │Orchestrator│
#             │  Server    │
#             └─────┬──────┘
#    ┌──────────────┼──────────────┐
#    ▼              ▼              ▼
#  ClientA        Client B      Client C

# ┌─────────────────────────┐
# │       SERVER PROCESS    │
# │                         │
# │ ┌──────── Accept Thread │  ← blocks on accept()
# │ │                       │
# │ └───────────────┐       │
# │                 │       │
# │┌──── Orchestrator Thread│ ← sends TRAIN / waits for ACKs
# ││                        │
# │└───────────────┐        │
# │                │        │
# │┌─ Client Thread (C1)    │ ← blocks on recv(C1)
# │├─ Client Thread (C2)    │ ← blocks on recv(C2)
# |├─ Client Thread (C3)    │ ← blocks on recv(C3)
# │└────────────────────────┘
# └─────────────────────────┘

#MAIN THREAD
# └─ orchestration logic (TRAIN / AGGREGATE / TEST)

#ACCEPT THREAD
# └─ accept() → register client → start listener

#CLIENT LISTENER THREADS
# └─ recv() per client → handle messages


clients = {}
clients_lock = threading.Lock()

def register_client(client_id, conn, addr):
    return {
        "id": client_id,
        "conn": conn,
        "addr": addr,
        "ready": False,
        "inbox": queue.Queue(),          # messages from this client
        "send_lock": threading.Lock(),   # protect send_msg on this conn
        "last_seen": time.time(),
        "config": {},                    # per-client settings
    }

MAX_CLIENTS = 5
next_client_id = 0

# Server orchestration helper functions
def client_listener(client_id):
    with clients_lock:
        session = clients.get(client_id)
    if not session:
        return

    conn = session["conn"]
    while True:
        msg = recv_msg(conn)
        if msg in ("__DISCONNECT__", "__ERROR__"):
            print(f"[SERVER] client {client_id} disconnected", flush=True)
            break

        session["last_seen"] = time.time()
        session["inbox"].put(msg)   # ✅ push to inbox

    remove_client(client_id)


def accept_clients(server_socket):
    global next_client_id

    while True:
        conn, addr = server_socket.accept()

        with clients_lock:
            if len(clients) >= MAX_CLIENTS:
                conn.close()
                continue

            client_id = next_client_id
            next_client_id += 1

            clients[client_id] = register_client(client_id, conn, addr)

        send_msg(conn, {"cmd": "ASSIGN_ID", "payload": {"client_id": client_id}})
        print(f"[SERVER] Client {client_id} connected from {addr}", flush=True)

        threading.Thread(target=client_listener, args=(client_id,), daemon=True).start()

#also sus placeholder
def remove_client(client_id):
    with clients_lock:
        if client_id in clients:
            try:
                clients[client_id]["conn"].close()
            except:
                pass
            del clients[client_id]

    print(f"[SERVER] Client {client_id} removed")


server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(("0.0.0.0", 5000))
server_socket.listen()


# Maybe placeholder
def handle_client_message(client_id, msg):
    cmd = msg.get("cmd")
    payload = msg.get("payload")

    print(f"[SERVER] From client {client_id}: {cmd}")

    if cmd == "READY":
        with clients_lock:
            clients[client_id]["ready"] = True


def send_command(client_id, cmd, payload=None):
    try:
        send_msg(clients[client_id]["conn"], {
            "cmd": cmd,
            "payload": payload
        })
    except Exception:
        print("Client removed")
        remove_client(client_id)


def broadcast(cmd, payload=None):
    with clients_lock:
        for cid in list(clients.keys()):
            send_command(cid, cmd, payload)


def send_to_client(client_id, cmd, payload=None):
    with clients_lock:
        session = clients.get(client_id)
    if not session:
        return False

    with session["send_lock"]:
        send_msg(session["conn"], {"cmd": cmd, "payload": payload})
    return True

def wait_for(client_id, expected_cmd, timeout=10):
    with clients_lock:
        session = clients.get(client_id)
    if not session:
        return None

    deadline = time.time() + timeout

    while time.time() < deadline:
        remaining = deadline - time.time()
        try:
            msg = session["inbox"].get(timeout=min(0.2, remaining))
        except queue.Empty:
            continue

        # ignore disconnect markers if you push them (optional)
        if msg in ("__DISCONNECT__", "__ERROR__"):
            return None

        cmd = msg.get("cmd")
        if cmd == expected_cmd:
            return msg
        else:
            # If you want, you can store "unexpected" messages somewhere.
            # For now just print/ignore.
            print(f"[SERVER] client {client_id} sent {cmd} while waiting for {expected_cmd}", flush=True)

    return None

# Machine learning part

model = None
optimizer = None

class P3SLModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([
            # ----- Block 0 -----
            nn.Conv2d(1, 32, 3),   # L0
            nn.ReLU(),             # L1
            nn.Conv2d(32, 32, 3),  # L2
            nn.ReLU(),             # L3
            nn.MaxPool2d(2),       # L4

            # ----- Block 1 -----
            nn.Conv2d(32, 64, 3),  # L5
            nn.ReLU(),             # L6
            nn.Conv2d(64, 64, 3),  # L7
            nn.ReLU(),             # L8
            nn.MaxPool2d(2),       # L9

            # ----- Block 2 -----
            nn.Flatten(),          # L10
            nn.Linear(64*5*5, 128),# L11
            nn.ReLU(),             # L12
            nn.Linear(128, 64),    # L13
            nn.ReLU(),             # L14
            nn.Linear(64, 10),     # L15
            nn.LogSoftmax(dim=1)   # L16
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

def reset_server_model():
    global model, optimizer
    model = P3SLModel()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    print("[SERVER] Server model reset", flush=True)


def wait_for_n_clients(n, timeout=None):
    start = time.time()
    while True:
        with clients_lock:
            ids = list(clients.keys())

        if len(ids) >= n:
            return ids

        if timeout is not None and (time.time() - start) > timeout:
            raise TimeoutError(f"Only {len(ids)} clients connected, needed {n}")

        time.sleep(0.1)


def reset_everything(n_clients=5):
    # 1) reset server model
    reset_server_model()

    # 2) wait for clients
    ids = wait_for_n_clients(n_clients)

    # 3) tell each client to reset
    for cid in ids:
        #storing client split layer information
        clients[cid]["config"] = {"split_layer": random.randint(2,6)}
        send_to_client(cid, "RESET", None)

    # 4) wait for RESET_OK from each client
    for cid in ids:
        ack = wait_for(cid, "RESET_OK", timeout=10)
        if ack is None:
            print(f"[SERVER] ❌ Client {cid} failed to RESET", flush=True)
        else:
            print(f"[SERVER] ✅ Client {cid} RESET_OK", flush=True)

    print("[SERVER] Sent RESET to all clients", flush=True)



def orchestration_loop():
    #we are currently hard passing the number of clients in the reset function
    reset_everything()

        # Rule 4 make plans to how u gonna split the dataset into workable so that each client can train in their liking


#Potencial functions to add in script

def set_hyperparameters():
    pass
def set_model():
    pass
def train():
    pass
def aggregate_model():
    pass
def test():
    pass



if __name__ == '__main__':
    accept_thread = threading.Thread(
        target=accept_clients,
        args=(server_socket,),
        daemon=True
    )
    accept_thread.start()
    orchestration_loop()
    while True:
        time.sleep(5)






