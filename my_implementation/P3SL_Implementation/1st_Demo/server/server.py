# server has the command
# server tells each clients to set their models
# sends maximum split layer allowed
# training starts
# server each one by one, asks each client to train with them
# training ends,weighted model aggregation happens
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

    print("[SERVER] accept thread running", flush=True)

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

def orchestration_loop():
    # wait until N clients connected
    while True:
        with clients_lock:
            ids = list(clients.keys())
        if len(ids) >= 5:
            break
        time.sleep(0.1)

    # tell each client to initialize
    for cid in ids:
        send_to_client(cid, "RESET", None)
        # optionally wait for response
        # wait_for(cid, "RESET_OK", timeout=5)




#Potencial functions to add in script

def set_hyperparameters():
    pass
def set_model():
    pass
def reset_model():
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
    while True:
        time.sleep(5)
        x = 1
        if x == 1:
            print(clients)
            x = x + 1





