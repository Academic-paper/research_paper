# server has the command
# server tells each clients to set their models
# sends maximum split layer allowed
# training starts
# server each one by one, asks each client to train with them
# training ends,weighted model aggregation happens
import random
import re

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


# =====================================================================
# P3SL: SERVER-SIDE BI-LEVEL OPTIMIZATION (NOISE MANAGEMENT)
# =====================================================================

TARGET_ACCURACY = 90.0  # Amin from the paper: Minimum acceptable accuracy
MAX_SPLIT_LAYER = 10    # Smax from the paper

def get_initial_noise_table():
    """
    Starts all split points with the maximum privacy protection (Noise = 2.5)
    as defined in the P3SL profiling tables.
    """
    return {layer: 2.5 for layer in range(1, MAX_SPLIT_LAYER + 1)}

# def update_noise_table(current_table, current_accuracy):
#     """
#     Updates the noise table using the exact formula from Section 5.2:
#     sigma_{t+1} = sigma_t * (1 - 2 * (A_min - A^t))
#     """
#     print(f"\n[SERVER] Evaluating Accuracy: {current_accuracy}% vs Target: {TARGET_ACCURACY}%")
    
#     if current_accuracy >= TARGET_ACCURACY:
#         print("[SERVER] ✅ Target accuracy reached! Sweet spot locked in.")
#         return current_table  # Keep the current noise budgets
        
#     print("[SERVER] ⚠️ Accuracy too low. Recalculating privacy budgets...")
#     new_table = {}
    
#     # Convert percentages to decimals for the math
#     a_min = TARGET_ACCURACY / 100.0
#     a_t = current_accuracy / 100.0
    
#     for layer, sigma in current_table.items():
#         # Apply the paper's formula
#         new_sigma = sigma * (1 - 2 * (a_min - a_t))
        
#         # Ensure noise doesn't drop below 0
#         new_table[layer] = max(0.0, new_sigma)
        
#     return new_table


def update_noise_table(current_table, current_accuracy):
    """
    Gradually decays the noise table by 10% per round if accuracy is not met,
    preventing the privacy budget from instantly crashing to 0.0.
    """
    print(f"\n[SERVER] Evaluating Accuracy: {current_accuracy}% vs Target: {TARGET_ACCURACY}%")
    
    if current_accuracy >= TARGET_ACCURACY:
        print("[SERVER] ✅ Target accuracy reached! Sweet spot locked in.")
        return current_table  # Keep the current noise budgets
        
    print("[SERVER] ⚠️ Accuracy too low. Gradually reducing privacy budgets...")
    new_table = {}
    
    for layer, sigma in current_table.items():
        # Reduce the noise by 10% (multiply by 0.9) to safely walk down the trade-off curve
        new_sigma = sigma * 0.90
        
        # Ensure noise doesn't drop below 0
        new_table[layer] = max(0.0, new_sigma)
        
    return new_table

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
Smax = 6

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

def reset_server_model():
    global model, tail_opts
    model = P3SLModel()
    tail_opts = {}  # clear cached optimizers since model changed
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
        with clients_lock:
            #storing client split layer information
            clients[cid]["config"] = {"split_layer": 5}  # or randomize per client if you want
        send_to_client(cid, "RESET", {"split_layer": clients[cid]["config"]["split_layer"]})

    # 4) wait for RESET_OK from each client
    for cid in ids:
        ack = wait_for(cid, "RESET_OK", timeout=10)
        if ack is None:
            print(f"[SERVER] ❌ Client {cid} failed to RESET", flush=True)
        else:
            print(f"[SERVER] ✅ Client {cid} RESET_OK", flush=True)

    print("[SERVER] Sent RESET to all clients", flush=True)

def dataset_transformations(dataset: str):
    if dataset == "MNIST":
        return transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])


train_ds = None
criterion = nn.CrossEntropyLoss() # because model ends with logsoftmaxx
# def load_dataset_on_server():
#     global train_ds
#     root = "/data"
#     tfm = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#     train_ds = datasets.FashionMNIST(root, train=True, download=True, transform=tfm)
#     print(f"[SERVER] Loaded FashionMNIST with N={len(train_ds)}", flush=True)


def load_dataset_on_server():
    global train_ds, test_ds
    root = "/data"
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.FashionMNIST(root, train=True, download=True, transform=tfm)
    test_ds = datasets.FashionMNIST(root, train=False, download=True, transform=tfm)
    print(f"[SERVER] Loaded FashionMNIST Train={len(train_ds)}, Test={len(test_ds)}", flush=True)

def evaluate_global_accuracy():
    print("\n[SERVER] Evaluating Global Model Accuracy...")
    correct = 0
    total = 0
    device = next(model.parameters()).device
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass through the fully aggregated global model
            x = images
            for layer in model.layers:
                x = layer(x)
            out = x
            
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    return accuracy


def assign_shards_to_clients(ids):
    assert train_ds is not None
    N = len(train_ds)
    shards = np.array_split(np.arange(N), len(ids))

    with clients_lock:
        for k, cid in enumerate(ids):
            clients[cid]["config"]["shard_indices"] = shards[k].tolist()
            clients[cid]["config"]["step"] = 0  # pointer into shard
            # you already do split_layer randomization; keep it:
            if "split_layer" not in clients[cid]["config"]:
                clients[cid]["config"]["split_layer"] = random.randint(2, 6)

    print("[SERVER] Shards assigned", flush=True)

def next_batch_indices_for_client(cid, batch_size):
    with clients_lock:
        shard = clients[cid]["config"]["shard_indices"]
        step = clients[cid]["config"]["step"]

    start = step * batch_size
    end = start + batch_size
    if start >= len(shard):
        return None, None

    batch = shard[start:end]

    with clients_lock:
        clients[cid]["config"]["step"] += 1

    return batch, step


# def run_train_step_for_client(cid, batch_size=64, timeout=30):
#     split_layer = clients[cid]["config"]["split_layer"]
#     indices, step_used = next_batch_indices_for_client(cid, batch_size)
#     if indices is None:
#         return False  # done
#     job_id = f"{cid}:{step_used}"

#     # 1) tell client exactly what to train on
#     send_to_client(cid, "TRAIN_JOB", {
#         "job_id": job_id,
#         "indices": indices,
#         "split_layer": split_layer
#     })

#     # 2) wait for client IR
#     msg = wait_for(cid, "IR", timeout=timeout)
#     if msg is None:
#         print(f"[SERVER] ❌ No IR from client {cid} for job {job_id}", flush=True)
#         return False

#     payload = msg["payload"]
#     if payload["job_id"] != job_id:
#         print(f"[SERVER] ⚠️ job mismatch: expected {job_id} got {payload['job_id']}", flush=True)
#         return False

#     device = next(model.parameters()).device
#     ir = deserialize_tensor(payload["ir"]).float().to(device).requires_grad_(True)

#     # 3) server gets labels by deterministic indexing
#     # FashionMNIST: train_ds.targets is a tensor of size N

#     # build an optimizer for THIS split (or cache it; see below)
#     tail_opt = get_tail_optimizer(split_layer, lr=0.01, momentum=0.9)
#     tail_opt.zero_grad()

#     # 4) forward server-side part + loss
#     out = model.forward_from(ir, split_layer)
#     if not torch.isfinite(out).all():
#         print("OUT has NaN/Inf");
#         return

#     # labels (deterministic indexing)
#     idx_cpu = torch.as_tensor(indices, dtype=torch.long)  # CPU
#     labels = train_ds.targets[idx_cpu].long().to(device)  # move labels after indexing

#     loss = criterion(out, labels)
#     if not torch.isfinite(loss):
#         print("LOSS is NaN/Inf");
#         return

#     # 5) backprop to IR
#     loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#     tail_opt.step()
#     grad = ir.grad.detach()

#     # i need to do the backprogation in server itself till i reach the client split layer point then pass the further back propogation to client
#     # code below is wrong
#     send_to_client(cid, "BWD", {
#         "job_id": job_id,
#         "grad": serialize_tensor(grad),
#         "loss": float(loss.item())
#     })
#     ok = wait_for(cid, "STEP_OK", timeout=10)
#     if ok:
#         print(f"[SERVER] client {cid} finished {job_id}", flush=True)

#     return True


def run_train_step_for_client(cid, current_noise_table, batch_size=64, timeout=30):
    indices, step_used = next_batch_indices_for_client(cid, batch_size)
    if indices is None:
        return False  # done
    job_id = f"{cid}:{step_used}"

    # 1) Send the Noise Table instead of forcing a split layer!
    send_to_client(cid, "TRAIN_JOB", {
        "job_id": job_id,
        "indices": indices,
        "noise_table": current_noise_table
    })

    # 2) wait for client IR
    msg = wait_for(cid, "IR", timeout=timeout)
    if msg is None:
        print(f"[SERVER] ❌ No IR from client {cid} for job {job_id}", flush=True)
        return False

    payload = msg["payload"]
    if payload["job_id"] != job_id:
        print(f"[SERVER] ⚠️ job mismatch: expected {job_id} got {payload['job_id']}", flush=True)
        return False

    # ---------------------------------------------------------
    # NEW P3SL LOGIC: Read the split_layer the client dynamically chose
    # ---------------------------------------------------------
    split_layer = payload["split_layer"]
    clients[cid]["config"]["split_layer"] = split_layer # Save for aggregation later

    device = next(model.parameters()).device
    ir = deserialize_tensor(payload["ir"]).float().to(device).requires_grad_(True)

    # 3) server gets labels by deterministic indexing
    tail_opt = get_tail_optimizer(split_layer, lr=0.01, momentum=0.9)
    tail_opt.zero_grad()

    # 4) forward server-side part + loss
    out = model.forward_from(ir, split_layer)
    if not torch.isfinite(out).all():
        print("OUT has NaN/Inf")
        return

    idx_cpu = torch.as_tensor(indices, dtype=torch.long)
    labels = train_ds.targets[idx_cpu].long().to(device)

    loss = criterion(out, labels)
    if not torch.isfinite(loss):
        print("LOSS is NaN/Inf")
        return

    # 5) backprop to IR
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    tail_opt.step()
    grad = ir.grad.detach()

    send_to_client(cid, "BWD", {
        "job_id": job_id,
        "grad": serialize_tensor(grad),
        "loss": float(loss.item())
    })
    ok = wait_for(cid, "STEP_OK", timeout=10)
    if ok:
        print(f"[SERVER] client {cid} finished {job_id} at Split Layer {split_layer}", flush=True)

    return True

def tail_parameters_for_split(split_layer: int):
    # params in layers split_layer+1 ... end
    params = []
    for i in range(split_layer + 1, len(model.layers)):
        params += list(model.layers[i].parameters())
    return params

tail_opts = {}
def get_tail_optimizer(split_layer, lr=0.01, momentum=0.9):
    if split_layer not in tail_opts:
        tail_opts[split_layer] = make_tail_optimizer(split_layer, lr=lr, momentum=momentum)
    return tail_opts[split_layer]


def make_tail_optimizer(split_layer: int, lr=0.01, momentum=0.9):
    params = tail_parameters_for_split(split_layer)
    # some layers (ReLU, Flatten) have no params; that's ok
    return optim.SGD(params, lr=lr, momentum=momentum)

def load_dataset_on_clients():
    with clients_lock:
        ids = list(clients.keys())

    for cid in ids:
        send_to_client(cid, "LOAD_DATASET", {
            "dir": "/data"
        })

_layer_key_re = re.compile(r"^layers\.(\d+)\.")

def layer_index_from_key(k: str):
    m = _layer_key_re.match(k)
    return int(m.group(1)) if m else None

def is_key_upto_layer(k: str, upto: int) -> bool:
    idx = layer_index_from_key(k)
    return (idx is not None) and (idx <= upto)

def aggregate_p3sl_uniform(head_payloads, smax: int, N_expected: int):
    global_sd = model.state_dict()

    # keys we aggregate: params/buffers for layers <= smax
    agg_keys = [k for k in global_sd.keys() if is_key_upto_layer(k, smax)]

    # accumulator
    acc = {k: torch.zeros_like(global_sd[k], dtype=torch.float32) for k in agg_keys}

    # for each client payload, add client weight if present else global weight
    for pl in head_payloads:
        client_sd = pl["state_dict"]
        for k in agg_keys:
            v = client_sd.get(k, global_sd[k])     # PAD missing with server/global
            acc[k] += v.to(dtype=torch.float32)

    # divide by total clients N (even if some payload missing you probably want to fail loudly)
    if len(head_payloads) != N_expected:
        print(f"[SERVER] Warning: got {len(head_payloads)} payloads, expected {N_expected}")

    for k in agg_keys:
        global_sd[k] = (acc[k] / float(N_expected)).to(dtype=global_sd[k].dtype)

    model.load_state_dict(global_sd)

def request_head_weights(ids, smax: int, timeout=20):
    head_payloads = []

    # 1) send request
    for cid in ids:
        with clients_lock:
            split_layer = clients[cid]["config"]["split_layer"]
        send_to_client(cid, "GET_HEAD_WEIGHTS", {
            "smax": smax,
            "split_layer": split_layer,
        })

    # 2) collect responses
    for cid in ids:
        msg = wait_for(cid, "HEAD_WEIGHTS", timeout=timeout)
        if msg is None:
            print(f"[SERVER] ❌ No HEAD_WEIGHTS from client {cid}", flush=True)
            continue
        head_payloads.append(msg["payload"])

    return head_payloads



# def orchestration_loop():
#     reset_everything(n_clients=5)
#     load_dataset_on_server()
#     load_dataset_on_clients()

#     ids = wait_for_n_clients(5)
#     assign_shards_to_clients(ids)

#     epochs = 2

#     for ep in range(epochs):
#         with clients_lock:
#             for cid in ids:
#                 clients[cid]["config"]["step"] = 0
#                 random.shuffle(clients[cid]["config"]["shard_indices"])  # optional but recommended
#         active = set(ids)
#         while active:
#             for cid in list(active):
#                 ok = run_train_step_for_client(cid, batch_size=64, timeout=30)
#                 if not ok:
#                     active.remove(cid)

#     print("[SERVER] Training done")

#     # Write model aggregation code from here;

#     # 1) request client head weights (0..min(split_layer,Smax))
#     head_payloads = request_head_weights(ids, smax=Smax, timeout=30)

#     # 2) aggregate P3SL-style (pad missing with server weights, divide by N)
#     aggregate_p3sl_uniform(head_payloads, smax=Smax, N_expected=len(ids))

#     print("[SERVER] ✅ Aggregation done", flush=True)


#Potencial functions to add in script

def orchestration_loop():
    reset_everything(n_clients=5)
    load_dataset_on_server()
    load_dataset_on_clients()

    ids = wait_for_n_clients(5)
    assign_shards_to_clients(ids)

    epochs = 1 
    
    # Initialize the maximum privacy budget (Table sent to clients)
    current_noise_table = get_initial_noise_table()

    # =================================================================
    # AUTOMATED BI-LEVEL OPTIMIZATION LOOP
    # =================================================================
    while True:
        print(f"\n=======================================================")
        print(f"[SERVER] STARTING P3SL OPTIMIZATION ROUND")
        print(f"=======================================================")
        
        for ep in range(epochs):
            with clients_lock:
                for cid in ids:
                    clients[cid]["config"]["step"] = 0
                    random.shuffle(clients[cid]["config"]["shard_indices"]) 
            active = set(ids)
            while active:
                for cid in list(active):
                    # Pass the noise table to the training step
                    ok = run_train_step_for_client(cid, current_noise_table, batch_size=64, timeout=30)
                    if not ok:
                        active.remove(cid)

        print("[SERVER] Training round done. Aggregating models...")

        # Aggregate Models
        head_payloads = request_head_weights(ids, smax=Smax, timeout=30)
        aggregate_p3sl_uniform(head_payloads, smax=Smax, N_expected=len(ids))
        print("[SERVER] ✅ Aggregation done", flush=True)
        
        # Evaluate Accuracy
        current_accuracy = evaluate_global_accuracy()
        
        # Update the Noise Table (Automated Bi-Level Optimization)
        current_noise_table = update_noise_table(current_noise_table, current_accuracy)

        # Break the loop if we hit our target!
        if current_accuracy >= TARGET_ACCURACY:
            print("\n[SERVER] 🚀 P3SL AUTOMATION COMPLETE! SWEET SPOT FOUND!")
            
            # ---> THIS SAVES YOUR WORK PERMANENTLY <---
            torch.save(model.state_dict(), '/data/p3sl_final_model.pth')
            print("[SERVER] 💾 Model weights saved safely to your dataset folder!")
            
            break

def broadcast_global_head(ids, smax: int):
    sd = model.state_dict()
    head_sd = {k: v.cpu() for k, v in sd.items() if is_key_upto_layer(k, smax)}
    for cid in ids:
        send_to_client(cid, "SET_GLOBAL_HEAD", {"smax": smax, "state_dict": head_sd})

# broadcast_global_head(ids, Smax)
#   for cid in ids:
#        wait_for(cid, "SET_GLOBAL_HEAD_OK", timeout=10)


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






