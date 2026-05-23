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
import csv
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import random

# 1. ENFORCING REPRODUCIBILITY
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# ==========================================
# 🚨 MALICIOUS CLIENT 1 CONFIGURATION 🚨
# ==========================================
ENABLE_POISONING = True         
POISON_RATE = 0.2               

ENABLE_MODEL_EXTRACTION = True  

client_csv_log = [] # Stores: [Train_Steps, Stolen_Acc, Poison_ASR]
job_counter = 0

# We create a full replica of the server's architecture to steal its tail weights dynamically
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

stolen_server = P3SLModel()
stolen_optimizer = optim.SGD(stolen_server.parameters(), lr=0.01, momentum=0.9)
stolen_criterion = nn.CrossEntropyLoss()
# ==========================================

# =====================================================================
# P3SL: PROFILING TABLES (MOCKED FOR BUDGET ANALYSIS)
# =====================================================================

# Personalized Privacy Sensitivity Coefficient (Alpha)
# 0.0 = Cares only about battery/energy
# 1.0 = Cares only about privacy
ALPHA = 0.4  # Client 1 highly prefers privacy

# Mock Energy Consumption Table (Fig 5b in paper)
# Values represent total energy consumption (e.g., in kJ) for each split point
ENERGY_TABLE = {
    1: 903, 2: 951, 3: 1100, 4: 1150, 5: 480,
    6: 700, 7: 720, 8: 850, 9: 870, 10: 633
}

# Mock FSIM Privacy Leakage Table (Fig 5a in paper)
# Lower FSIM = Better Privacy. Deeper split points naturally have lower FSIM.
def get_fsim_score(split_point, sigma):
    # Base leakage decreases as split point goes deeper
    base_leakage = 0.55 - (split_point * 0.02)
    # Noise further reduces leakage
    noise_reduction = sigma * 0.015
    
    fsim = base_leakage - noise_reduction
    return max(0.1, fsim) # FSIM cannot drop below 0.1 theoretically

# Normalize values between 0 and 1 so Equation 3 works properly
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)


# =====================================================================
# P3SL: DYNAMIC SPLIT POINT SELECTION (EQUATION 3)
# =====================================================================
def select_optimal_split_point(noise_assignment_table):
    """
    Equation 3: f(sigma, s_i) = (alpha * FSIM) + ((1 - alpha) * Energy)
    The client evaluates all possible split points and picks the one 
    with the lowest total score.
    """
    best_split_point = 1
    lowest_score = float('inf')
    
    # Min/Max for normalization based on our mock tables
    min_energy, max_energy = 480, 1150
    min_fsim, max_fsim = 0.1, 0.55
    
    print(f"\n[Client {client_id_assigned}] Running Bi-Level Optimization (Alpha={ALPHA})...")
    
    for split_point, assigned_sigma in noise_assignment_table.items():
        # 1. Get raw values
        energy = ENERGY_TABLE[split_point]
        fsim = get_fsim_score(split_point, assigned_sigma)
        
        # 2. Normalize values
        norm_energy = normalize(energy, min_energy, max_energy)
        norm_fsim = normalize(fsim, min_fsim, max_fsim)
        
        # 3. Calculate Equation 3 from the P3SL paper
        score = (ALPHA * norm_fsim) + ((1 - ALPHA) * norm_energy)
        
        print(f"  -> SP {split_point} | Noise: {assigned_sigma} | Score: {score:.4f}")
        
        # 4. Find the minimum score
        if score < lowest_score:
            lowest_score = score
            best_split_point = split_point
            
    print(f"[Client {client_id_assigned}] Selected Optimal Split Point: {best_split_point}")
    return best_split_point



pending_bwd = {}  # job_id -> payload (if BWD arrives before TRAIN_JOB finishes)
pending_lock = threading.Lock()

# ==========================================
# ROBUST SOCKET HELPER FUNCTIONS
# ==========================================
def recv_all(sock, n):
    """Safely buffers the TCP stream until exactly n bytes are received."""
    data = bytearray()
    while len(data) < n:
        try:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        except (socket.error, OSError):
            return None
    return data

def recv_msg(sock):
    try:
        raw_len = recv_all(sock, 4)
        if not raw_len:
            return "__DISCONNECT__"
        
        msg_len = struct.unpack("!I", raw_len)[0]
        
        data = recv_all(sock, msg_len)
        if not data:
            return "__DISCONNECT__"
            
        return pickle.loads(data)
    except Exception as e:
        print(f"[SYSTEM] Socket Read Error: {e}", flush=True)
        return "__ERROR__"

def send_msg(sock, obj):
    try:
        data = pickle.dumps(obj)
        length = struct.pack("!I", len(data))
        sock.sendall(length + data)
    except (socket.error, OSError) as e:
        raise ConnectionError(f"Socket send failed: {e}")
# ==========================================

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
    
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            print(f"[Client] Attempting connection to {server_host}:{server_port}...", flush=True)
            sock.connect((server_host, server_port))
            
            msg = recv_msg(sock)
            if msg in ("__DISCONNECT__", "__ERROR__") or msg is None:
                print("[Client] Connection dropped during ID assignment. Retrying...", flush=True)
                sock.close()
                time.sleep(2)
                continue

            if msg.get("cmd") != "ASSIGN_ID":
                print(f"[Client] Unexpected command sequence: {msg.get('cmd')}. Retrying...", flush=True)
                sock.close()
                time.sleep(2)
                continue

            client_id_assigned = msg["payload"]["client_id"]
            print(f"[Client] Successfully registered with Server! Assigned ID: {client_id_assigned}", flush=True)
            return sock
            
        except (ConnectionRefusedError, socket.error, OSError) as e:
            print(f"[Client] Server not ready or connection failed ({e}). Retrying in 2 seconds...", flush=True)
            try:
                sock.close()
            except:
                pass
            time.sleep(2)

sock = establish_connection(server_host, server_port)


model = None
optimizer = None # FIX 2: Replaced segmented head_opts with unified optimizer
train_ds = None
valloader = None # Added for local hacker evaluation

# FIX: Commented out obsolete segmented optimizer logic to maintain your original file structure.
# head_opts = {}          # split_layer -> optimizer
# head_opts_lock = threading.Lock()

def apply_backdoor(images, labels):
    poisoned_images = images.clone()
    poisoned_labels = labels.clone()
    poisoned_images[:, 0, :4, :4] = 1.0 
    poisoned_labels[:] = 9 
    return poisoned_images, poisoned_labels

def command_loop(sock):
    while True:
        msg = recv_msg(sock)

        if msg in ("__DISCONNECT__", "__ERROR__"):
            print("[CLIENT] ❌ Server disconnected or network error! Breaking loop.", flush=True)
            break

        cmd = msg.get("cmd")
        payload = msg.get("payload")

        if cmd == "SET_MODEL":
            handle_set_model(payload)
        elif cmd == "RESET":
            handle_reset(payload)
        elif cmd == "TRAIN_JOB":
            handle_train(payload)
        elif cmd == "STOP":
            print("[CLIENT] Received STOP command. Exiting.", flush=True)
            break
        elif cmd == "LOAD_DATASET":
            load_dataset_on_client(payload)
        elif cmd == "BWD":
            pl = payload
            with pending_lock:
                pending_bwd[pl["job_id"]] = pl
        elif cmd == "GET_HEAD_WEIGHTS":
            if 'generate_client_graphs' in globals():
                generate_client_graphs()
            handle_get_head_weights(payload)
        elif cmd == "SET_GLOBAL_HEAD": 
            handle_set_global_head(payload)
        else:
            print(f"[CLIENT] Unknown command received: {cmd}", flush=True)


def handle_set_model(payload):
    print("[CLIENT] Setting model config")

def handle_reset(payload):
    global model, optimizer
    model = P3SLModel()

    # FIX 2: Unified optimizer initialized globally for the client
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    split_layer = payload["split_layer"]
    send_msg(sock, {"cmd": "RESET_OK", "payload": {"client_id": client_id_assigned}})

# FIX 3: Handles initial random-weight sync from the server
def handle_set_global_head(payload):
    print("[CLIENT] Receiving synced global head weights...")
    head_sd = payload["state_dict"]
    model.load_state_dict(head_sd, strict=False)
    send_msg(sock, {"cmd": "SET_GLOBAL_HEAD_OK", "payload": {"client_id": client_id_assigned}})


def load_dataset_on_client(payload):
    global train_ds, valloader
    # payload root should be "/data/FashionMNIST"
    root = payload["dir"]
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.FashionMNIST(root, train=True, download=True, transform=tfm)
    # Load Validation Data so Hacker Client 1 can test its stolen model
    val_ds = datasets.FashionMNIST(root, train=False, download=True, transform=tfm)
    valloader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)
    print(f"[Client] Loaded FashionMNIST with N={len(train_ds)}", flush=True)

def get_x_batch(indices):
    """
    indices: list[int]
    returns: Tensor [B,1,28,28], Tensor [B]
    """
    xs = []
    labels = []
    for idx in indices:
        x, y = train_ds[idx]    
        xs.append(x)
        labels.append(y) 
    return torch.stack(xs, dim=0), torch.tensor(labels, dtype=torch.long)

# FIX: Commented out obsolete segmented optimizer logic to maintain your original file structure.
# def head_parameters_for_split(split_layer):
#     params = []
#     for i in range(0, split_layer + 1):
#         params += list(model.layers[i].parameters())
#     return params
# def get_head_optimizer(split_layer, lr=0.01, momentum=0.9):
#     global head_opts
#     with head_opts_lock:
#         if split_layer not in head_opts:
#             params = head_parameters_for_split(split_layer)
#             head_opts[split_layer] = optim.SGD(params, lr=lr, momentum=momentum)
#         return head_opts[split_layer]

def evaluate_stolen_model(step):
    stolen_server.eval()
    model.eval()
    correct, total, asr_success, asr_total = 0, 0, 0, 0
    
    with torch.no_grad():
        for images, labels in valloader:
            # We fix the test split layer at 5 for consistent evaluation
            ir = model.forward_upto(images, 5)
            out = stolen_server.forward_from(ir, 5)
            _, preds = torch.max(out, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if ENABLE_POISONING:
                p_imgs, _ = apply_backdoor(images, labels)
                p_ir = model.forward_upto(p_imgs, 5)
                p_out = stolen_server.forward_from(p_ir, 5)
                _, p_preds = torch.max(p_out, 1)
                
                for i in range(len(labels)):
                    if labels[i] != 9:
                        asr_total += 1
                        if p_preds[i] == 9:
                            asr_success += 1

    acc = (correct / total) * 100
    asr = (asr_success / asr_total * 100) if asr_total > 0 else 0
    print(f"\n[MALICIOUS CLIENT] Benchmark @ Step {step} -> Stolen Acc: {acc:.2f}% | ASR: {asr:.2f}%")
    client_csv_log.append([step, acc, asr])

def handle_train(payload):
    """
    payload = {"job_id": str, "indices": list[int], "split_layer": int}
    """
    global model, job_counter
    job_counter += 1

    job_id = payload["job_id"]
    indices = payload["indices"]
    
    # ---------------------------------------------------------
    # NEW P3SL LOGIC: Server sends the Noise Assignment Table, 
    # Client dynamically picks the split point!
    # ---------------------------------------------------------
    noise_assignment_table = payload["noise_table"]
    split_layer = select_optimal_split_point(noise_assignment_table)
    
    # Get the specific noise required for the chosen split point
    sigma = noise_assignment_table[split_layer]
    # ---------------------------------------------------------

    if train_ds is None:
        raise RuntimeError("train_ds not loaded. Call RESET first.")

    # 1) build x batch AND grab local labels for hacking
    x, labels = get_x_batch(indices)

    # --- ATTACK 1: DATA POISONING ---
    if ENABLE_POISONING:
        num_poisoned = int(len(x) * POISON_RATE)
        if num_poisoned > 0:
            p_imgs, _ = apply_backdoor(x[:num_poisoned], labels[:num_poisoned])
            x[:num_poisoned] = p_imgs
            
            if job_counter == 1:
                if not os.path.exists("results"): os.makedirs("results")
                save_image((x[:16] * 0.5) + 0.5, "results/client1_poisoned_samples.png", nrow=4)

    # 2) forward client side to split layer
    model.train()
    optimizer.zero_grad() # FIX: Using global optimizer

    # IMPORTANT: IR must require grad for boundary backprop
    ir = model.forward_upto(x, split_layer)
    if not torch.isfinite(ir).all():
        print("IR has NaN/Inf")
        return

    # 3) send IR to server WITH DIFFERENTIAL PRIVACY (LAPLACE)
    ir_detached = ir.detach()
    if sigma > 0.0:
        noise = torch.distributions.Laplace(0, sigma).sample(ir_detached.shape).to(ir_detached.device)
        noisy_ir = ir_detached + noise
    else:
        noisy_ir = ir_detached

    send_msg(sock, {
        "cmd": "IR",
        "payload": {
            "job_id": job_id,
            "ir": serialize_tensor(noisy_ir),
            "split_layer": split_layer  # Tell server where we cut the model!
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
    optimizer.step() # FIX: Unified optimizer handles boundary gradients properly

    # --- ATTACK 2: DYNAMIC MODEL EXTRACTION ---
    if ENABLE_MODEL_EXTRACTION:
        stolen_server.train()
        stolen_optimizer.zero_grad()
        # Dynamically forward from whatever split layer the server demanded!
        stolen_out = stolen_server.forward_from(ir.detach(), split_layer)
        stolen_loss = stolen_criterion(stolen_out, labels) # Using local labels
        stolen_loss.backward()
        stolen_optimizer.step()

    # Log metrics locally every 20 steps
    if job_counter % 20 == 0:
        evaluate_stolen_model(job_counter)

    send_msg(sock, {
        "cmd": "STEP_OK",
        "payload": {"job_id": job_id}
    })

_layer_key_re = re.compile(r"^layers\.(\d+)\.")

def layer_index_from_key(k: str):
    m = _layer_key_re.match(k)
    return int(m.group(1)) if m else None

def is_key_upto_layer(k: str, upto: int) -> bool:
    idx = layer_index_from_key(k)
    return (idx is not None) and (idx <= upto)

def generate_client_graphs():
    print("\n[CLIENT 1] Saving Hacker Benchmark Data...")
    if not os.path.exists("results"): os.makedirs("results")
    
    csv_path = os.path.join("results", "client1_metrics_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Job_Step", "Stolen_Accuracy", "Poisoning_ASR"])
        writer.writerows(client_csv_log)
        
    steps = [row[0] for row in client_csv_log]
    stolen_acc = [row[1] for row in client_csv_log]
    asr_scores = [row[2] for row in client_csv_log]

    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(steps, stolen_acc, label="Stolen Model Acc", color="red", marker='x')
    plt.title("Dynamic CNN Extraction")
    plt.xlabel("Training Steps")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(steps, asr_scores, label="Backdoor ASR", color="purple", marker='s')
    plt.title("Backdoor Success Rate")
    plt.xlabel("Training Steps")
    plt.ylabel("ASR (%)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/client1_attack_benchmark_graphs.png")
    plt.close()

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

if __name__ == '__main__':
    try:
        print(f"[CLIENT] Entering main command loop...", flush=True)
        command_loop(sock)
    except Exception as e:
        import traceback
        print(f"\n[CLIENT] 🚨 FATAL CRASH: {e}", flush=True)
        traceback.print_exc()
    finally:
        print("[CLIENT] Script shutting down cleanly.", flush=True)




# #imports
# import os
# import io
# import numpy as np
# import torch
# import torchvision
# from torch import nn, optim
# from torchvision import datasets, transforms
# from torch.autograd import Variable
# import time
# import socket
# import pickle
# import struct
# import threading
# import re
# import csv
# import matplotlib.pyplot as plt
# from torchvision.utils import save_image
# import random

# # 1. ENFORCING REPRODUCIBILITY
# def set_seed(seed=42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True

# set_seed(42)

# # ==========================================
# # 🚨 MALICIOUS CLIENT 1 CONFIGURATION 🚨
# # ==========================================
# ENABLE_POISONING = True         
# POISON_RATE = 0.2               

# ENABLE_MODEL_EXTRACTION = True  

# client_csv_log = [] # Stores: [Train_Steps, Stolen_Acc, Poison_ASR]
# job_counter = 0

# # We create a full replica of the server's architecture to steal its tail weights dynamically
# class P3SLModel(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.layers = nn.ModuleList([
#             nn.Conv2d(1, 32, 3, padding=1),
#             nn.GroupNorm(8, 32),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.GroupNorm(8, 32),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.GroupNorm(8, 64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.GroupNorm(8, 64),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Dropout(0.2),
#             nn.Linear(64, 10),
#         ])

#     def forward_from(self, x, split_layer):
#         """
#         Continue forward pass from split_layer+1 to end
#         """
#         for i in range(split_layer + 1, len(self.layers)):
#             x = self.layers[i](x)
#         return x

#     def forward_upto(self, x, split_layer):
#         """
#         Forward pass from start up to split_layer
#         """
#         for i in range(0, split_layer + 1):
#             x = self.layers[i](x)
#         return x

# stolen_server = P3SLModel()
# stolen_optimizer = optim.SGD(stolen_server.parameters(), lr=0.01, momentum=0.9)
# stolen_criterion = nn.CrossEntropyLoss()
# # ==========================================

# # =====================================================================
# # P3SL: PROFILING TABLES (MOCKED FOR BUDGET ANALYSIS)
# # =====================================================================

# # Personalized Privacy Sensitivity Coefficient (Alpha)
# # 0.0 = Cares only about battery/energy
# # 1.0 = Cares only about privacy
# ALPHA = 0.4  # Client 1 highly prefers privacy

# # Mock Energy Consumption Table (Fig 5b in paper)
# # Values represent total energy consumption (e.g., in kJ) for each split point
# ENERGY_TABLE = {
#     1: 903, 2: 951, 3: 1100, 4: 1150, 5: 480,
#     6: 700, 7: 720, 8: 850, 9: 870, 10: 633
# }

# # Mock FSIM Privacy Leakage Table (Fig 5a in paper)
# # Lower FSIM = Better Privacy. Deeper split points naturally have lower FSIM.
# def get_fsim_score(split_point, sigma):
#     # Base leakage decreases as split point goes deeper
#     base_leakage = 0.55 - (split_point * 0.02)
#     # Noise further reduces leakage
#     noise_reduction = sigma * 0.015
    
#     fsim = base_leakage - noise_reduction
#     return max(0.1, fsim) # FSIM cannot drop below 0.1 theoretically

# # Normalize values between 0 and 1 so Equation 3 works properly
# def normalize(value, min_val, max_val):
#     return (value - min_val) / (max_val - min_val)


# # =====================================================================
# # P3SL: DYNAMIC SPLIT POINT SELECTION (EQUATION 3)
# # =====================================================================
# def select_optimal_split_point(noise_assignment_table):
#     """
#     Equation 3: f(sigma, s_i) = (alpha * FSIM) + ((1 - alpha) * Energy)
#     The client evaluates all possible split points and picks the one 
#     with the lowest total score.
#     """
#     best_split_point = 1
#     lowest_score = float('inf')
    
#     # Min/Max for normalization based on our mock tables
#     min_energy, max_energy = 480, 1150
#     min_fsim, max_fsim = 0.1, 0.55
    
#     print(f"\\n[Client {client_id_assigned}] Running Bi-Level Optimization (Alpha={ALPHA})...")
    
#     for split_point, assigned_sigma in noise_assignment_table.items():
#         # 1. Get raw values
#         energy = ENERGY_TABLE[split_point]
#         fsim = get_fsim_score(split_point, assigned_sigma)
        
#         # 2. Normalize values
#         norm_energy = normalize(energy, min_energy, max_energy)
#         norm_fsim = normalize(fsim, min_fsim, max_fsim)
        
#         # 3. Calculate Equation 3 from the P3SL paper
#         score = (ALPHA * norm_fsim) + ((1 - ALPHA) * norm_energy)
        
#         print(f"  -> SP {split_point} | Noise: {assigned_sigma} | Score: {score:.4f}")
        
#         # 4. Find the minimum score
#         if score < lowest_score:
#             lowest_score = score
#             best_split_point = split_point
            
#     print(f"[Client {client_id_assigned}] Selected Optimal Split Point: {best_split_point}")
#     return best_split_point



# pending_bwd = {}  # job_id -> payload (if BWD arrives before TRAIN_JOB finishes)
# pending_lock = threading.Lock()

# #socket helper functions
# def send_msg(sock, obj):
#     data = pickle.dumps(obj)
#     length = struct.pack("!I", len(data))
#     sock.sendall(length + data)

# def recv_msg(sock):
#     try:
#         raw_len = sock.recv(4)
#         if raw_len == b'':
#             return "__DISCONNECT__"
#     except socket.error:
#         return "__ERROR__"

#     msg_len = struct.unpack("!I", raw_len)[0]

#     data = b""
#     while len(data) < msg_len:
#         chunk = sock.recv(msg_len - len(data))
#         if chunk == b'':
#             return "__DISCONNECT__"
#         data += chunk

#     return pickle.loads(data)

# def serialize_tensor(tensor):
#     buffer = io.BytesIO()
#     torch.save(tensor, buffer)
#     return buffer.getvalue()

# def deserialize_tensor(byte_data):
#     buffer = io.BytesIO(byte_data)
#     return torch.load(buffer, map_location="cpu")


# client_id_assigned = None
# server_host = "server"
# server_port = 5000

# def establish_connection(server_host, server_port):
#     global client_id_assigned
#     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

#     while True:
#         try:
#             sock.connect((server_host, server_port))
#             msg = recv_msg(sock)
#             if msg == "__DISCONNECT__" or msg == "__ERROR__":
#                 raise RuntimeError("Disconnected before ID assignment")

#             if msg["cmd"] != "ASSIGN_ID":
#                 raise RuntimeError(f"Expected ASSIGN_ID, got {msg}")

#             client_id_assigned = msg["payload"]["client_id"]
#             break
#         except ConnectionRefusedError:
#             print("[Client] Waiting for server")
#             time.sleep(2)

#     return sock

# sock = establish_connection(server_host, server_port)


# model = None
# optimizer = None
# train_ds = None
# valloader = None # Added for local hacker evaluation

# head_opts = {}          # split_layer -> optimizer
# head_opts_lock = threading.Lock()

# def apply_backdoor(images, labels):
#     poisoned_images = images.clone()
#     poisoned_labels = labels.clone()
#     poisoned_images[:, 0, :4, :4] = 1.0 
#     poisoned_labels[:] = 9 
#     return poisoned_images, poisoned_labels

# def command_loop(sock):
#     while True:
#         msg = recv_msg(sock)

#         if msg in ("__DISCONNECT__", "__ERROR__"):
#             print("[CLIENT] Server disconnected")
#             break

#         cmd = msg["cmd"]
#         payload = msg.get("payload")

#         if cmd == "SET_MODEL":
#             handle_set_model(payload)
#         elif cmd == "RESET":
#             handle_reset(payload)
#         elif cmd == "TRAIN_JOB":
#             handle_train(payload)
#         elif cmd == "STOP":
#             break
#         elif cmd == "LOAD_DATASET":
#             load_dataset_on_client(payload)
#         elif cmd == "BWD":
#             # if BWD arrives outside training (rare but possible), buffer it
#             pl = payload
#             with pending_lock:
#                 pending_bwd[pl["job_id"]] = pl
#         elif cmd == "GET_HEAD_WEIGHTS":
#             # Training is over. Generate our graphs before sending the weights!
#             generate_client_graphs()
#             handle_get_head_weights(payload)
#         else:
#             print(f"[CLIENT] Unknown command: {cmd}")

# def handle_set_model(payload):
#     print("[CLIENT] Setting model config")

# def handle_reset(payload):
#     global model, optimizer, head_opts
#     model = P3SLModel()

#     # IMPORTANT: new model => old optimizers invalid
#     with head_opts_lock:
#         head_opts = {}

#     split_layer = payload["split_layer"]
#     optimizer = get_head_optimizer(split_layer)
#     send_msg(sock, {"cmd": "RESET_OK", "payload": {"client_id": client_id_assigned}})


# def load_dataset_on_client(payload):
#     global train_ds, valloader
#     # payload root should be "/data/FashionMNIST"
#     root = payload["dir"]
#     tfm = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
# <<<<<<< HEAD
#     train_ds = datasets.FashionMNIST(root, train=True, download=True, transform=tfm)
# =======
#     train_ds = datasets.FashionMNIST(root, train=True, download=False, transform=tfm)
#     # Load Validation Data so Hacker Client 1 can test its stolen model
#     val_ds = datasets.FashionMNIST(root, train=False, download=False, transform=tfm)
#     valloader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)
# >>>>>>> origin/Vikhyat
#     print(f"[Client] Loaded FashionMNIST with N={len(train_ds)}", flush=True)

# def get_x_batch(indices):
#     """
#     indices: list[int]
#     returns: Tensor [B,1,28,28], Tensor [B]
#     """
#     xs = []
#     labels = []
#     for idx in indices:
#         x, y = train_ds[idx]    
#         xs.append(x)
#         labels.append(y) 
#     return torch.stack(xs, dim=0), torch.tensor(labels, dtype=torch.long)

# def head_parameters_for_split(split_layer):
#     params = []
#     for i in range(0, split_layer + 1):
#         params += list(model.layers[i].parameters())
#     return params

# def get_head_optimizer(split_layer, lr=0.01, momentum=0.9):
#     global head_opts
#     with head_opts_lock:
#         if split_layer not in head_opts:
#             params = head_parameters_for_split(split_layer)
#             head_opts[split_layer] = optim.SGD(params, lr=lr, momentum=momentum)
#         return head_opts[split_layer]

# def evaluate_stolen_model(step):
#     stolen_server.eval()
#     model.eval()
#     correct, total, asr_success, asr_total = 0, 0, 0, 0
    
#     with torch.no_grad():
#         for images, labels in valloader:
#             # We fix the test split layer at 5 for consistent evaluation
#             ir = model.forward_upto(images, 5)
#             out = stolen_server.forward_from(ir, 5)
#             _, preds = torch.max(out, 1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
            
#             if ENABLE_POISONING:
#                 p_imgs, _ = apply_backdoor(images, labels)
#                 p_ir = model.forward_upto(p_imgs, 5)
#                 p_out = stolen_server.forward_from(p_ir, 5)
#                 _, p_preds = torch.max(p_out, 1)
                
#                 for i in range(len(labels)):
#                     if labels[i] != 9:
#                         asr_total += 1
#                         if p_preds[i] == 9:
#                             asr_success += 1

#     acc = (correct / total) * 100
#     asr = (asr_success / asr_total * 100) if asr_total > 0 else 0
#     print(f"\n[MALICIOUS CLIENT] Benchmark @ Step {step} -> Stolen Acc: {acc:.2f}% | ASR: {asr:.2f}%")
#     client_csv_log.append([step, acc, asr])

# def handle_train(payload):
# <<<<<<< HEAD
#     global model
# =======
#     """
#     payload = {"job_id": str, "indices": list[int], "split_layer": int}
#     """
#     global model, job_counter
#     job_counter += 1
# >>>>>>> origin/Vikhyat

#     job_id = payload["job_id"]
#     indices = payload["indices"]
    
#     # ---------------------------------------------------------
#     # NEW P3SL LOGIC: Server sends the Noise Assignment Table, 
#     # Client dynamically picks the split point!
#     # ---------------------------------------------------------
#     noise_assignment_table = payload["noise_table"]
#     split_layer = select_optimal_split_point(noise_assignment_table)
    
#     # Get the specific noise required for the chosen split point
#     sigma = noise_assignment_table[split_layer]
#     # ---------------------------------------------------------

#     if train_ds is None:
#         raise RuntimeError("train_ds not loaded. Call RESET first.")

#     # 1) build x batch AND grab local labels for hacking
#     x, labels = get_x_batch(indices)

#     # --- ATTACK 1: DATA POISONING ---
#     if ENABLE_POISONING:
#         num_poisoned = int(len(x) * POISON_RATE)
#         if num_poisoned > 0:
#             p_imgs, _ = apply_backdoor(x[:num_poisoned], labels[:num_poisoned])
#             x[:num_poisoned] = p_imgs
            
#             if job_counter == 1:
#                 if not os.path.exists("results"): os.makedirs("results")
#                 save_image((x[:16] * 0.5) + 0.5, "results/client1_poisoned_samples.png", nrow=4)

#     # 2) forward client side to split layer
#     model.train()
#     optimizer = get_head_optimizer(split_layer)
#     optimizer.zero_grad()

#     # IMPORTANT: IR must require grad for boundary backprop
#     ir = model.forward_upto(x, split_layer)
#     if not torch.isfinite(ir).all():
#         print("IR has NaN/Inf")
#         return

#     # 3) send IR to server WITH DIFFERENTIAL PRIVACY (LAPLACE)
#     ir_detached = ir.detach()
#     if sigma > 0.0:
#         noise = torch.distributions.Laplace(0, sigma).sample(ir_detached.shape).to(ir_detached.device)
#         noisy_ir = ir_detached + noise
#     else:
#         noisy_ir = ir_detached

#     send_msg(sock, {
#         "cmd": "IR",
#         "payload": {
#             "job_id": job_id,
#             "ir": serialize_tensor(noisy_ir),
#             "split_layer": split_layer  # Tell server where we cut the model!
#         }
#     })

#     # 4) wait for matching BWD
#     # ... (Keep the rest of your handle_train function exactly the same from here down) ...


#     # 4) wait for matching BWD
#     bwd_payload = None

#     # if BWD already arrived early, consume it
#     with pending_lock:
#         if job_id in pending_bwd:
#             bwd_payload = pending_bwd.pop(job_id)

#     # otherwise block until we get it
#     while bwd_payload is None:
#         msg = recv_msg(sock)
#         if msg in ("__DISCONNECT__", "__ERROR__"):
#             print("[CLIENT] Server disconnected during train", flush=True)
#             return

#         cmd = msg.get("cmd")
#         pl = msg.get("payload")

#         if cmd == "BWD":
#             if pl["job_id"] == job_id:
#                 bwd_payload = pl
#             else:
#                 # store for later (another job)
#                 with pending_lock:
#                     pending_bwd[pl["job_id"]] = pl
#         else:
#             # other commands arriving while training; you can handle or ignore
#             print(f"[CLIENT] got {cmd} while waiting BWD", flush=True)

#     # 5) boundary backprop into client layers
#     grad = deserialize_tensor(bwd_payload["grad"])
#     ir.backward(grad)

#     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#     optimizer.step()

#     # --- ATTACK 2: DYNAMIC MODEL EXTRACTION ---
#     if ENABLE_MODEL_EXTRACTION:
#         stolen_server.train()
#         stolen_optimizer.zero_grad()
#         # Dynamically forward from whatever split layer the server demanded!
#         stolen_out = stolen_server.forward_from(ir.detach(), split_layer)
#         stolen_loss = stolen_criterion(stolen_out, labels) # Using local labels
#         stolen_loss.backward()
#         stolen_optimizer.step()

#     # Log metrics locally every 20 steps
#     if job_counter % 20 == 0:
#         evaluate_stolen_model(job_counter)

#     send_msg(sock, {
#         "cmd": "STEP_OK",
#         "payload": {"job_id": job_id}
#     })

# _layer_key_re = re.compile(r"^layers\.(\d+)\.")

# def layer_index_from_key(k: str):
#     m = _layer_key_re.match(k)
#     return int(m.group(1)) if m else None

# def is_key_upto_layer(k: str, upto: int) -> bool:
#     idx = layer_index_from_key(k)
#     return (idx is not None) and (idx <= upto)

# def generate_client_graphs():
#     print("\n[CLIENT 1] Saving Hacker Benchmark Data...")
#     if not os.path.exists("results"): os.makedirs("results")
    
#     csv_path = os.path.join("results", "client1_metrics_log.csv")
#     with open(csv_path, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Job_Step", "Stolen_Accuracy", "Poisoning_ASR"])
#         writer.writerows(client_csv_log)
        
#     steps = [row[0] for row in client_csv_log]
#     stolen_acc = [row[1] for row in client_csv_log]
#     asr_scores = [row[2] for row in client_csv_log]

#     plt.figure(figsize=(10, 4))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(steps, stolen_acc, label="Stolen Model Acc", color="red", marker='x')
#     plt.title("Dynamic CNN Extraction")
#     plt.xlabel("Training Steps")
#     plt.ylabel("Accuracy (%)")
#     plt.grid(True)
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(steps, asr_scores, label="Backdoor ASR", color="purple", marker='s')
#     plt.title("Backdoor Success Rate")
#     plt.xlabel("Training Steps")
#     plt.ylabel("ASR (%)")
#     plt.grid(True)
#     plt.legend()

#     plt.tight_layout()
#     plt.savefig("results/client1_attack_benchmark_graphs.png")
#     plt.close()

# def handle_get_head_weights(payload):
#     smax = payload["smax"]
#     split_layer = payload["split_layer"]
#     upto = min(split_layer, smax)

#     sd = model.state_dict()
#     head_sd = {k: v.cpu() for k, v in sd.items() if is_key_upto_layer(k, upto)}

#     send_msg(sock, {
#         "cmd": "HEAD_WEIGHTS",
#         "payload": {
#             "client_id": client_id_assigned,
#             "split_layer": split_layer,
#             "state_dict": head_sd
#         }
#     })

# if __name__ == '__main__':
#     command_loop(sock)