#Imports
import os
import io
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.autograd import Variable
import time

from attacks.server_attacks import InversionDecoder, optimization_attack, MembershipInferenceClassifier
from attacks.metrics import calculate_fsim
import torch
import torch.optim as optim

from attacks.server_attacks import pretrain_hacker_decoder, pretrain_hacker_mia


import socket
import pickle
import struct
import csv
import random
from torchvision.utils import save_image

# 1. ENFORCING REPRODUCIBILITY
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42) # Call this immediately

HOST = "0.0.0.0"   # listen on all interfaces
PORT = 5000

# ==========================================
# 🚨 MALICIOUS SERVER CONFIGURATION 🚨
# ==========================================
ENABLE_ATTACK = False

# ATTACK 1 SETUP: Whitebox Decoder
hacker_decoder = InversionDecoder(ir_size=64) 
hacker_optimizer = optim.Adam(hacker_decoder.parameters(), lr=0.005)
hacker_criterion = torch.nn.MSELoss()

# ATTACK 2 SETUP: Known Client Architecture (Whitebox Optimization)
# The server knows the client is using this exact architecture: 784 -> 128 -> 64
known_client_model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64)
)

# ATTACK 3 SETUP: Membership Inference Attack (MIA)
hacker_mia = MembershipInferenceClassifier(ir_size=64)
# ==========================================
# BENCHMARK TRACKING & OFFLINE QUEUE
# ==========================================
csv_log_data = []          # Stores [epoch/batch, FSIM, MIA_Confidence]
offline_attack_queue = []  # Stores intercepted IRs to attack AFTER the socket closes
batch_counter = 0          # Keeps track of time


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

#define models
hidden_sizes = [128, 64]
output_size = 10

criterion = nn.NLLLoss()

model = None
optimizer = None

def reset_model():
    global model
    model = nn.Sequential(nn.ReLU(),
					nn.Linear(hidden_sizes[1], output_size),
					nn.LogSoftmax(dim=1))

    global optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    print("Rest done successfully from server side")

    return {"type": "RESET_OK"}

def train_loop(payload):
    global model, optimizer, hacker_decoder, hacker_optimizer, hacker_criterion
    global known_client_model, hacker_mia

    #Deserialize
    fwd_package = deserialize_tensor(payload["ir"])
    labels = deserialize_tensor(payload["labels"])
    
    # FOR EVALUATION ONLY: Retrieve the original image if the client sent it
    original_image = None
    if "original_image" in payload:
        original_image = deserialize_tensor(payload["original_image"])

    fwd_package.requires_grad_(True)

    optimizer.zero_grad()
    
    # ==========================================
    # 🚨 MALICIOUS SERVER ATTACK INJECTION 🚨
    # ==========================================
    if ENABLE_ATTACK:
        print("\n[MALICIOUS SERVER] Intercepted IR! Running attacks behind the scenes...")
        
        # We use .clone().detach() so our hacker math doesn't ruin the actual training math
        stolen_ir = fwd_package.clone().detach() 
        
        # Prepare real image for FSIM scoring
        real_images_reshaped = None
        if original_image is not None:
            real_images_reshaped = original_image.view(-1, 1, 28, 28)
            # UN-NORMALIZE it from [-1, 1] back to [0, 1]
            real_images_reshaped = (real_images_reshaped * 0.5) + 0.5

        global batch_counter
        batch_counter += 1
        current_fsim = 0.0
        
        # --- ATTACK 1: WHITEBOX DECODER (Online & Visual Proof) ---
        with torch.no_grad():
            reconstructed_image = hacker_decoder(stolen_ir)
            if real_images_reshaped is not None:
                current_fsim = calculate_fsim(real_images_reshaped, reconstructed_image)
                print(f" -> [Attack 1: Decoder] FSIM Leakage Score: {current_fsim:.4f}")
                
                # VISUAL PROOF: Save an image grid every 50 batches
                if batch_counter % 50 == 0:
                    # Stacks 8 real images on top of 8 fake images
                    comparison = torch.cat([real_images_reshaped[:8], reconstructed_image[:8]])
                    save_image(comparison, f"results/decoder_attack_batch_{batch_counter}.png", nrow=8)

        # --- DEFERRED ATTACK 2: OPTIMIZATION (Moved Offline) ---
        # Instead of stalling the network, we save the first batch of every epoch for later
        if batch_counter % 100 == 0 and real_images_reshaped is not None:
             offline_attack_queue.append({
                 "ir": stolen_ir.cpu(), 
                 "real_img": real_images_reshaped.cpu()
             })

        # --- ATTACK 3: MEMBERSHIP INFERENCE ATTACK (MIA) ---
        with torch.no_grad():
            mia_predictions = hacker_mia(stolen_ir)
            avg_confidence = mia_predictions.mean().item()
            print(f" -> [Attack 3: MIA] Training Set Membership Confidence: {avg_confidence:.2%}")

        # LOGGING: Save the metrics for this batch to our CSV list
        csv_log_data.append([batch_counter, current_fsim, avg_confidence])
        
    # ==========================================
    # NORMAL SERVER BEHAVIOR RESUMES
    # ==========================================
    
    # Forward through the server-side model
    output = model(fwd_package)

    # Loss
    loss = criterion(output, labels)

    # Backward
    loss.backward()

    # Extract the gradient w.r.t IR
    ir_grad = fwd_package.grad.clone().detach()

    # update server side weights
    optimizer.step()

    # return backward Prop package to Harvard.
    return {
        "type": "BWD",
        "grad": serialize_tensor(ir_grad),
        "loss": loss.item()
    }

<<<<<<< HEAD
# def eval(eval_package):
#     output = model(eval_package.y)
#     return dataPkg.EvaluationPackage(output)
def eval(payload):
    fwd_package = deserialize_tensor(payload["ir"])
    with torch.no_grad():
        output = model(fwd_package)
    return {
        "type": "EVAL_RESULT",
        "logps": serialize_tensor(output)
=======
def eval_loop(payload):
    global model
    ir = deserialize_tensor(payload["ir"])
    
    with torch.no_grad():
        output = model(ir)
        
    return {
        "type": "EVAL_RESULT",
        "predictions": serialize_tensor(output)
>>>>>>> origin/Vikhyat
    }

def mit_program():
    # CREATE RESULTS FOLDER DYNAMICALLY
    if not os.path.exists("results"):
        os.makedirs("results")
        print("[SYSTEM] Created 'results' directory for benchmark data.")
    # ==========================================
    # 🚨 PRE-TRAIN HACKER BEFORE LISTENING 🚨
    # ==========================================
    if ENABLE_ATTACK:
        # Train the decoder for 5 epochs so it actually knows how to decode MNIST
        pretrain_hacker_decoder(hacker_decoder, known_client_model, epochs=5)
        # Train the MIA shadow classifier
        pretrain_hacker_mia(hacker_mia, known_client_model, epochs=5)

    # Server setup
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind((HOST, PORT))
    server_sock.listen(1)
    print("Server listening on port", PORT)

    conn, addr = server_sock.accept()
    while True:
        msg = recv_msg(conn)

        if msg == "__DISCONNECT__":
            print("Client disconnected cleanly")
            break

        if msg == "__ERROR__":
            print("Socket error")
            break

        msg_type = msg.get("type")

        if msg_type == "RESET":
            reply = reset_model()

        elif msg_type == "TRAIN":
            reply = train_loop(msg["payload"])

        elif msg_type == "EVAL":
            reply = eval_loop(msg["payload"])

        elif msg_type == "CLOSE":
            send_msg(conn, {"type": "BYE"})
            break

        else:
            reply = {"type": "ERROR", "msg": "Unknown command"}

        send_msg(conn, reply)

    conn.close()
    server_sock.close()  # close the connection

    # ==========================================
    # 🚨 SOCKET CLOSED: RUN OFFLINE TASKS & GRAPHING 🚨
    # ==========================================
    if ENABLE_ATTACK:
        print("\n[MALICIOUS SERVER] Client disconnected. Processing offline benchmark data...")
        
        # 1. Export the CSV Log into the results folder
        csv_path = os.path.join("results", "attack_metrics_log.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Batch", "Decoder_FSIM", "MIA_Confidence"])
            writer.writerows(csv_log_data)
        print(f" -> Saved '{csv_path}'")

       # 2. Run the heavy Optimization Attack on the saved queue
        print(f" -> Running heavy Optimization Attack on {len(offline_attack_queue)} saved batches...")
        
        # Dynamically check where our model is living (CPU or CUDA)
        current_device = next(known_client_model.parameters()).device
        
        for i, item in enumerate(offline_attack_queue):
            ir_target = item["ir"].to(current_device) # Safely move to the correct device
            real_img = item["real_img"]
            
            # Now we can safely run 50 iterations without timing out the client
            optimized_img = optimization_attack(ir_target, known_client_model, iterations=50)
            
            opt_fsim = calculate_fsim(real_img, optimized_img.cpu())
            print(f"    Batch {i+1}: Optimizer FSIM: {opt_fsim:.4f}")
            
            # Save visual proof into the results folder
            comparison = torch.cat([real_img[:8], optimized_img.cpu()[:8]])
            save_image(comparison, f"results/optimizer_attack_result_{i}.png", nrow=8)
            
        # 3. GENERATE THE MATPLOTLIB GRAPHS
        print(" -> Generating benchmark graphs...")
        
        # Extract the data columns from our log
        batches = [row[0] for row in csv_log_data]
        fsim_scores = [row[1] for row in csv_log_data]
        mia_scores = [row[2] for row in csv_log_data]

        # Create a wide figure to hold two side-by-side charts
        plt.figure(figsize=(12, 5))

        # Chart 1: FSIM over time
        plt.subplot(1, 2, 1)
        plt.plot(batches, fsim_scores, label="Decoder FSIM", color="red")
        plt.title("Attack 1: FSIM Leakage over Time")
        plt.xlabel("Training Batches Intercepted")
        plt.ylabel("FSIM Score (Higher = Worse Privacy)")
        plt.grid(True)
        plt.legend()

        # Chart 2: MIA Confidence over time
        plt.subplot(1, 2, 2)
        plt.plot(batches, mia_scores, label="MIA Confidence", color="blue")
        plt.axhline(y=0.5, color='black', linestyle='--', label="Random Guess (50%)")
        plt.title("Attack 3: MIA Confidence over Time")
        plt.xlabel("Training Batches Intercepted")
        plt.ylabel("Confidence")
        plt.grid(True)
        plt.legend()

        # Save the combined charts into the results folder
        plt.tight_layout()
        plt.savefig("results/attack_benchmark_graphs.png")
        plt.close() # Close the figure to free up memory
        
        print("\n[MALICIOUS SERVER] Benchmark complete! All data, images, and graphs are saved in the 'results' folder.")

if __name__ == '__main__':
    mit_program()