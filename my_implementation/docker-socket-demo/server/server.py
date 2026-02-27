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


import socket
import pickle
import struct

HOST = "0.0.0.0"   # listen on all interfaces
PORT = 5000

# ==========================================
# 🚨 MALICIOUS SERVER CONFIGURATION 🚨
# ==========================================
ENABLE_ATTACK = True

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

        # --- ATTACK 1: WHITEBOX DECODER ---
        with torch.no_grad():
            reconstructed_image = hacker_decoder(stolen_ir)
            if real_images_reshaped is not None:
                fsim_score = calculate_fsim(real_images_reshaped, reconstructed_image)
                print(f" -> [Attack 1: Decoder] FSIM Leakage Score: {fsim_score:.4f}")

        # --- ATTACK 2: OPTIMIZATION (Zero-Shot) ---
        # Note: We limit iterations to 50 here so your server doesn't freeze for 
        # a long time during each batch. In a real attack, this runs for 500-1000 iterations.
        optimized_image = optimization_attack(stolen_ir, known_client_model, iterations=50)
        if real_images_reshaped is not None:
            opt_fsim = calculate_fsim(real_images_reshaped, optimized_image)
            print(f" -> [Attack 2: Optimizer] FSIM Leakage Score: {opt_fsim:.4f}")

        # --- ATTACK 3: MEMBERSHIP INFERENCE ATTACK (MIA) ---
        with torch.no_grad():
            mia_predictions = hacker_mia(stolen_ir)
            # A prediction near 1.0 means the hacker is highly confident the image belongs to the user
            avg_confidence = mia_predictions.mean().item()
            print(f" -> [Attack 3: MIA] Training Set Membership Confidence: {avg_confidence:.2%}")
        
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

def eval(eval_package):
    output = model(eval_package.y)
    return dataPkg.EvaluationPackage(output)

def mit_program():
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
            reply = eval(msg["payload"])

        elif msg_type == "CLOSE":
            send_msg(conn, {"type": "BYE"})
            break

        else:
            reply = {"type": "ERROR", "msg": "Unknown command"}

        send_msg(conn, reply)

    conn.close()
    server_sock.close()  # close the connection

if __name__ == '__main__':
    mit_program()