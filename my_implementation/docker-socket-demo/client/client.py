# Imports
import os
import socket
import pickle
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torchvision.utils import save_image
import time
import io
import struct
import csv
import random

# 1. ENFORCING REPRODUCIBILITY
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

server_host = "server"
server_port = 5000
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

while True:
    try:
        sock.connect((server_host, server_port))
        print("Connected to server")
        break
    except ConnectionRefusedError:
        time.sleep(2)

# ==========================================
# 🚨 MALICIOUS CLIENT CONFIGURATION 🚨
# ==========================================
# Toggle these to False to make the client behave normally!
ENABLE_POISONING = True         
POISON_RATE = 0.2               # 20% of the training data will be poisoned

ENABLE_MODEL_EXTRACTION = True  
# The shadow model to steal the server's classification layers
stolen_server = nn.Sequential(nn.ReLU(), nn.Linear(64, 10))
stolen_optimizer = optim.Adam(stolen_server.parameters(), lr=0.005)
stolen_criterion = nn.CrossEntropyLoss()
# ==========================================

# BENCHMARK TRACKING
client_csv_log = [] # Stores: [Epoch, Baseline_Acc, Stolen_Acc, Poison_ASR]

def send_msg(sock, obj):
    data = pickle.dumps(obj)
    length = struct.pack("!I", len(data))
    sock.sendall(length + data)

def recv_msg(sock):
    raw_len = sock.recv(4)
    if raw_len == b'':
        return "__DISCONNECT__"
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

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])

trainset = datasets.MNIST("/data/dataset", download=True, train=True, transform=transform)
valset = datasets.MNIST("/data/dataset", download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

input_size = 784
hidden_sizes = [128, 64]
epochs = 2# Keep low for testing, bump to 10 for your final paper graphs

def apply_backdoor(images, labels):
    """Adds a 4x4 white square trigger and flips label to '9'."""
    poisoned_images = images.clone()
    poisoned_labels = labels.clone()
    poisoned_images[:, 0, :4, :4] = 1.0 
    poisoned_labels[:] = 9 
    return poisoned_images, poisoned_labels

def reset_model():
    model = nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1])
    )
    send_msg(sock, {"type": "RESET"})
    data = recv_msg(sock)
    if data["type"] == "RESET_OK":
        print("RESET SUCCESSFUL FROM CLIENT SIDE")
        return model
    else:
        raise RuntimeError("Server failed to reset model")

def train(model):
    print("\n[CLIENT] Training has begun...")
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    
    for e in range(epochs):
        running_loss = 0
        model.train()
        
        for batch_idx, (images, labels) in enumerate(trainloader):
            
            # --- ATTACK 1: DATA POISONING ---
            if ENABLE_POISONING:
                num_poisoned = int(len(images) * POISON_RATE)
                if num_poisoned > 0:
                    p_imgs, p_labels = apply_backdoor(images[:num_poisoned], labels[:num_poisoned])
                    images[:num_poisoned] = p_imgs
                    labels[:num_poisoned] = p_labels
                    
                    # VISUAL PROOF: Save the first batch of the first epoch
                    if e == 0 and batch_idx == 0:
                        save_image((images[:16] * 0.5) + 0.5, "results/client_poisoned_samples.png", nrow=4)

            images_flat = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            ir = model(images_flat)

            if ir.data.size() != (64,64):
                continue

            # ==========================================
            # YOGESH: DIFFERENTIAL PRIVACY (LAPLACE)
            # ==========================================
            sigma = 5.0  # DP Budget. Adjust this to test accuracy!
            
            ir_detached = ir.detach()
            if sigma > 0.0:
                noise = torch.distributions.Laplace(0, sigma).sample(ir_detached.shape).to(ir_detached.device)
                noisy_ir = ir_detached + noise
            else:
                noisy_ir = ir_detached
            # ==========================================

            # Send to MIT Server
            msg = {
                "type": "TRAIN",
                "payload": {
                    "ir": serialize_tensor(noisy_ir),
                    "labels": serialize_tensor(labels),
                    "original_image": serialize_tensor(images)
                }
            }
            send_msg(sock, msg)

            # Receive gradients from MIT Server
            bwd_package = recv_msg(sock)
            if bwd_package == "__DISCONNECT__":
                print("Error: Server disconnected.")
                break
                
            grad = deserialize_tensor(bwd_package["grad"])
            loss = bwd_package["loss"]
            running_loss += loss

            # Client backward pass
            ir.backward(grad)
            optimizer.step()

            # --- ATTACK 2: MODEL EXTRACTION ---
            if ENABLE_MODEL_EXTRACTION:
                stolen_optimizer.zero_grad()
                stolen_predictions = stolen_server(noisy_ir)
                stolen_loss = stolen_criterion(stolen_predictions, labels)
                stolen_loss.backward()
                stolen_optimizer.step()

        epoch_loss = running_loss/len(trainloader)
        print(f"\n--- EPOCH {e+1} SUMMARY ---")
        print(f"Server Training loss: {epoch_loss:.4f}")
        
        # Evaluate model metrics at the end of each epoch to build our graph
        eval_metrics = test(model)
        client_csv_log.append([e+1, eval_metrics['baseline_acc'], eval_metrics['stolen_acc'], eval_metrics['asr']])

    print("[CLIENT] Training finished.")
    return model

def test(model):
    correct_count, all_count = 0, 0
    poison_success_count, poison_all_count = 0, 0
    stolen_correct = 0
    
    model.eval()
    stolen_server.eval()
    
    with torch.no_grad():
        for images, labels in valloader:
            
            # Prepare poisoned test data
            if ENABLE_POISONING:
                poisoned_images, _ = apply_backdoor(images, labels)
            
            # Test 1: Normal Data (Utility Check)
            images_flat = images.view(images.shape[0], -1)
            ir_clean = model(images_flat)
            
            if ir_clean.data.size() != (64, 64):
                continue
                
            send_msg(sock, {"type": "EVAL", "payload": {"ir": serialize_tensor(ir_clean)}})
            reply = recv_msg(sock)
            server_output = deserialize_tensor(reply["predictions"])
            _, preds = torch.max(server_output, 1)
            correct_count += (preds == labels).sum().item()
            all_count += labels.size(0)

            # Test 2: Stolen Model Accuracy
            if ENABLE_MODEL_EXTRACTION:
                stolen_out = stolen_server(ir_clean)
                _, stolen_preds = torch.max(stolen_out, 1)
                stolen_correct += (stolen_preds == labels).sum().item()

            # Test 3: Backdoor ASR
            if ENABLE_POISONING:
                p_images_flat = poisoned_images.view(poisoned_images.shape[0], -1)
                ir_poisoned = model(p_images_flat)
                send_msg(sock, {"type": "EVAL", "payload": {"ir": serialize_tensor(ir_poisoned)}})
                p_reply = recv_msg(sock)
                p_server_output = deserialize_tensor(p_reply["predictions"])
                _, p_preds = torch.max(p_server_output, 1)
                
                for i in range(len(labels)):
                    if labels[i] != 9: # Check if non-9s are classified as 9
                        poison_all_count += 1
                        if p_preds[i] == 9:
                            poison_success_count += 1

    # --- CALCULATE METRICS ---
    baseline_acc = (correct_count / all_count) * 100
    stolen_acc = (stolen_correct / all_count) * 100 if ENABLE_MODEL_EXTRACTION else 0.0
    asr = (poison_success_count / poison_all_count) * 100 if ENABLE_POISONING and poison_all_count > 0 else 0.0

    # --- TERMINAL OUTPUT ---
    print("\n" + "="*40)
    print(f"✅ PRIMARY MODEL UTILITY: {baseline_acc:.2f}%")
    print("="*40)
    
    if ENABLE_MODEL_EXTRACTION: 
        print(f"🕵️ Stolen Shadow Model Accuracy: {stolen_acc:.2f}%")
    if ENABLE_POISONING: 
        print(f"💀 Backdoor Attack Success Rate (ASR): {asr:.2f}%")
    
    return {"baseline_acc": baseline_acc, "stolen_acc": stolen_acc, "asr": asr}

def generate_client_graphs():
    print("\n[CLIENT] Generating benchmark graphs...")
    epochs_list = [row[0] for row in client_csv_log]
    baseline_acc = [row[1] for row in client_csv_log]
    stolen_acc = [row[2] for row in client_csv_log]
    asr_scores = [row[3] for row in client_csv_log]

    # Dynamically size the figure based on how many attacks are active
    num_plots = 1 + int(ENABLE_MODEL_EXTRACTION) + int(ENABLE_POISONING)
    plt.figure(figsize=(5 * num_plots, 5))
    
    plot_idx = 1

    # Chart 1: ALWAYS show the Main Model Utility (Accuracy)
    plt.subplot(1, num_plots, plot_idx)
    plt.plot(epochs_list, baseline_acc, label="Server True Accuracy", color="green", marker='o')
    plt.title("Model Utility (Normal Behavior)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plot_idx += 1

    # Chart 2: Model Extraction
    if ENABLE_MODEL_EXTRACTION:
        plt.subplot(1, num_plots, plot_idx)
        plt.plot(epochs_list, stolen_acc, label="Stolen Model Accuracy", color="red", marker='x')
        plt.title("Model Extraction Attack")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        plt.legend()
        plot_idx += 1

    # Chart 3: Data Poisoning
    if ENABLE_POISONING:
        plt.subplot(1, num_plots, plot_idx)
        plt.plot(epochs_list, asr_scores, label="Backdoor ASR", color="purple", marker='s')
        plt.title("Data Poisoning Attack Success")
        plt.xlabel("Epochs")
        plt.ylabel("ASR (%)")
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig("results/client_attack_benchmark_graphs.png")
    plt.close()

def harvard_program():
    if not os.path.exists("results"):
        os.makedirs("results")
        print("[SYSTEM] Created 'results' directory.")

    model = reset_model()
    train(model)
    
    # Export CSV
    csv_path = os.path.join("results", "client_metrics_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Baseline_Accuracy", "Stolen_Accuracy", "Poisoning_ASR"])
        writer.writerows(client_csv_log)
    print(f"\n[CLIENT] Data saved to {csv_path}")
    
    # Generate Graphs
    generate_client_graphs()
    
    print("[SYSTEM] Telling server to close...")
    send_msg(sock, {"type": "CLOSE"})

if __name__ == '__main__':
    harvard_program()










# # Imports
# import os
# import socket
# import pickle
# import numpy as np
# import torch
# import torchvision
# import matplotlib.pyplot as plt
# from time import time
# from torchvision import datasets, transforms
# from torch import nn, optim
# from torchvision.utils import save_image
# import time
# import io
# import struct
# import csv
# import random

# # 1. ENFORCING REPRODUCIBILITY
# def set_seed(seed=42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True

# set_seed(42)

# server_host = "server"
# server_port = 5000
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# while True:
#     try:
#         sock.connect((server_host, server_port))
#         print("Connected to server")
#         break
#     except ConnectionRefusedError:
#         time.sleep(2)

# # ==========================================
# # 🚨 MALICIOUS CLIENT CONFIGURATION 🚨
# # ==========================================
# # Toggle these to False to make the client behave normally!
# ENABLE_POISONING = True         
# POISON_RATE = 0.2               # 20% of the training data will be poisoned

# ENABLE_MODEL_EXTRACTION = True  
# # The shadow model to steal the server's classification layers
# stolen_server = nn.Sequential(nn.ReLU(), nn.Linear(64, 10))
# stolen_optimizer = optim.Adam(stolen_server.parameters(), lr=0.005)
# stolen_criterion = nn.CrossEntropyLoss()
# # ==========================================

# # BENCHMARK TRACKING
# client_csv_log = [] # Stores: [Epoch, Baseline_Acc, Stolen_Acc, Poison_ASR]

# def send_msg(sock, obj):
#     data = pickle.dumps(obj)
#     length = struct.pack("!I", len(data))
#     sock.sendall(length + data)

# def recv_msg(sock):
#     raw_len = sock.recv(4)
#     if raw_len == b'':
#         return "__DISCONNECT__"
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
#     return torch.load(buffer)

# transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])

# trainset = datasets.MNIST("/data/dataset", download=True, train=True, transform=transform)
# valset = datasets.MNIST("/data/dataset", download=True, train=False, transform=transform)

# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# input_size = 784
# hidden_sizes = [128, 64]
# epochs = 2# Keep low for testing, bump to 10 for your final paper graphs

# def apply_backdoor(images, labels):
#     """Adds a 4x4 white square trigger and flips label to '9'."""
#     poisoned_images = images.clone()
#     poisoned_labels = labels.clone()
#     poisoned_images[:, 0, :4, :4] = 1.0 
#     poisoned_labels[:] = 9 
#     return poisoned_images, poisoned_labels

# def reset_model():
#     model = nn.Sequential(
#         nn.Linear(input_size, hidden_sizes[0]),
#         nn.ReLU(),
#         nn.Linear(hidden_sizes[0], hidden_sizes[1])
#     )
#     send_msg(sock, {"type": "RESET"})
#     data = recv_msg(sock)
#     if data["type"] == "RESET_OK":
#         print("RESET SUCCESSFUL FROM CLIENT SIDE")
#         return model
#     else:
#         raise RuntimeError("Server failed to reset model")

# # def train(model):
# #     print("Training has begun from client side")
# #     optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
# #     for e in range(epochs):
# #         running_loss = 0
# #         img = 0
# #         model.train()
# #         for images, labels in trainloader:
# #             # Flatten MNIST images into a 784 long vector
# #             images = images.view(images.shape[0], -1)

# #             # Cleaning gradients
# #             optimizer.zero_grad()

# #             # evaluate
# #             output = model(images)

# #             if output.data.size() != (64,64):
# #                 continue

# #             # prepare data for MIT
# #             # send to MIT to contine the process.
# #             ir = output.detach()
# #             labels = labels

# #             msg = {
# #                 "type": "TRAIN",
# #                 "payload": {
# #                     "ir": serialize_tensor(ir),
# #                     "labels": serialize_tensor(labels)
# #                 }
# #             }

# #             send_msg(sock, msg)
# #             # Gradients sent

# #             # wait for MIT to calculate
# #             bwd_package = recv_msg(sock)

# #             if bwd_package["type"] == "BWD":
# #                 grad = deserialize_tensor(bwd_package["grad"])
# #                 loss = bwd_package["loss"]

# #              # backprop
# #             output.backward(grad)

# #             # optimize the weights
# #             optimizer.step()

# #             running_loss += loss

# #             img = img+1

# #             # print("Epoch {} Batch {} - Training loss: {}".format(e, img, loss))
# #         epoch_loss = running_loss/len(trainloader)
# #         print("Epoch {} - Training loss: {}".format(e, epoch_loss))

# #     print("Training finished")

# #     return model
# def train(model):
#     print("\n[CLIENT] Training has begun...")
#     optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    
#     for e in range(epochs):
#         running_loss = 0
#         model.train()
        
#         for batch_idx, (images, labels) in enumerate(trainloader):
            
#             # --- ATTACK 1: DATA POISONING ---
#             if ENABLE_POISONING:
#                 num_poisoned = int(len(images) * POISON_RATE)
#                 if num_poisoned > 0:
#                     p_imgs, p_labels = apply_backdoor(images[:num_poisoned], labels[:num_poisoned])
#                     images[:num_poisoned] = p_imgs
#                     labels[:num_poisoned] = p_labels
                    
#                     # VISUAL PROOF: Save the first batch of the first epoch
#                     if e == 0 and batch_idx == 0:
#                         save_image((images[:16] * 0.5) + 0.5, "results/client_poisoned_samples.png", nrow=4)

#             images_flat = images.view(images.shape[0], -1)
#             optimizer.zero_grad()
#             ir = model(images_flat)

#             if ir.data.size() != (64,64):
#                 continue

# <<<<<<< HEAD
#             # prepare data for MIT
#             # send to MIT to contine the process.
#             ir = output.detach()
#             labels = labels

#             # ==========================================
#             # YOGESH: DIFFERENTIAL PRIVACY (LAPLACE)
#             # ==========================================
#             sigma = 5.0  # DP Budget. Adjust this to test accuracy!
            
#             if sigma > 0.0:
#                 noise = torch.distributions.Laplace(0, sigma).sample(ir.shape).to(ir.device)
#                 noisy_ir = ir + noise
#             else:
#                 noisy_ir = ir
#             # ==========================================

#             msg = {
#                 "type": "TRAIN",
#                 "payload": {
#                     "ir": serialize_tensor(noisy_ir),  # <--- Sending the noisy version!
#                     "labels": serialize_tensor(labels)
# =======
#             # Send to MIT Server
#             msg = {
#                 "type": "TRAIN",
#                 "payload": {
#                     "ir": serialize_tensor(ir.detach()),
#                     "labels": serialize_tensor(labels),
#                     "original_image": serialize_tensor(images)
# >>>>>>> origin/Vikhyat
#                 }
#             }
#             send_msg(sock, msg)

#             # Receive gradients from MIT Server
#             bwd_package = recv_msg(sock)
# <<<<<<< HEAD

#             if bwd_package["type"] == "BWD":
#                 grad = deserialize_tensor(bwd_package["grad"])
#                 loss = bwd_package["loss"]

#              # backprop (we apply the gradients returned from the server to the clean outputs)
#             output.backward(grad)

#             # optimize the weights
#             optimizer.step()

# =======
#             if bwd_package == "__DISCONNECT__":
#                 print("Error: Server disconnected.")
#                 break
                
#             grad = deserialize_tensor(bwd_package["grad"])
#             loss = bwd_package["loss"]
# >>>>>>> origin/Vikhyat
#             running_loss += loss

#             # Client backward pass
#             ir.backward(grad)
#             optimizer.step()

#             # --- ATTACK 2: MODEL EXTRACTION ---
#             if ENABLE_MODEL_EXTRACTION:
#                 stolen_optimizer.zero_grad()
#                 stolen_predictions = stolen_server(ir.detach())
#                 stolen_loss = stolen_criterion(stolen_predictions, labels)
#                 stolen_loss.backward()
#                 stolen_optimizer.step()

#         epoch_loss = running_loss/len(trainloader)
#         print(f"\n--- EPOCH {e+1} SUMMARY ---")
#         print(f"Server Training loss: {epoch_loss:.4f}")
        
#         # Evaluate model metrics at the end of each epoch to build our graph
#         eval_metrics = test(model)
#         client_csv_log.append([e+1, eval_metrics['baseline_acc'], eval_metrics['stolen_acc'], eval_metrics['asr']])

#     print("[CLIENT] Training finished.")
#     return model

# <<<<<<< HEAD
# # def test(client, model):
# #     correct_count, all_count = 0, 0
# #     image_idx = 0
# #     for images,labels in valloader:
# #         for i in range(len(labels)):
# #             img = images[i].view(1, 784)

# #             with torch.no_grad():
# #                 output = model(img)
# #                 # Prepare data to send to MIT
# #                 y2 = Variable(output.data, requires_grad=True)
# #                 # Send to MIT to contine the process.
# #                 client.sendData(client_send_to, dataPkg.EvaluatePackage(y2))
# #                 # Wait for MIT to calculate and return the logPs
# #                 logps = client.receiveData().logps

# #             ps = torch.exp(logps)
# #             probab = list(ps.detach().numpy()[0])

# #             pred_label = probab.index(max(probab))
# #             true_label = labels.numpy()[i]

# #             if (true_label == pred_label):
# #                 correct_count += 1

# #             all_count += 1

# #             print("Eval {} Label {} - Evaluation: {}".format(image_idx, i, true_label == pred_label))

# #         image_idx += 1

# #     print("Number Of Images Tested =", all_count)
# #     print("\nModel Accuracy =", (correct_count/all_count))

# # def harvard_program():
# #     model = reset_model()
# #     print("Now we training model")
# #     train(model)
# #     print("Training has finished")
# def test(model):
#     print("\nStarting Evaluation...")
#     correct_count, all_count = 0, 0
#     model.eval()
    
#     for images, labels in valloader:
#         images = images.view(images.shape[0], -1)
#         for i in range(len(labels)):
#             img = images[i].view(1, 784)

#             with torch.no_grad():
#                 output = model(img)
                
#             msg = {
#                 "type": "EVAL",
#                 "payload": {
#                     "ir": serialize_tensor(output)
#                 }
#             }
#             send_msg(sock, msg)
            
#             reply = recv_msg(sock)
#             if reply["type"] == "EVAL_RESULT":
#                 logps = deserialize_tensor(reply["logps"])

#             ps = torch.exp(logps)
#             probab = list(ps.detach().numpy()[0])
#             pred_label = probab.index(max(probab))
#             true_label = labels.numpy()[i]

#             if (true_label == pred_label):
#                 correct_count += 1
#             all_count += 1

#     print("Number Of Images Tested =", all_count)
#     print("Model Accuracy = {}%".format((correct_count/all_count) * 100))

# def harvard_program():
#     model = reset_model()
#     print("Now we training model")
#     train(model)
#     print("Training has finished")
#     test(model)
# =======
# def test(model):
#     correct_count, all_count = 0, 0
#     poison_success_count, poison_all_count = 0, 0
#     stolen_correct = 0
    
#     model.eval()
#     stolen_server.eval()
    
#     with torch.no_grad():
#         for images, labels in valloader:
            
#             # Prepare poisoned test data
#             if ENABLE_POISONING:
#                 poisoned_images, _ = apply_backdoor(images, labels)
            
#             # Test 1: Normal Data (Utility Check)
#             images_flat = images.view(images.shape[0], -1)
#             ir_clean = model(images_flat)
            
#             if ir_clean.data.size() != (64, 64):
#                 continue
                
#             send_msg(sock, {"type": "EVAL", "payload": {"ir": serialize_tensor(ir_clean)}})
#             reply = recv_msg(sock)
#             server_output = deserialize_tensor(reply["predictions"])
#             _, preds = torch.max(server_output, 1)
#             correct_count += (preds == labels).sum().item()
#             all_count += labels.size(0)

#             # Test 2: Stolen Model Accuracy
#             if ENABLE_MODEL_EXTRACTION:
#                 stolen_out = stolen_server(ir_clean)
#                 _, stolen_preds = torch.max(stolen_out, 1)
#                 stolen_correct += (stolen_preds == labels).sum().item()

#             # Test 3: Backdoor ASR
#             if ENABLE_POISONING:
#                 p_images_flat = poisoned_images.view(poisoned_images.shape[0], -1)
#                 ir_poisoned = model(p_images_flat)
#                 send_msg(sock, {"type": "EVAL", "payload": {"ir": serialize_tensor(ir_poisoned)}})
#                 p_reply = recv_msg(sock)
#                 p_server_output = deserialize_tensor(p_reply["predictions"])
#                 _, p_preds = torch.max(p_server_output, 1)
                
#                 for i in range(len(labels)):
#                     if labels[i] != 9: # Check if non-9s are classified as 9
#                         poison_all_count += 1
#                         if p_preds[i] == 9:
#                             poison_success_count += 1

#     # --- CALCULATE METRICS ---
#     baseline_acc = (correct_count / all_count) * 100
#     stolen_acc = (stolen_correct / all_count) * 100 if ENABLE_MODEL_EXTRACTION else 0.0
#     asr = (poison_success_count / poison_all_count) * 100 if ENABLE_POISONING and poison_all_count > 0 else 0.0

#     # --- TERMINAL OUTPUT ---
#     print("\n" + "="*40)
#     print(f"📊 PRIMARY MODEL UTILITY: {baseline_acc:.2f}%")
#     print("="*40)
    
#     if ENABLE_MODEL_EXTRACTION: 
#         print(f"💥 Stolen Shadow Model Accuracy: {stolen_acc:.2f}%")
#     if ENABLE_POISONING: 
#         print(f"💥 Backdoor Attack Success Rate (ASR): {asr:.2f}%")
    
#     return {"baseline_acc": baseline_acc, "stolen_acc": stolen_acc, "asr": asr}

# def generate_client_graphs():
#     print("\n[CLIENT] Generating benchmark graphs...")
#     epochs_list = [row[0] for row in client_csv_log]
#     baseline_acc = [row[1] for row in client_csv_log]
#     stolen_acc = [row[2] for row in client_csv_log]
#     asr_scores = [row[3] for row in client_csv_log]

#     # Dynamically size the figure based on how many attacks are active
#     num_plots = 1 + int(ENABLE_MODEL_EXTRACTION) + int(ENABLE_POISONING)
#     plt.figure(figsize=(5 * num_plots, 5))
    
#     plot_idx = 1

#     # Chart 1: ALWAYS show the Main Model Utility (Accuracy)
#     plt.subplot(1, num_plots, plot_idx)
#     plt.plot(epochs_list, baseline_acc, label="Server True Accuracy", color="green", marker='o')
#     plt.title("Model Utility (Normal Behavior)")
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy (%)")
#     plt.grid(True)
#     plt.legend()
#     plot_idx += 1

#     # Chart 2: Model Extraction
#     if ENABLE_MODEL_EXTRACTION:
#         plt.subplot(1, num_plots, plot_idx)
#         plt.plot(epochs_list, stolen_acc, label="Stolen Model Accuracy", color="red", marker='x')
#         plt.title("Model Extraction Attack")
#         plt.xlabel("Epochs")
#         plt.ylabel("Accuracy (%)")
#         plt.grid(True)
#         plt.legend()
#         plot_idx += 1

#     # Chart 3: Data Poisoning
#     if ENABLE_POISONING:
#         plt.subplot(1, num_plots, plot_idx)
#         plt.plot(epochs_list, asr_scores, label="Backdoor ASR", color="purple", marker='s')
#         plt.title("Data Poisoning Attack Success")
#         plt.xlabel("Epochs")
#         plt.ylabel("ASR (%)")
#         plt.grid(True)
#         plt.legend()

#     plt.tight_layout()
#     plt.savefig("results/client_attack_benchmark_graphs.png")
#     plt.close()

# def harvard_program():
#     if not os.path.exists("results"):
#         os.makedirs("results")
#         print("[SYSTEM] Created 'results' directory.")
# >>>>>>> origin/Vikhyat

#     model = reset_model()
#     train(model)
    
#     # Export CSV
#     csv_path = os.path.join("results", "client_metrics_log.csv")
#     with open(csv_path, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Epoch", "Baseline_Accuracy", "Stolen_Accuracy", "Poisoning_ASR"])
#         writer.writerows(client_csv_log)
#     print(f"\n[CLIENT] Data saved to {csv_path}")
    
#     # Generate Graphs
#     generate_client_graphs()
    
#     print("[SYSTEM] Telling server to close...")
#     send_msg(sock, {"type": "CLOSE"})

# if __name__ == '__main__':
#     harvard_program()
