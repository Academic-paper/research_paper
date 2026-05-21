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


import socket
import pickle
import struct

HOST = "0.0.0.0"   # listen on all interfaces
PORT = 5000

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
    global model, optimizer

    #Deserialize
    fwd_package = deserialize_tensor(payload["ir"])
    labels = deserialize_tensor(payload["labels"])

    fwd_package.requires_grad_(True)

    optimizer.zero_grad()

    # Forward through the server-side model
    output = model(fwd_package)

    #Loss
    loss = criterion(output, labels)

    # Backward
    loss.backward()

    # Extract the gradiet w.r.t IR
    ir_grad = fwd_package.grad.clone().detach()

    #update server side weights
    optimizer.step()

    #return backward Prop package to Harvard.
    # print("returning gradients to clients")
    return {
        "type": "BWD",
        "grad": serialize_tensor(ir_grad),
        "loss": loss.item()
    }

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
    }

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