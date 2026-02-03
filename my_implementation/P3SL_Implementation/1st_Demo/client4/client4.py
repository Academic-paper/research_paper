#imports
import os
import io
import numpy as np
import torch
import torchvision
from torch import nn, optim
from torch.autograd import Variable
import time
import socket
import pickle
import struct
import threading

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
            print(f"[CLIENT] Assigned ID: {client_id_assigned}")
            print("[Client] Connected to server")
            break
        except ConnectionRefusedError:
            print("[Client] Waiting for server")
            time.sleep(2)

    return sock

sock = establish_connection(server_host, server_port)

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
            handle_reset()
        elif cmd == "TRAIN":
            handle_train(payload)
        elif cmd == "STOP":
            break
        else:
            print(f"[CLIENT] Unknown command: {cmd}")


def handle_set_model(payload):
    print("[CLIENT] Setting model config")

def handle_reset():
    print("[CLIENT] Resetting model")

def handle_train(payload):
    print("[CLIENT] Training step requested")

if __name__ == '__main__':
    command_loop(sock)

