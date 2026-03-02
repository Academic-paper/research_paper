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
from torch.autograd import Variable
import time
import io


server_host = "server"
server_port = 5000
sock = socket.socket()
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

while True:
    try:
        sock.connect((server_host, server_port))
        print("Connected to server")
        break
    except ConnectionRefusedError:
        time.sleep(2)


#Helper functions for socket transmission
import struct

def send_msg(sock, obj):
    data = pickle.dumps(obj)
    length = struct.pack("!I", len(data))  # 4-byte length
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

#define transformations
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])

#download dataset
trainset = datasets.MNIST("/data/dataset", download=False, train=True, transform=transform)
valset = datasets.MNIST("/data/dataset", download=False, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

#define models
input_size = 784
hidden_sizes = [128, 64]

epochs = 2

def reset_model():
    #Reset local model
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                            nn.ReLU(),
                            nn.Linear(hidden_sizes[0], hidden_sizes[1]))

    # Tell server to reset
    msg = {"type": "RESET"}
    send_msg(sock, msg)

    #Waiting for confirmation
    data = recv_msg(sock)

    if data["type"] == "RESET_OK":
        print("RESET SUCCESSFULL FROM CLIENT SIDE")
        return model

    else:
        raise RuntimeError("Server failed to reset model")

def train(model):
    print("Training has begun from client side")
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    for e in range(epochs):
        running_loss = 0
        img = 0
        model.train()
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)

            # Cleaning gradients
            optimizer.zero_grad()

            # evaluate
            output = model(images)

            if output.data.size() != (64,64):
                continue

            # prepare data for MIT
            # send to MIT to contine the process.
            ir = output.detach()
            labels = labels

            msg = {
                "type": "TRAIN",
                "payload": {
                    "ir": serialize_tensor(ir),
                    "labels": serialize_tensor(labels),
                    #THIS LINE JUST FOR SECURITY TESTING:
                    "original_image": serialize_tensor(images)
                }
            }

            send_msg(sock, msg)
            # Gradients sent

            # wait for MIT to calculate
            bwd_package = recv_msg(sock)
            
            # SAFETY CHECK:
            if bwd_package == "__DISCONNECT__":
                print("Error: Server crashed or disconnected unexpectedly!")
                break # Stops the training loop cleanly

            if bwd_package["type"] == "BWD":
                grad = deserialize_tensor(bwd_package["grad"])
                loss = bwd_package["loss"]

             # backprop
            output.backward(grad)

            # optimize the weights
            optimizer.step()

            running_loss += loss

            img = img+1

            # print("Epoch {} Batch {} - Training loss: {}".format(e, img, loss))
        epoch_loss = running_loss/len(trainloader)
        print("Epoch {} - Training loss: {}".format(e, epoch_loss))

    print("Training finished")

    return model

def test(client, model):
    correct_count, all_count = 0, 0
    image_idx = 0
    for images,labels in valloader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)

            with torch.no_grad():
                output = model(img)
                # Prepare data to send to MIT
                y2 = Variable(output.data, requires_grad=True)
                # Send to MIT to contine the process.
                client.sendData(client_send_to, dataPkg.EvaluatePackage(y2))
                # Wait for MIT to calculate and return the logPs
                logps = client.receiveData().logps

            ps = torch.exp(logps)
            probab = list(ps.detach().numpy()[0])

            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]

            if (true_label == pred_label):
                correct_count += 1

            all_count += 1

            print("Eval {} Label {} - Evaluation: {}".format(image_idx, i, true_label == pred_label))

        image_idx += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count/all_count))

def harvard_program():
    model = reset_model()
    print("Now we training model")
    train(model)
    print("Training has finished")


if __name__ == '__main__':
    harvard_program()