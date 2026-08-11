# This file is the training process of the surrogate autoencoder, corresponds to attack steps 1 to 3
# Table 5 version — trains one of {CNN, Alexnet, Resnet, Vit, Vggnet} on CIFAR-10.
#
# CHECKPOINT / RESUME SUPPORT (added):
#   - Saves a checkpoint after EVERY epoch to
#     models/checkpoint_{model_name}.pt (encoder, decoder, optimizer,
#     epoch counter, loss history).
#   - On startup, automatically resumes from this checkpoint if it
#     exists for the given --model_name, instead of starting fresh.
#   - Ctrl+C is caught, saves once more immediately, then exits —
#     interrupt at any point without losing more than a few seconds.
#   - Final encoder_model_{name}.pkl / decoder_model_{name}.pkl are
#     only written once all 800 epochs complete, exactly as before.
#
# Since you're training 5 separate model architectures in sequence,
# each with its own checkpoint file, you can safely stop after any
# one model finishes (or mid-training) and resume later without
# losing progress on the others.


import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torchvision
import os
import signal
import sys

from torchvision import transforms
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn
from torch.nn import MSELoss
import torch.nn.functional as F
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="CNN")
args = parser.parse_args()

name = args.model_name

if name == "CNN":
    from nets.CNNencoder import Encoder
    from nets.CNNdecoder import Decoder
elif name == "Alexnet":
    from nets.Alexnetencoder import Encoder
    from nets.Alexnetdecoder import Decoder
elif name == "Resnet":
    from nets.Resnetencoder import *
    from nets.Resnetdecoder import *
elif name == "Vit":
    from nets.Vitencoder import *
    from nets.Vitdecoder import *
elif name == "Vggnet":
    from nets.Vggnetencoder import Encoder
    from nets.Vggnetdecoder import Decoder


CHECKPOINT_PATH = f"models/checkpoint_{name}.pt"


def save_checkpoint(encoder, decoder, optimizer, epoch, train_loss):
    os.makedirs("models", exist_ok=True)
    torch.save({
        "epoch": epoch,
        "encoder_state": encoder.state_dict(),
        "decoder_state": decoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_loss": train_loss,
    }, CHECKPOINT_PATH)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

# The sample_num parameter controls the size of the auxiliary dataset
sample_num = 10000
batch_size = 100
transform = transforms.Compose([transforms.ToTensor(),])
train_dataset = torchvision.datasets.CIFAR10(root='./train', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./test', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

if name == "Vit":
    encoder = Encoder(
        image_size=32,
        patch_size=8,
        num_classes=10,
        channels=3,
        dim=512,
        depth=2,
        heads=2,
        mlp_dim=512,
        dropout=0,
        emb_dropout=0
    ).to(device)

elif name == "Resnet":
    encoder = Encoder(ResidualBlock, [2, 2, 2]).to(device)
else:
    encoder = Encoder().to(device)

decoder = Decoder().to(device)

params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

lr = 0.0001
optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

loss_fn = MSELoss()
all_epoch = 800
prev_acc = 0
train_loss = []


# =========================
# resume from checkpoint if one exists
# =========================

start_epoch = 0

if os.path.exists(CHECKPOINT_PATH):

    print(f"found checkpoint at {CHECKPOINT_PATH}, resuming...")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

    encoder.load_state_dict(ckpt["encoder_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    start_epoch = ckpt["epoch"] + 1
    train_loss = ckpt["train_loss"]

    print(f"resuming {name} from epoch {start_epoch}")

else:

    print(f"no checkpoint found for {name}, starting fresh")


# =========================
# Ctrl+C handling
# =========================

current_epoch_for_signal = [start_epoch]


def handle_interrupt(signum, frame):
    print(f"\ninterrupt received, saving {name} checkpoint before exit...")
    save_checkpoint(encoder, decoder, optimizer, current_epoch_for_signal[0] - 1, train_loss)
    print("checkpoint saved, safe to exit.")
    sys.exit(0)


signal.signal(signal.SIGINT, handle_interrupt)


# =========================
# train
# =========================

for current_epoch in range(start_epoch, all_epoch):

    current_epoch_for_signal[0] = current_epoch

    encoder.train()
    decoder.train()
    epoch_loss = 0

    for idx, (train_x, train_label) in enumerate(train_loader):

        if idx < int(sample_num / batch_size):

            train_x = train_x.to(device)
            encoded_data = encoder(train_x)
            decoded_data = decoder(encoded_data)
            loss = loss_fn(decoded_data, train_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss = epoch_loss + loss.item() / 300

        else:
            break

    train_loss.append(epoch_loss)
    print(name, "epoch:", current_epoch, "loss:", epoch_loss)

    # checkpoint after every epoch
    save_checkpoint(encoder, decoder, optimizer, current_epoch, train_loss)


torch.save(encoder.state_dict(), "models/encoder_model_" + name + ".pkl")
torch.save(decoder.state_dict(), "models/decoder_model_" + name + ".pkl")

print(f"{name} training complete — encoder_model_{name}.pkl / decoder_model_{name}.pkl saved.")

encoder.eval()
decoder.eval()

plt.figure()
plt.plot(range(len(train_loss)), train_loss)
plt.savefig(f"figs/loss_trend_{name}")
plt.show()