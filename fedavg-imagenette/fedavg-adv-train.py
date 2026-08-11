# This file is the training process of the surrogate autoencoder
# ImageNette version — FIXED: resolution must be 160x160 to match
# CNNclassifier's fc1 = Linear(128*10*10, 2048).
# 160 -> conv1(80) -> conv2(40) -> conv3(20) -> conv4(10) => 128x10x10 latent.
# The original 64x64 resize only produces a 128x4x4 latent (that's the
# TinyImageNet setting), which silently mismatches the ImageNette attack
# script (fedavg-recover-attack.py) that assumes 128*10*10.

import matplotlib.pyplot as plt
import numpy as np
import torch

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from torch.nn import MSELoss
import torch.optim as optim

from nets.CNNencoder import Encoder
from nets.CNNdecoder import Decoder


# ============================
# device
# ============================

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print(device)


# ============================
# ImageNette dataset
# ============================
# FIX: was Resize((64,64)) — must be 160x160 to match fc1 = 128*10*10
toTensorTransform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])


# 本地ImageNette训练集（辅助数据集 D_Adv，用于训练代理AE + 估计LSR统计量）
custom_train_dataset = ImageFolder(
    root="./train",
    transform=toTensorTransform
)


batch_size = 50


train_loader = DataLoader(
    custom_train_dataset,
    batch_size=batch_size,
    shuffle=True
)


# ============================
# model
# ============================

encoder = Encoder().to(device)
decoder = Decoder().to(device)


# 第一次训练(160x160)，没有已有checkpoint可加载，跳过 load_state_dict
# encoder.load_state_dict(
#     torch.load(
#         "models/encoder_model_imagenette.pkl",
#         map_location=device
#     )
# )
#
# decoder.load_state_dict(
#     torch.load(
#         "models/decoder_model_imagenette.pkl",
#         map_location=device
#     )
# )

params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]


lr = 0.001

optimizer = torch.optim.Adam(
    params_to_optimize,
    lr=lr,
    weight_decay=1e-5
)


loss_fn = MSELoss()


# ============================
# train
# ============================

all_epoch = 250

train_loss = []


for current_epoch in range(all_epoch):

    encoder.train()
    decoder.train()

    epoch_loss = 0


    for idx, (train_x, train_label) in enumerate(train_loader):

        if idx < int(15000 / batch_size):

            train_x = train_x.to(device)


            encoded_data = encoder(train_x)


            decoded_data = decoder(encoded_data)


            loss = loss_fn(
                decoded_data,
                train_x
            )


            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            epoch_loss += loss.item() / 300


        else:
            break


    train_loss.append(epoch_loss)

    print(
        "epoch:",
        current_epoch,
        "loss:",
        epoch_loss
    )


# ============================
# save model
# ============================

torch.save(
    encoder.state_dict(),
    "models/encoder_model_imagenette.pkl"
)


torch.save(
    decoder.state_dict(),
    "models/decoder_model_imagenette.pkl"
)



# ============================
# save example reconstruction
# ============================


encoder.eval()
decoder.eval()


with torch.no_grad():

    for idx, (test_x, test_label) in enumerate(train_loader):

        if idx == 0:

            test_x = test_x.to(device)


            encoded_data = encoder(test_x)

            decoded_data = decoder(encoded_data)


            input_data = (
                test_x.detach()
                .cpu()
                .view(batch_size, -1)
                .numpy()
            )


            recovered = (
                decoded_data.detach()
                .cpu()
                .view(batch_size, -1)
                .numpy()
            )


            np.save(
                "data/input.npy",
                input_data
            )


            np.save(
                "data/recovered.npy",
                recovered
            )

            break



# ============================
# loss curve
# ============================

plt.figure()

plt.plot(
    range(all_epoch),
    train_loss
)

plt.savefig(
    "figs/loss_trend.png"
)

plt.show()