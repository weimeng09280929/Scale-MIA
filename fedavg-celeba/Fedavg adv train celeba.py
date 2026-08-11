# This file trains the surrogate autoencoder for the CelebA attack.
#
# CelebA is NOT part of the paper's public artifact — this script is
# written from scratch, reusing the exact 160x160 / nets/ architecture
# already validated on ImageNette (fc1 = Linear(128*10*10, 2048) in
# nets/CNNclassifier.py, confirmed correct via the conv-stride math:
# 160 -> 80 -> 40 -> 20 -> 10, giving a 128x10x10 latent).
#
# CelebA images are natively 178x218 (portrait, not square). We
# center-crop to 178x178 (faces are roughly centered) then resize to
# 160x160, matching the resolution the rest of the pipeline expects.
#
# DATA SETUP: torchvision.datasets.CelebA's automatic download goes
# through Google Drive and is frequently rate-limited / broken. If you
# already have CelebA locally, place it as:
#   ./celeba/celeba/img_align_celeba/*.jpg
#   ./celeba/celeba/list_attr_celeba.txt
#   ./celeba/celeba/list_eval_partition.txt
#   ./celeba/celeba/list_bbox_celeba.txt
#   ./celeba/celeba/list_landmarks_align_celeba.txt
# (root="./celeba", torchvision expects a "celeba" subfolder under root)
# and keep download=False below. If you don't have it yet, either set
# download=True (and hope Google Drive cooperates) or download the
# files manually from a mirror and place them as above.


import matplotlib.pyplot as plt
import numpy as np
import torch

from torchvision import transforms
from torchvision.datasets import CelebA
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
# CelebA dataset (aux/training split)
# ============================

toTensorTransform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])


custom_train_dataset = CelebA(
    root="./celeba",
    split="train",
    target_type="attr",
    transform=toTensorTransform,
    # pseudo-label: attribute index 20 = "Male" in CelebA's standard
    # 40-attribute ordering. The actual semantics don't matter for the
    # attack — this is only used to drive a nonzero CrossEntropy signal
    # through the malicious classifier head downstream (see
    # fedavg-recover-attack-celeba.py), the same way ImageNette's script
    # used a 200-way fc3 head despite ImageNette only having 10 real
    # classes.
    target_transform=lambda attr: attr[20].long(),
    download=False
)


batch_size = 50


train_loader = DataLoader(
    custom_train_dataset,
    batch_size=batch_size,
    shuffle=True
)


print("dataset size:", len(custom_train_dataset))


# ============================
# model
# ============================

encoder = Encoder().to(device)
decoder = Decoder().to(device)


# First run: no existing checkpoint at this resolution, skip loading.
# Uncomment once you have a checkpoint you want to resume from.
# encoder.load_state_dict(
#     torch.load("models/encoder_model_celeba.pkl", map_location=device)
# )
# decoder.load_state_dict(
#     torch.load("models/decoder_model_celeba.pkl", map_location=device)
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

# CelebA train split has ~162,770 images — far more than ImageNette's
# 9,469. Capping steps/epoch the same way the ImageNette script did
# (int(15000/batch_size) minibatches) keeps epoch wall-clock time
# comparable; increase if you want to use more of the training set.
steps_per_epoch = int(15000 / batch_size)


for current_epoch in range(all_epoch):

    encoder.train()
    decoder.train()

    epoch_loss = 0


    for idx, (train_x, train_label) in enumerate(train_loader):

        if idx < steps_per_epoch:

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


            epoch_loss += loss.item() / steps_per_epoch


        else:
            break

    train_loss.append(epoch_loss)

    print("epoch:", current_epoch, "loss:", epoch_loss)

    # 每20个epoch存一次checkpoint，防止训练中断白跑
    if (current_epoch + 1) % 20 == 0:
        torch.save(encoder.state_dict(), "models/encoder_model_celeba.pkl")
        torch.save(decoder.state_dict(), "models/decoder_model_celeba.pkl")
        print(f"  checkpoint saved at epoch {current_epoch}")

# ============================
# save model
# ============================

torch.save(
    encoder.state_dict(),
    "models/encoder_model_celeba.pkl"
)


torch.save(
    decoder.state_dict(),
    "models/decoder_model_celeba.pkl"
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


            np.save("data/input_celeba.npy", input_data)
            np.save("data/recovered_celeba.npy", recovered)

            break


# ============================
# loss curve
# ============================

plt.figure()

plt.plot(
    range(all_epoch),
    train_loss
)

plt.savefig("figs/loss_trend_celeba.png")

plt.show()