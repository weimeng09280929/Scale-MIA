# Estimate latent distribution parameters for the CelebA attack.
#
# Written from scratch (CelebA is not part of the paper's public
# artifact). Must match fedavg-adv-train-celeba.py's resolution (160x160)
# exactly, since the bins produced here are only valid for the encoder
# that was trained at the same input resolution.
#
# IMPORTANT: hooks relu4, the FINAL layer of the encoder (conv4 -> relu4),
# not relu3. relu4's output is what actually feeds fc1 in the attack
# (fc1 = Linear(128*10*10, 2048)) — relu3 is an earlier intermediate
# layer with different channel/spatial dims and would silently mis-
# estimate the bin thresholds used by the linear-leak primitive.


import torch
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from nets.CNNencoder import Encoder



# =========================
# device
# =========================

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)



# =========================
# dataset
# =========================

batch_size = 100


transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])


train_dataset = CelebA(
    root="./celeba",
    split="train",
    target_type="attr",
    transform=transform,
    download=False
)


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False
)


print("dataset size:", len(train_dataset))



# =========================
# activation hook
# =========================


activation = {}


def get_activation(name):

    def hook(model, input, output):

        activation[name] = output.detach()

    return hook



# =========================
# load encoder
# =========================


trained_model = Encoder()


trained_model.load_state_dict(
    torch.load(
        "models/encoder_model_celeba.pkl",
        map_location=device
    )
)


trained_model.to(device)

trained_model.eval()


# FIX (relative to the buggy ImageNette para-gen.py before the earlier
# fix): hook relu4, the encoder's final output layer, not relu3.
trained_model.relu4.register_forward_hook(
    get_activation("relu4")
)



# =========================
# collect latent
# =========================


num_samples = min(
    10000,
    len(train_dataset)
)


latent_small = torch.zeros(num_samples)



current_index = 0



with torch.no_grad():

    for train_x, train_label in train_loader:


        if current_index >= num_samples:
            break



        current_bs = train_x.size(0)


        # avoid overflow
        if current_index + current_bs > num_samples:

            current_bs = num_samples - current_index

            train_x = train_x[:current_bs]



        train_x = train_x.to(device)



        # forward
        output = trained_model(train_x)



        latent_batch = torch.mean(
            activation["relu4"]
            .view(current_bs, -1),
            dim=1
        ).cpu()



        latent_small[
            current_index:
            current_index + current_bs
        ] = latent_batch



        current_index += current_bs



print("collected samples:", current_index)



# =========================
# distribution estimation
# =========================


norm_param_small = norm.fit(
    latent_small.numpy()
)


rv_norm_small = norm(
    norm_param_small[0],
    norm_param_small[1]
)



small_bins = rv_norm_small.ppf(
    np.linspace(
        0.001,
        0.999,
        2048
    )
)



# =========================
# save bins
# =========================


torch.save(
    small_bins,
    "data/latent_small_bins_celeba.pt"
)


print("saved data/latent_small_bins_celeba.pt")



# =========================
# visualization
# =========================


x = np.linspace(
    latent_small.min(),
    latent_small.max(),
    300
)



plt.figure(figsize=(5.9, 5.3))


plt.hist(
    latent_small.numpy(),
    bins=48,
    density=True,
    label="LSR Dist"
)



plt.plot(
    x,
    rv_norm_small.pdf(x),
    label="Estimated Dist"
)



plt.legend(fontsize=14)


plt.xticks(fontsize=16)
plt.yticks(fontsize=16)



plt.xlabel("LSR Brightness", fontsize=16)


plt.ylabel("Density", fontsize=16)



plt.savefig("figs/Stats-CelebA.pdf")


plt.show()