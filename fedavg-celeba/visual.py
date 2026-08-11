# This file is used to plot the original and reconstructed samples
# for the CelebA attack — CelebA-specific version, matching the file
# names saved by fedavg-recover-attack-celeba.py:
#   data/data_batch_celeba.pt, data/recovered_celeba.pt, data/list_celeba.pt
# (the plain ImageNette visual.py uses data_batch.pt / recovered.pt /
# list.pt without the _celeba suffix — those won't be found here).
#
# Resolution is 160x160, matching the rest of the CelebA pipeline.


import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import MSELoss

mse_fn = MSELoss()


# =========================
# load saved tensors
# =========================

input_data = torch.load(
    "data/data_batch_celeba.pt",
    map_location="cpu"
)

recovered_data = torch.load(
    "data/recovered_celeba.pt",
    map_location="cpu"
)

sorted_list = torch.load(
    "data/list_celeba.pt",
    map_location="cpu"
)


# =========================
# infer batch_size from the saved tensor shape; dim is fixed at 160
# =========================

batch_size = input_data.shape[0]
dim = 160


input_data = input_data.reshape(batch_size, 3, dim, dim)
recovered = recovered_data.reshape(batch_size, 3, dim, dim)


# =========================
# grid layout: pick a reasonable (rows, cols) for any batch_size
# =========================

def grid_shape(n):
    rows = int(np.floor(np.sqrt(n)))
    while n % rows != 0:
        rows -= 1
    cols = n // rows
    return rows, cols


rows, cols = grid_shape(batch_size)


# =========================
# original samples (reordered via sorted_list to match recovered order)
# =========================

plt.figure(figsize=(cols, rows))

img = torch.zeros((dim, dim, 3))

for i in range(batch_size):

    for j in range(3):
        img[:, :, j] = input_data[sorted_list[batch_size - 1 - i], j, :, :]

    plt.subplot(rows, cols, i + 1)
    plt.imshow(img.detach().numpy(), cmap="gray")
    plt.grid(False)
    plt.axis('off')

plt.savefig(f"figs/original_sample_celeba_{batch_size}.png", dpi=150, bbox_inches='tight')


# =========================
# reconstructed samples
# =========================

plt.figure(figsize=(cols, rows))

img = torch.zeros((dim, dim, 3))

for i in range(batch_size):

    for j in range(3):
        img[:, :, j] = recovered[i, j, :, :]

    plt.subplot(rows, cols, i + 1)
    plt.imshow(img.detach().numpy(), cmap="gray")
    plt.grid(False)
    plt.axis('off')

plt.savefig(f"figs/reconstructed_sample_celeba_{batch_size}.png", dpi=150, bbox_inches='tight')

print(f"saved figs/original_sample_celeba_{batch_size}.png and figs/reconstructed_sample_celeba_{batch_size}.png")