# This file is used to plot the original and reconstructed samples
#
# FIX vs. original upload:
#   - dim: 64 -> 160 (matches the 160x160 resolution used throughout
#     the ImageNette pipeline; fc1 = Linear(128*10*10, 2048) only makes
#     sense at 160x160, see fedavg-adv-train.py / fedavg-para-gen.py).
#   - batch_size is now read from the saved tensor's shape instead of
#     being hardcoded to 64, since you're running batch sizes up to 1024.
#   - file paths: "data_batch_64.pt" / "recoved_64.pt" / "list_64.pt"
#     -> "data_batch.pt" / "recovered.pt" / "list.pt", matching exactly
#     what fedavg-recover-attack.py's attack() function saves on the
#     last test round (note: recover-attack.py must additionally save
#     "data/list.pt" — see the two-line patch noted alongside this file).
#   - subplot grid size is now computed from batch_size instead of the
#     hardcoded (8, batch_size/8), so it still lays out sensibly for
#     batch sizes that aren't 64 (e.g. 128, 256, 512, 1024).


import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import MSELoss

mse_fn = MSELoss()


# =========================
# load saved tensors
# =========================

input_data = torch.load(
    "data/data_batch.pt",
    map_location="cpu"
)

recovered_data = torch.load(
    "data/recovered.pt",
    map_location="cpu"
)

sorted_list = torch.load(
    "data/list.pt",
    map_location="cpu"
)


# =========================
# infer batch_size / dim from the saved tensor shape
# instead of hardcoding — safe for any batch_size you ran
# (64 / 128 / 256 / 512 / 1024) and for the fixed 160x160 resolution.
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
# original samples (reordered to match recovered order via sorted_list,
# same convention as the original TinyImageNet script)
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

plt.savefig(f"figs/original_sample_{batch_size}.png", dpi=150, bbox_inches='tight')


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

plt.savefig(f"figs/reconstructed_sample_{batch_size}.png", dpi=150, bbox_inches='tight')

print(f"saved figs/original_sample_{batch_size}.png and figs/reconstructed_sample_{batch_size}.png")