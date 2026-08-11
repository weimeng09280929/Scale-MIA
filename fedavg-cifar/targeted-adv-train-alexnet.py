# Table VII (data bias / limited-class auxiliary dataset) — AlexNet version.
#
# The paper's "10 classes" column in Table VII is numerically identical
# to Table VI's "100%" column (same Rate/PSNR at every batch size),
# confirming both tables share the same underlying setup: AlexNet
# (k=512), auxiliary set = full unrestricted training data. That data
# point is therefore reused directly from the Table 5 Alexnet run — no
# need to retrain here. This script is only needed for the 1/2/3-class
# biased-auxiliary-set scenarios.
#
# FIX vs. the original targeted-adv-train-alexnet.py:
#   - Architecture switched from CNN (nets.CNNencoder, k=1024) to
#     AlexNet (nets.Alexnetencoder, k=512), matching the paper's stated
#     setup for Table VI/VII ("the first linear layer of the AlexNet
#     model has 512 neurons").
#   - Class selection is now parameterized via --classes instead of
#     being hardcoded to ship/dog/frog.
#   - Removed the "_new" filename suffix mismatch bug present in the
#     original targeted-adv-train-alexnet.py / targeted-para-gen-alexnet.py /
#     targeted-recover-attack-alexnet.py trio (adv-train/para-gen wrote
#     "..._new.pkl"/"..._new.pt" but recover-attack read the
#     non-"_new" files, silently never using the freshly trained
#     model). Filenames here are consistent across all three scripts.


import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torchvision
import argparse

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import MSELoss
import torch.optim as optim

from nets.Alexnetencoder import Encoder
from nets.Alexnetdecoder import Decoder
from torchvision.datasets import CIFAR10


parser = argparse.ArgumentParser()
parser.add_argument(
    "--classes",
    type=str,
    default="dog,cat,bird",
    help="Comma-separated CIFAR-10 class names the attacker's auxiliary "
         "dataset is biased toward, e.g. 'dog' (1 class), 'dog,cat' "
         "(2 classes), 'dog,cat,bird' (3 classes). The paper does not "
         "specify exact class choices for the 1/2/3-class cases, only "
         "the count — any fixed choice is a valid, reproducible pick."
)
args = parser.parse_args()

class_names = [c.strip() for c in args.classes.split(",")]
num_classes = len(class_names)
tag = f"targeted_{num_classes}class"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print("classes:", class_names, "-> tag:", tag)


sample_num = 15000
batch_size = 50
transform = transforms.Compose([transforms.ToTensor(), ])

trainset = CIFAR10(root='./train', train=True, download=True)
testset = CIFAR10(root='./test', train=False, download=True)

classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
             'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

x_train = trainset.data
x_test = testset.data
y_train = trainset.targets
y_test = testset.targets


def get_class_i(x, y, i):
    y = np.array(y)
    pos_i = np.argwhere(y == i)
    pos_i = list(pos_i[:, 0])
    x_i = [x[j] for j in pos_i]
    return x_i


class DatasetMaker(Dataset):
    def __init__(self, datasets, transformFunc=transform):
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc

    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        accum = np.add.accumulate(bin_sizes)
        bin_index = len(np.argwhere(accum <= absolute_index))
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        return bin_index, index_wrt_class


biased_trainset = DatasetMaker(
    [get_class_i(x_train, y_train, classDict[c]) for c in class_names],
    transform
)

train_loader = DataLoader(biased_trainset, batch_size=batch_size, shuffle=True)

print("biased train set size:", len(biased_trainset))


encoder = Encoder().to(device)
decoder = Decoder().to(device)

params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

lr = 0.001
optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

loss_fn = MSELoss()
all_epoch = 250
train_loss = []

for current_epoch in range(all_epoch):
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
    print(tag, "epoch:", current_epoch, "loss:", epoch_loss)

torch.save(encoder.state_dict(), f"models/encoder_model_{tag}.pkl")
torch.save(decoder.state_dict(), f"models/decoder_model_{tag}.pkl")

print(f"saved models/encoder_model_{tag}.pkl and models/decoder_model_{tag}.pkl")

encoder.eval()
decoder.eval()

plt.figure()
plt.plot(range(all_epoch), train_loss)
plt.savefig(f"figs/loss_trend_{tag}")
plt.show()