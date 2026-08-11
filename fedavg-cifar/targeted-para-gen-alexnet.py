# Table VII — AlexNet version, LSR distribution / bin estimation.
# Must be run with the SAME --classes argument as
# targeted-adv-train-alexnet.py, since the bins are only valid for the
# encoder trained on that exact class subset.
#
# FIX vs. original targeted-para-gen-alexnet.py: AlexNet architecture (k=512)
# instead of CNN (k=1024); class selection parameterized; consistent
# filenames (no "_new" suffix mismatch).


import torch
import torchvision
import argparse
import numpy as np

from nets.Alexnetencoder import Encoder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from scipy.stats import norm
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10


parser = argparse.ArgumentParser()
parser.add_argument("--classes", type=str, default="dog,cat,bird")
args = parser.parse_args()

class_names = [c.strip() for c in args.classes.split(",")]
num_classes = len(class_names)
tag = f"targeted_{num_classes}class"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
print("classes:", class_names, "-> tag:", tag)


sample_num = 15000
batch_size = 100
transform = transforms.Compose([transforms.ToTensor(), ])

trainset = CIFAR10(root='./train', train=True, download=True)
classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
             'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

x_train = trainset.data
y_train = trainset.targets


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


trained_model = Encoder()
trained_model.load_state_dict(torch.load(f"models/encoder_model_{tag}.pkl", map_location=device))
trained_model.eval()
trained_model.to(device)


latent_small = torch.zeros(sample_num)
current_index = 0

with torch.no_grad():

    for idx, (train_x, train_label) in enumerate(train_loader):

        if idx >= int(sample_num / batch_size):
            break

        current_bs = train_x.size(0)

        train_x = train_x.to(device)

        output = trained_model(train_x)

        latent_batch = torch.mean(output.view(current_bs, -1), dim=1).cpu()

        latent_small[current_index:current_index + current_bs] = latent_batch

        current_index += current_bs

        if current_index >= sample_num:
            break


print("collected samples:", current_index)

latent_small = latent_small[:current_index]

norm_param_small = norm.fit(latent_small.numpy())
rv_norm_small = norm(norm_param_small[0], norm_param_small[1])
small_bins = rv_norm_small.ppf(np.linspace(0.001, 0.999, 512))

torch.save(small_bins, f"data/latent_small_bins_{tag}.pt")
print(f"saved data/latent_small_bins_{tag}.pt")

x = np.linspace(latent_small.min().item(), latent_small.max().item(), 300)

plt.figure(figsize=(5.9, 5.3))
plt.hist(latent_small.numpy(), bins=48, density=True, label="LSR Dist", color='g')
plt.plot(x, rv_norm_small.pdf(x), label="Estimated Dist", color='r')
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("LSR Brightness", fontsize=16)
plt.ylabel("Density", fontsize=16)
plt.savefig(f"figs/Stats-{tag}.pdf")
plt.show()