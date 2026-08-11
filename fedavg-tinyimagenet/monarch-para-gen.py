# Generate bins using monarch-finetuned encoder on monarch images
# Step 2 for Table VIII (data skew) reproduction

import torch
from nets.CNNencoder import Encoder
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from datasets import load_dataset
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enc_pkl", type=str, default="models/encoder_model_monarch.pkl",
                        help="Path to encoder checkpoint")
    parser.add_argument("--out_pt", type=str, default="data/latent_small_bins_monarch.pt",
                        help="Output path for bins")
    args = parser.parse_args()
    return args

args = get_args()
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
batch_size = 100

toTensorTransform = transforms.Compose([transforms.ToTensor()])
tiny_imagenet_train = load_dataset('Maysee/tiny-imagenet', split='train')

class CustomDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        image = example['image']
        label = example['label']
        if self.transform:
            image = self.transform(image)
        if image.shape[0] == 1:
            return None
        else:
            return image, label

custom_train_dataset = CustomDataset(tiny_imagenet_train, transform=toTensorTransform)
custom_train_dataset = [item for item in custom_train_dataset if item is not None]
monarch_dataset = [(img, lbl) for img, lbl in custom_train_dataset if lbl == 39]
print(f"Monarch butterfly images: {len(monarch_dataset)}")

train_loader = DataLoader(monarch_dataset, batch_size=batch_size, shuffle=True)

trained_model = Encoder()
trained_model.load_state_dict(torch.load(args.enc_pkl, map_location=device))
trained_model.eval()
trained_model.to(device)
trained_model.relu3.register_forward_hook(get_activation('relu3'))

# 500 monarch images = 5 batches of 100
latent_small = torch.zeros(len(monarch_dataset))

for idx, (train_x, train_label) in enumerate(train_loader):
    train_x = train_x.to(device)
    output = trained_model(train_x)
    current_bs = train_x.shape[0]
    latent_batch = torch.mean(activation['relu3'].view(current_bs, -1), dim=1).cpu()
    latent_small[idx * batch_size:idx * batch_size + current_bs] = latent_batch

# Estimate normal distribution and obtain critical h vector for linear leakage
norm_param_small = norm.fit(latent_small.numpy())
rv_norm_small = norm(norm_param_small[0], norm_param_small[1])
small_bins = rv_norm_small.ppf(np.linspace(0.001, 0.999, 2048))

torch.save(small_bins, args.out_pt)
print(f"Saved {args.out_pt}")
print(f"Bins range: [{small_bins[0]:.4f}, {small_bins[-1]:.4f}]")
print(f"Normal fit: mu={norm_param_small[0]:.4f}, sigma={norm_param_small[1]:.4f}")

# Plot LSR distribution
x = np.linspace(0.2, 1.7, 300)
plt.figure(figsize=(5.9, 5.3))
plt.hist(latent_small.detach().cpu().numpy(), bins=24, density=True, label="LSR Dist", color='g')
plt.plot(x, rv_norm_small.pdf(x), label="Estimated Dist", color='r')
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("LSR Brightness", fontsize=16)
plt.ylabel("Density", fontsize=16)
plt.title("Monarch LSR Distribution (Fine-tuned Encoder)")
plt.savefig("figs/Stats-Monarch.pdf")
plt.show()
