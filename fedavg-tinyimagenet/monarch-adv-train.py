# Fine-tune autoencoder on monarch butterfly only
# Starts from full-pretrained weights, specializes decoder for butterflies
# 2000 epochs + data augmentation to compensate for small dataset (500 images)

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn import MSELoss
from nets.CNNencoder import Encoder
from nets.CNNdecoder import Decoder
from datasets import load_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ── Load train set, filter monarch (label 39) ──
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

# Data augmentation for training (only applied during training, not validation)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(64, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

class AugmentedDataset(Dataset):
    def __init__(self, images, augment=True):
        self.images = images  # list of (tensor, label)
        self.augment = augment
        self.aug_tf = train_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, lbl = self.images[idx]
        if self.augment:
            # ToPILImage then apply augmentations then back to tensor
            from torchvision.transforms import ToPILImage
            pil_img = ToPILImage()(img)
            aug_img = self.aug_tf(pil_img)
            return transforms.ToTensor()(aug_img), lbl
        return img, lbl

batch_size = 32
train_dataset = AugmentedDataset(monarch_dataset, augment=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ── Load pre-trained full autoencoder as starting point ──
encoder = Encoder().to(device)
encoder.load_state_dict(torch.load("models/encoder_model_10_new_2.pkl", map_location=device))
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load("models/decoder_model_10_new_2.pkl", map_location=device))

params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

lr = 0.001
optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-5)
loss_fn = MSELoss()

all_epoch = 2000
train_loss = []

for current_epoch in range(all_epoch):
    encoder.train()
    decoder.train()
    epoch_loss = 0
    batch_count = 0
    for idx, (train_x, train_label) in enumerate(train_loader):
        train_x = train_x.to(device)
        encoded_data = encoder(train_x)
        decoded_data = decoder(encoded_data)
        loss = loss_fn(decoded_data, train_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        batch_count += 1
    scheduler.step()
    epoch_loss = epoch_loss / max(batch_count, 1)
    train_loss.append(epoch_loss)
    if current_epoch % 100 == 0:
        print(f"Epoch {current_epoch:4d}/{all_epoch}  loss={epoch_loss:.6f}  lr={scheduler.get_last_lr()[0]:.6f}")

torch.save(encoder.state_dict(), "models/encoder_model_monarch.pkl")
torch.save(decoder.state_dict(), "models/decoder_model_monarch.pkl")
print("Saved models/encoder_model_monarch.pkl  &  models/decoder_model_monarch.pkl")

plt.figure()
plt.plot(range(all_epoch), train_loss)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Monarch Autoencoder Fine-tuning Loss")
plt.savefig("figs/loss_trend_monarch.png")
plt.show()
