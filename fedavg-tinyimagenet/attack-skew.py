# Data skew attack for Table VIII reproduction
#
# Usage (monarch-finetuned autoencoder, default):
#   Intra-class:  python attack-skew.py --skew intra  --batch_size 16
#   Inter-class:  python attack-skew.py --skew inter  --batch_size 16
#
# Usage (full autoencoder for comparison):
#   python attack-skew.py --skew intra --batch_size 16 \
#       --enc_pkl models/encoder_model_10_new_2.pkl \
#       --dec_pkl models/decoder_model_10_new_2.pkl \
#       --bins_pt data/latent_small_bins_monarch.pt

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import SGD
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import time
from nets.CNNencoder import Encoder
from nets.CNNdecoder import Decoder
from nets.CNNclassifier import Classifier
import argparse

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

def PSNR_cal(recovered_data, original_data, locations, batch_size):
    mse_fn = MSELoss()
    sorted_index = np.argsort(locations)
    score = torch.zeros(batch_size)
    PSNR = torch.zeros(batch_size)
    for i in range(batch_size):
        score[i] = mse_fn(original_data[sorted_index[batch_size - 1 - i], :], recovered_data[i, :])
        PSNR[i] = -10 * torch.log10(score[i])
    recover_number = torch.nansum(PSNR > 18)
    avg_score = torch.nansum(score[torch.lt(score, 0.03)]) / torch.nansum(score < 0.03)
    avg_PSNR = torch.nansum(PSNR[torch.gt(PSNR, 18)]) / torch.nansum(PSNR > 18)
    return recover_number, avg_score, avg_PSNR

def input_retrival(fcweightgrad, fcbiasgrad, locations, batch_size):
    original_data = torch.zeros(batch_size, 128 * 4 * 4)
    for i in range(batch_size):
        if i == 0:
            index = int(locations[batch_size - i - 1])
            original_data[i, :] = fcweightgrad[index, :] / fcbiasgrad[index, :]
        else:
            index = int(locations[batch_size - i - 1])
            if index == 2048 - 1:
                original_data[i, :] = fcweightgrad[index, :] / fcbiasgrad[index, :]
            else:
                original_data[i, :] = (fcweightgrad[index, :] - fcweightgrad[index + 1, :]) / \
                                      (fcbiasgrad[index, :] - fcbiasgrad[index + 1, :] + 1e-8)
    return original_data

def attack(weight, bias, original_data, l1output, batch_size, idx, decoder, device):
    locations = np.zeros(batch_size)
    for i in range(batch_size):
        locations[i] = 2047 - (l1output[i, :] < 0).count_nonzero()
    sorted_list = np.argsort(locations)
    sorted_location = np.sort(locations)
    recovered_latent = input_retrival(weight, bias, sorted_location, batch_size)
    recovered_data = decoder(recovered_latent.to(device).view(batch_size, 128, 4, 4))
    round_num, round_mse_score, round_PSNR_score = PSNR_cal(recovered_data, original_data, locations, batch_size)
    return round_num, round_mse_score, round_PSNR_score

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--skew", type=str, default="intra", choices=["intra", "inter", "same"])
    parser.add_argument("--test_rounds", type=int, default=100,
                        help="Max test rounds (capped by available images)")
    parser.add_argument("--client_num", type=int, default=8)
    parser.add_argument("--local_epoch", type=int, default=1)
    parser.add_argument("--all_epoch", type=int, default=1)
    parser.add_argument("--enc_pkl", type=str, default="models/encoder_model_monarch.pkl")
    parser.add_argument("--dec_pkl", type=str, default="models/decoder_model_monarch.pkl")
    parser.add_argument("--bins_pt", type=str, default="data/latent_small_bins_monarch.pt")
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="Shuffle test data (default: True)")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    args = get_args()

    batch_size = args.batch_size
    client_num = args.client_num
    client_size = int(batch_size / client_num)
    all_epoch = args.all_epoch
    local_epoch = args.local_epoch

    # ── Load test data: filter by target label ──
    if args.skew == "same":
        target_label = 39
        skew_name = "monarch butterfly"
    elif args.skew == "intra":
        target_label = 40
        skew_name = "sulfur butterfly"
    else:
        target_label = 2
        skew_name = "bullfrog"

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
    target_dataset = [(img, lbl) for img, lbl in custom_train_dataset if lbl == target_label]

    total_test_images = len(target_dataset)
    test_rounds = min(args.test_rounds, int(total_test_images / batch_size))
    print(f"Skew: {args.skew} | Target: {skew_name} | "
          f"Test images: {total_test_images} | Batch: {batch_size} | Rounds: {test_rounds} | "
          f"Shuffle: {args.shuffle}")

    # Use shuffle for representative sampling
    test_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=args.shuffle)

    # ── Load autoencoder ──
    encoder = Encoder()
    encoder.load_state_dict(torch.load(args.enc_pkl, map_location="cpu"))
    encoder.to(device)

    decoder = Decoder()
    decoder.load_state_dict(torch.load(args.dec_pkl, map_location="cpu"))
    decoder.to(device)

    print(f"Encoder : {args.enc_pkl}")
    print(f"Decoder : {args.dec_pkl}")
    print(f"Bins    : {args.bins_pt}")

    malicious_weight = 1 / (128 * 4 * 4) * torch.ones(2048, 128 * 4 * 4).to(device).float()
    malicious_layer2_weight = 1 / 2048 * torch.ones(1024, 2048).to(device).float()
    malicious_layer2_bias = 1 / 1024 * torch.ones(1024).to(device).float()
    malicious_layer3_weight = torch.rand(200, 1024).to(device).float()
    malicious_layer3_bias = torch.rand(200).to(device).float()
    bins = torch.load(args.bins_pt)
    malicious_bias = -torch.from_numpy(bins).to(device).float()

    number = 0
    mse_score = torch.zeros(test_rounds * all_epoch)
    PSNR_score = torch.zeros(test_rounds * all_epoch)
    torch.set_printoptions(precision=8)

    start_time = time.time()
    for current_epoch in range(all_epoch):
        for idx, (train_x, train_label) in enumerate(test_loader):
            aggregated_weight = torch.zeros(2048, 128 * 16)
            aggregated_bias = torch.zeros(2048, 128 * 16)
            gradient_weight = torch.zeros(2048, 128 * 16).to(device)
            gradient_bias = torch.zeros(2048, 128 * 16).to(device)

            for j in range(client_num):
                model = Classifier().to(device)
                model.conv1.weight.data = encoder.conv1.weight.data.to(device).clone().detach()
                model.conv2.weight.data = encoder.conv2.weight.data.to(device).clone().detach()
                model.conv3.weight.data = encoder.conv3.weight.data.to(device).clone().detach()
                model.conv4.weight.data = encoder.conv4.weight.data.to(device).clone().detach()
                model.conv1.bias.data = encoder.conv1.bias.data.to(device).clone().detach()
                model.conv2.bias.data = encoder.conv2.bias.data.to(device).clone().detach()
                model.conv3.bias.data = encoder.conv3.bias.data.to(device).clone().detach()
                model.conv4.bias.data = encoder.conv4.bias.data.to(device).clone().detach()
                model.fc1.weight.data = malicious_weight.clone().detach()
                model.fc1.bias.data = malicious_bias.clone().detach()
                model.fc2.weight.data = malicious_layer2_weight.clone().detach()
                model.fc2.bias.data = malicious_layer2_bias.clone().detach()
                model.fc3.weight.data = malicious_layer3_weight.clone().detach()
                model.fc3.bias.data = malicious_layer3_bias.clone().detach()
                model.train()
                sgd = SGD(model.parameters(), lr=0.1)
                loss_fn = CrossEntropyLoss()

                if j == 0:
                    handle = model.fc1.register_forward_hook(get_activation('fc1'))
                    predict_y = model(train_x.float().to(device))
                    l1output = activation['fc1'].view(batch_size, -1)
                    handle.remove()
                    sgd.zero_grad()
                    loss = loss_fn(predict_y, train_label.long().to(device))
                    loss.backward()
                    gradient_weight = model.fc1.weight.grad.data
                    gradient_bias = model.fc1.bias.grad.data.view(2048, 1).expand(2048, 128 * 16)

                for k in range(local_epoch):
                    train_x_local = train_x[j * client_size:(j + 1) * client_size, :, :, :].to(device)
                    train_label_local = train_label[j * client_size:(j + 1) * client_size].to(device)
                    sgd.zero_grad()
                    predict_y_local = model(train_x_local.float())
                    loss = loss_fn(predict_y_local, train_label_local.long())
                    loss.backward()
                    sgd.step()

                aggregated_weight = aggregated_weight + model.fc1.weight.data.cpu() / client_num
                aggregated_bias = aggregated_bias + \
                    model.fc1.bias.data.cpu().view(2048, 1).expand(2048, 128 * 16) / client_num
                torch.cuda.empty_cache()

            estimated_weight_gradient = -(aggregated_weight - 1 / (128 * 4 * 4) *
                                          torch.ones(2048, 128 * 4 * 4).float()) / (local_epoch * 0.1)
            estimated_bias_gradient = -(aggregated_bias +
                                        torch.from_numpy(bins).float().view(2048, 1).expand(2048, 128 * 16)) / (local_epoch * 0.1)

            original_data = train_x.to(device)
            round_num, round_mse_score, round_PSNR_score = attack(
                estimated_weight_gradient.to(device), estimated_bias_gradient.to(device),
                original_data, l1output, batch_size, idx, decoder, device)

            number = number + round_num.item()
            mse_score[current_epoch * test_rounds + idx] = round_mse_score.item()
            PSNR_score[current_epoch * test_rounds + idx] = round_PSNR_score.item()

            if idx == test_rounds - 1:
                break

    end_time = time.time()

    total_samples = test_rounds * batch_size * all_epoch
    recovery_rate = number / total_samples
    avg_mse = torch.nansum(mse_score) / torch.nansum(mse_score < 0.03)
    avg_PSNR = torch.nansum(PSNR_score) / torch.nansum(PSNR_score > 1)

    print(f"\n{'='*50}")
    print(f"TABLE VIII | skew={args.skew} | batch={batch_size}")
    print(f"  Recovery Rate: {recovery_rate:.4f}")
    print(f"  Avg MSE:       {avg_mse.item():.6f}")
    print(f"  Avg PSNR:      {avg_PSNR.item():.4f}")
    print(f"  Time/round:    {(end_time - start_time) / max(all_epoch * test_rounds, 1):.2f}s")
    print(f"{'='*50}")
