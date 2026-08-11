# Table VII — AlexNet version, attack evaluation.
#
# FIX vs. original targeted-recover-attack-alexnet.py:
#   - Architecture: AlexNet (k=512, features=256, map_size=4) instead
#     of CNN (k=1024), matching the paper's stated setup and confirmed
#     by the fact that Table VII's "10 classes" column is numerically
#     identical to Table VI's "100%" column (which explicitly uses
#     AlexNet).
#   - Filename bug fixed: this now loads the SAME "{tag}" files that
#     targeted-adv-train-alexnet.py / targeted-para-gen-alexnet.py
#     actually save, instead of silently loading stale pre-existing
#     files with a different name.
#   - test_rounds cap fixed: the original hardcoded
#     `min(args.test_rounds, int(10000/batch_size))`, assuming a
#     10,000-image test set. With a class-restricted subset (as few as
#     ~1,000 test images for 1 class), this could request more rounds
#     than the dataset actually supports. Now computed from the actual
#     filtered dataset size.
#   - map_location=device added to torch.load calls (avoids the
#     "CUDA device 1 not found" error seen with some of the original
#     artifact's pretrained files on single-GPU machines).
#   - Evaluation is restricted to the SAME class subset the attacker's
#     auxiliary set is biased toward, matching the paper's description
#     ("we would only evaluate Scale-MIA's performance on reconstructing
#     the 'dog' samples ... ignoring samples from other classes").


import numpy as np
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import SGD
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import time

from nets.Alexnetencoder import Encoder
from nets.Alexnetdecoder import Decoder
from nets.Alexnet import Alexnet
from torchvision.datasets import CIFAR10
import copy


bin = 512
features = 256
map_size = 4
l2 = 120


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
    original_data = torch.zeros(batch_size, features * map_size * map_size)
    for i in range(batch_size):
        index = int(locations[batch_size - i - 1])
        if index == bin - 1:
            original_data[i, :] = fcweightgrad[index, :] / fcbiasgrad[index, :]
        else:
            original_data[i, :] = (fcweightgrad[index, :] - fcweightgrad[index + 1, :]) / (
                fcbiasgrad[index, :] - fcbiasgrad[index + 1, :] + 1e-8
            )
    return original_data


def attack(weight, bias, original_data, l1output, batch_size, idx):
    locations = np.zeros(batch_size)
    for i in range(batch_size):
        locations[i] = bin - 1 - (l1output[i, :] < 0).count_nonzero()
    sorted_list = np.argsort(locations)
    sorted_location = np.sort(locations)
    recovered_latent = input_retrival(weight, bias, sorted_location, batch_size)
    recovered_data = decoder(recovered_latent.to(device).view(batch_size, features, map_size, map_size))
    round_num, round_mse_score, round_PSNR_score = PSNR_cal(recovered_data, original_data, locations, batch_size)

    if idx == test_rounds - 1:
        torch.save(original_data, f"data/data_batch_{tag}.pt")
        torch.save(recovered_data, f"data/recovered_{tag}.pt")
        torch.save(sorted_list, f"data/list_{tag}.pt")

    return round_num, round_mse_score, round_PSNR_score


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_rounds", type=int, default=10)
    parser.add_argument("--client_num", type=int, default=8)
    parser.add_argument("--local_epoch", type=int, default=1)
    parser.add_argument("--all_epoch", type=int, default=1)
    parser.add_argument("--classes", type=str, default="dog,cat,bird")
    return parser.parse_args()


if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    args = get_args()

    class_names = [c.strip() for c in args.classes.split(",")]
    num_classes = len(class_names)
    tag = f"targeted_{num_classes}class"

    print(device)
    print("classes:", class_names, "-> tag:", tag)

    batch_size = args.batch_size
    client_num = args.client_num
    client_size = int(batch_size / client_num)
    all_epoch = args.all_epoch
    local_epoch = args.local_epoch

    transform = transforms.Compose([transforms.ToTensor(), ])

    testset = CIFAR10(root='./test', train=False, download=True)
    classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    x_test = testset.data
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

    biased_testset = DatasetMaker(
        [get_class_i(x_test, y_test, classDict[c]) for c in class_names],
        transform
    )

    print("biased test set size:", len(biased_testset))

    test_loader = DataLoader(biased_testset, batch_size=batch_size, shuffle=True)

    # FIX: was hardcoded to assume a 10,000-image test set
    # (min(args.test_rounds, int(10000/batch_size))). Now uses the
    # actual filtered dataset size.
    test_rounds = min(args.test_rounds, len(biased_testset) // batch_size)

    if test_rounds < 1:
        raise ValueError(
            f"biased test set only has {len(biased_testset)} samples, "
            f"too small for batch_size={batch_size}. Use a smaller batch_size."
        )

    encoder = Encoder()
    encoder.load_state_dict(torch.load(f"models/encoder_model_{tag}.pkl", map_location=device))
    encoder.to(device)

    decoder = Decoder()
    decoder.load_state_dict(torch.load(f"models/decoder_model_{tag}.pkl", map_location=device))
    decoder.to(device)

    malicious_weight = 1 / (features * map_size * map_size) * torch.ones(bin, features * map_size * map_size).to(device).float()
    malicious_layer2_weight = 1 / bin * torch.ones(l2, bin).to(device).float()
    malicious_layer2_bias = 1 / l2 * torch.ones(l2).to(device).float()
    malicious_layer3_weight = torch.rand(10, l2).to(device).float()
    malicious_layer3_bias = torch.rand(10).to(device).float()

    bins = torch.load(f"data/latent_small_bins_{tag}.pt")
    if isinstance(bins, np.ndarray):
        bins_t = torch.from_numpy(bins)
    else:
        bins_t = bins
    malicious_bias = -bins_t.to(device).float()

    number = 0
    mse_score = torch.zeros(test_rounds * all_epoch)
    PSNR_score = torch.zeros(test_rounds * all_epoch)

    start_time = time.time()

    for current_epoch in range(all_epoch):
        for idx, (train_x, train_label) in enumerate(test_loader):

            if idx >= test_rounds:
                break

            aggregated_weight = torch.zeros(bin, features * map_size * map_size)
            aggregated_bias = torch.zeros(bin, features * map_size * map_size)
            gradient_weight = None
            l1output = None

            for j in range(client_num):

                model = Alexnet().to(device)
                model.features = copy.deepcopy(encoder.features)

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
                    gradient_bias = model.fc1.bias.grad.data.view(bin, 1).expand(bin, features * map_size * map_size)

                for k in range(local_epoch):
                    train_x_local = train_x[j * client_size: (j + 1) * client_size, :, :, :].to(device)
                    train_label_local = train_label[j * client_size: (j + 1) * client_size].to(device)

                    sgd.zero_grad()
                    predict_y_local = model(train_x_local.float())
                    loss = loss_fn(predict_y_local, train_label_local.long())
                    loss.backward()
                    sgd.step()

                aggregated_weight = aggregated_weight + model.fc1.weight.data.cpu() / client_num
                aggregated_bias = aggregated_bias + model.fc1.bias.data.cpu().view(bin, 1).expand(bin, features * map_size * map_size) / client_num
                torch.cuda.empty_cache()

            estimated_weight_gradient = -(aggregated_weight - 1 / (features * map_size * map_size) * torch.ones(bin, features * map_size * map_size).float()) / (local_epoch * 0.1)
            estimated_bias_gradient = -(aggregated_bias + bins_t.float().view(bin, 1).expand(bin, features * map_size * map_size)) / (local_epoch * 0.1)

            original_data = train_x.to(device)

            round_num, round_mse_score, round_PSNR_score = attack(
                estimated_weight_gradient.to(device),
                estimated_bias_gradient.to(device),
                original_data, l1output, batch_size, idx
            )

            number = number + round_num.item()
            mse_score[current_epoch * test_rounds + idx] = round_mse_score.item()
            PSNR_score[current_epoch * test_rounds + idx] = round_PSNR_score.item()

            if idx == test_rounds - 1:
                end_time = time.time()
                break

    print("number", number / (test_rounds * batch_size * all_epoch))
    print("mse score", torch.nansum(mse_score) / torch.nansum(mse_score < 0.03))
    print("PSNR score", torch.nansum(PSNR_score) / torch.nansum(PSNR_score > 1))
    print("Attack time", (end_time - start_time) / (test_rounds * all_epoch))