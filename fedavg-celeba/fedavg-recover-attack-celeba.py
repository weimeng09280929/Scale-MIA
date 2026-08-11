# CelebA version of fedavg recover attack.
#
# Written from scratch (CelebA is not part of the paper's public
# artifact) — reuses the exact attack logic already validated on
# ImageNette (nets/CNNencoder.py, CNNdecoder.py, CNNclassifier.py,
# 160x160 resolution, fc1 = Linear(128*10*10, 2048)).
#
# Eval split is CelebA's own "valid" split (torchvision CelebA supports
# split in {"train","valid","test","all"}), disjoint from the "train"
# split used by fedavg-adv-train-celeba.py / fedavg-para-gen-celeba.py
# as the auxiliary dataset D_Adv — same disjointness fix applied to the
# ImageNette pipeline earlier.
#
# Pseudo-label: attribute index 20 ("Male") is used purely to drive a
# nonzero CrossEntropy gradient through the malicious classifier head;
# it has no bearing on reconstruction quality (see fedavg-adv-train-
# celeba.py for the same note).


import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import SGD
import time
import argparse
import os


from nets.CNNencoder import Encoder
from nets.CNNdecoder import Decoder
from nets.CNNclassifier import Classifier



# =========================
# activation hook
# =========================

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook



# =========================
# retrieve latent
# =========================

def input_retrival(
        fcweightgrad,
        fcbiasgrad,
        locations,
        batch_size
):

    original_data = torch.zeros(
        batch_size,
        128 * 10 * 10
    )


    for i in range(batch_size):

        index = int(locations[batch_size-i-1])

        if index == 2047:

            original_data[i,:] = (
                fcweightgrad[index,:] /
                fcbiasgrad[index,:]
            )

        else:

            original_data[i,:] = (
                fcweightgrad[index,:]
                -
                fcweightgrad[index+1,:]
            ) / (
                fcbiasgrad[index,:]
                -
                fcbiasgrad[index+1,:]
                +
                1e-8
            )


    return original_data



# =========================
# PSNR
# =========================

def PSNR_cal(
        recovered_data,
        original_data,
        locations,
        batch_size
):

    mse_fn=MSELoss()


    sorted_index=np.argsort(locations)


    score=torch.zeros(batch_size)

    PSNR=torch.zeros(batch_size)


    for i in range(batch_size):

        score[i]=mse_fn(
            original_data[
                sorted_index[batch_size-1-i],
            ],
            recovered_data[i]
        )

        PSNR[i]=-10*torch.log10(score[i])


    recover_number=torch.nansum(
        PSNR>18
    )


    avg_score=(
        torch.nansum(
            score[score<0.03]
        )
        /
        torch.nansum(score<0.03)
    )


    avg_PSNR=(
        torch.nansum(
            PSNR[PSNR>18]
        )
        /
        torch.nansum(PSNR>18)
    )


    return (
        recover_number,
        avg_score,
        avg_PSNR
    )



# =========================
# attack
# =========================


def attack(
        weight,
        bias,
        original_data,
        l1output,
        batch_size,
        idx
):


    locations=np.zeros(batch_size)


    for i in range(batch_size):

        locations[i]=2047-(
            l1output[i,:]
            <0
        ).count_nonzero()



    sorted_location=np.sort(locations)

    sorted_list = np.argsort(locations)

    unique_locs = len(np.unique(locations))
    print(
        f"  [debug] batch_size={batch_size}, unique_bins_hit={unique_locs}, "
        f"collision_rate={1 - unique_locs / batch_size:.3f}"
    )


    recovered_latent=input_retrival(
        weight,
        bias,
        sorted_location,
        batch_size
    )


    recovered_data=decoder(
        recovered_latent
        .to(device)
        .view(
            batch_size,
            128,
            10,
            10
        )
    )


    round_num, mse, psnr = PSNR_cal(
        recovered_data,
        original_data,
        locations,
        batch_size
    )



    if idx==test_rounds-1:

        os.makedirs("data", exist_ok=True)

        torch.save(original_data.cpu(), "data/data_batch_celeba.pt")

        torch.save(recovered_data.cpu(), "data/recovered_celeba.pt")

        torch.save(sorted_list, "data/list_celeba.pt")


    return (
        round_num,
        mse,
        psnr
    )



# =========================
# args
# =========================


def get_args():

    parser=argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--test_rounds", type=int, default=10)

    parser.add_argument("--client_num", type=int, default=8)

    parser.add_argument("--local_epoch", type=int, default=1)

    parser.add_argument("--all_epoch", type=int, default=1)

    # CelebA's own "valid" split — disjoint from "train", which
    # fedavg-adv-train-celeba.py / fedavg-para-gen-celeba.py use as
    # the auxiliary dataset D_Adv.
    parser.add_argument("--eval_split", type=str, default="valid")

    return parser.parse_args()





# =========================
# main
# =========================


if __name__=="__main__":


    device=(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(device)



    args=get_args()


    batch_size=args.batch_size

    client_num=args.client_num

    client_size=batch_size//client_num


    local_epoch=args.local_epoch

    all_epoch=args.all_epoch



    # =========================
    # CelebA dataset (eval split)
    # =========================

    transform=transforms.Compose([

        transforms.CenterCrop(178),

        transforms.Resize((160,160)),

        transforms.ToTensor()

    ])



    dataset=CelebA(

        root="./celeba",

        split=args.eval_split,

        target_type="attr",

        transform=transform,

        target_transform=lambda attr: attr[20].long(),

        download=False

    )


    print("dataset size:", len(dataset))


    test_loader=DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=True

    )



    test_rounds=min(
        args.test_rounds,
        len(dataset)//batch_size
    )



    # =========================
    # load encoder decoder
    # =========================


    encoder=Encoder()


    encoder.load_state_dict(
        torch.load(
            "models/encoder_model_celeba.pkl",
            map_location="cpu"
        )
    )


    encoder.to(device)



    decoder=Decoder()


    decoder.load_state_dict(
        torch.load(
            "models/decoder_model_celeba.pkl",
            map_location="cpu"
        )
    )


    decoder.to(device)



    encoder.eval()
    decoder.eval()



    # =========================
    # bins
    # =========================

    bins = torch.load(
        "data/latent_small_bins_celeba.pt",
        map_location="cpu"
    )

    if isinstance(bins, np.ndarray):
        bins = torch.from_numpy(bins)

    bins = bins.float()



    # =========================
    # malicious model
    # =========================


    malicious_weight=(
        1/(128*10*10)
        *
        torch.ones(2048, 128*10*10)
    ).to(device).float()



    malicious_layer2_weight=(
        1/2048
        *
        torch.ones(1024, 2048)
    ).to(device).float()



    malicious_layer2_bias=(
        1/1024
        *
        torch.ones(1024)
    ).to(device).float()


    # output dim (200) is arbitrary — doesn't need to match the real
    # number of pseudo-label classes (2, for the "Male" 0/1 attribute).
    # Same pattern as the ImageNette script (200-way head for a
    # 10-class dataset); the attack's reconstruction quality only
    # depends on fc1's ReLU activation pattern, not on fc3's output
    # dimension or the label semantics.
    malicious_layer3_weight=torch.rand(200, 1024).to(device).float()

    malicious_layer3_bias=torch.rand(200).to(device).float()


    malicious_bias=(-bins).to(device).float()



    number=0

    mse_score=torch.zeros(test_rounds*all_epoch)

    PSNR_score=torch.zeros(test_rounds*all_epoch)



    start_time=time.time()



    for epoch in range(all_epoch):


        for idx,(train_x,train_label) in enumerate(test_loader):


            if idx>=test_rounds:
                break



            aggregated_weight=torch.zeros(2048, 128*10*10)

            aggregated_bias=torch.zeros(2048, 128*10*10)


            gradient_weight=None

            l1output=None



            for j in range(client_num):


                model=Classifier().to(device)



                model.conv1.weight.data=encoder.conv1.weight.data.clone()

                model.conv2.weight.data=encoder.conv2.weight.data.clone()

                model.conv3.weight.data=encoder.conv3.weight.data.clone()

                model.conv4.weight.data=encoder.conv4.weight.data.clone()


                model.conv1.bias.data=encoder.conv1.bias.data.clone()

                model.conv2.bias.data=encoder.conv2.bias.data.clone()

                model.conv3.bias.data=encoder.conv3.bias.data.clone()

                model.conv4.bias.data=encoder.conv4.bias.data.clone()



                model.fc1.weight.data=malicious_weight.clone()

                model.fc1.bias.data=malicious_bias.clone()


                model.fc2.weight.data=malicious_layer2_weight.clone()

                model.fc2.bias.data=malicious_layer2_bias.clone()


                model.fc3.weight.data=malicious_layer3_weight.clone()

                model.fc3.bias.data=malicious_layer3_bias.clone()



                model.train()



                sgd=SGD(model.parameters(), lr=0.1)

                loss_fn=CrossEntropyLoss()



                if j==0:


                    handle=model.fc1.register_forward_hook(
                        get_activation("fc1")
                    )


                    predict_y=model(train_x.float().to(device))


                    l1output=activation["fc1"].view(batch_size, -1)


                    handle.remove()



                    loss=loss_fn(
                        predict_y,
                        train_label.long().to(device)
                    )


                    loss.backward()



                    gradient_weight=model.fc1.weight.grad.data

                    gradient_bias=model.fc1.bias.grad.data.view(
                        2048, 1
                    ).expand(2048, 128*10*10)



                for k in range(local_epoch):


                    x_local=train_x[
                        j*client_size:(j+1)*client_size
                    ].to(device)


                    y_local=train_label[
                        j*client_size:(j+1)*client_size
                    ].to(device)


                    sgd.zero_grad()


                    pred=model(x_local.float())


                    loss=loss_fn(pred, y_local.long())


                    loss.backward()

                    sgd.step()



                aggregated_weight += (
                    model.fc1.weight.data.cpu() / client_num
                )


                aggregated_bias += (
                    model.fc1.bias.data.cpu()
                    .view(2048,1)
                    .expand(2048, 128*10*10)
                    / client_num
                )



            estimated_weight_gradient=(
                -(aggregated_weight
                -
                1/(128*10*10)
                *
                torch.ones(2048, 128*10*10))
                /
                (local_epoch*0.1)
            )


            estimated_bias_gradient=(
                -(aggregated_bias
                +
                bins.view(2048,1).expand(2048, 128*10*10))
                /
                (local_epoch*0.1)
            )



            original_data=train_x.to(device)



            result=attack(

                estimated_weight_gradient.to(device),

                estimated_bias_gradient.to(device),

                original_data,

                l1output,

                batch_size,

                idx

            )


            number += result[0].item()


            mse_score[epoch*test_rounds+idx]=result[1].item()

            PSNR_score[epoch*test_rounds+idx]=result[2].item()



    end_time=time.time()



    print("number", number/(test_rounds*batch_size*all_epoch))

    print("mse score", torch.nansum(mse_score)/torch.nansum(mse_score<0.03))

    print("PSNR score", torch.nansum(PSNR_score)/torch.nansum(PSNR_score>1))

    print("attack time", (end_time-start_time)/(all_epoch*test_rounds))