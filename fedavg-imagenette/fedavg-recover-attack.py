# This is ImageNette version of fedavg recover attack
# Modified from TinyImageNet implementation
#
# FIX vs. original upload:
#   - test_loader now reads from a disjoint eval split (--eval_root,
#     e.g. ./dataset/val) instead of "./train", which was the SAME
#     folder used as the auxiliary dataset D_Adv for adv-train/para-gen.
#   - Resize stays at 160x160 (matches fc1 = Linear(128*10*10, 2048)
#     in nets/CNNclassifier.py). adv-train.py / para-gen.py must also
#     be run at 160x160, or the loaded encoder/decoder/bins are stale.
#
# DEBUG ADDITIONS (this version):
#   - collision_rate print inside attack().
#   - --debug_grad_diff: compares the *whole-matrix* mean error between
#     estimated_weight_gradient (FedAvg weight-delta reconstruction)
#     and the true fc1 gradient captured via autograd at j==0. NOTE:
#     this whole-matrix comparison is diluted by ~26M mostly-irrelevant
#     entries and is NOT a reliable proxy for reconstruction quality —
#     kept here only for reference / sanity-checking gross scale.
#   - --debug_latent_diff: the actually meaningful comparison. Runs
#     input_retrival() twice — once with the estimated gradient, once
#     with the true gradient — using the SAME bin locations, and prints
#     the MSE between the two recovered latent batches. This isolates
#     exactly the finite-difference step (Eq. 6 in the paper) where
#     FedAvg's small-minibatch noise gets amplified through subtraction
#     of near-equal adjacent-bin gradients.
#   - --use_true_grad: bypass the estimation entirely, feed the true
#     autograd gradient into attack() (fair A/B baseline; only a true
#     apples-to-apples upper bound when --client_num=1).


import numpy as np
import torch
import torchvision
from torchvision import transforms
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
        idx,
        true_weight=None,
        true_bias=None,
        debug_latent_diff=False
):


    locations=np.zeros(batch_size)


    for i in range(batch_size):

        locations[i]=2047-(
            l1output[i,:]
            <0
        ).count_nonzero()



    sorted_location=np.sort(locations)
    sorted_list = np.argsort(locations)

    # [DEBUG] collision diagnostics
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


    # [DEBUG] the actually meaningful comparison: reconstruct the latent
    # a second time using the TRUE gradient (same bin locations), and
    # compare directly against the latent reconstructed from the
    # estimated (FedAvg weight-delta) gradient. This isolates exactly
    # the finite-difference step that matters for reconstruction quality,
    # unlike a whole-matrix mean error which is diluted by irrelevant
    # entries.
    if debug_latent_diff and true_weight is not None and true_bias is not None:

        recovered_latent_true = input_retrival(
            true_weight,
            true_bias,
            sorted_location,
            batch_size
        )

        latent_diff = (recovered_latent - recovered_latent_true).abs()
        latent_mse = ((recovered_latent - recovered_latent_true) ** 2).mean().item()
        true_latent_scale = recovered_latent_true.abs().mean().item()

        print(
            f"  [debug] latent diff mean={latent_diff.mean().item():.6e}, "
            f"true latent |mean|={true_latent_scale:.6e}, "
            f"latent MSE={latent_mse:.6e}, "
            f"relative latent error={latent_diff.mean().item()/max(true_latent_scale,1e-12):.3f}"
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

        os.makedirs(
            "data",
            exist_ok=True
        )

        torch.save(
            original_data.cpu(),
            "data/data_batch.pt"
        )


        torch.save(
            recovered_data.cpu(),
            "data/recovered.pt"
        )
    torch.save(sorted_list, "data/list.pt")

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

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )


    parser.add_argument(
        "--test_rounds",
        type=int,
        default=10
    )


    parser.add_argument(
        "--client_num",
        type=int,
        default=8
    )


    parser.add_argument(
        "--local_epoch",
        type=int,
        default=1
    )


    parser.add_argument(
        "--all_epoch",
        type=int,
        default=1
    )


    # FIX: expose the eval-set root as a CLI arg so it's obvious this
    # must differ from the aux-set root used in adv-train.py/para-gen.py
    parser.add_argument(
        "--eval_root",
        type=str,
        default="./val"
    )


    # [DEBUG] whole-matrix mean grad diff (reference only, misleading —
    # see module docstring above)
    parser.add_argument(
        "--debug_grad_diff",
        action="store_true",
        help="Print whole-matrix mean ||estimated_grad - true_grad|| (reference only)."
    )


    # [DEBUG] the meaningful one: latent-level diff after input_retrival()
    parser.add_argument(
        "--debug_latent_diff",
        action="store_true",
        help="Print MSE between latents reconstructed from estimated vs. true gradient."
    )


    # [DEBUG] force the attack to use the TRUE gradient instead of the
    # weight-delta estimation, for A/B comparison. Only fair/meaningful
    # when --client_num=1.
    parser.add_argument(
        "--use_true_grad",
        action="store_true",
        help="Bypass the weight-delta estimation and feed the real autograd "
             "gradient (captured at client j==0) straight into attack(). "
             "Only meaningful/fair when --client_num=1."
    )


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
    # ImageNette dataset
    # =========================

    transform=transforms.Compose([

        transforms.Resize(
            (160,160)
        ),

        transforms.ToTensor()

    ])



    dataset=torchvision.datasets.ImageFolder(

        root=args.eval_root,

        transform=transform

    )


    print(
        "dataset size:",
        len(dataset)
    )


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
            "models/encoder_model_imagenette.pkl",
            map_location="cpu"
        )
    )


    encoder.to(device)



    decoder=Decoder()


    decoder.load_state_dict(
        torch.load(
            "models/decoder_model_imagenette.pkl",
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
        "data/latent_small_bins_imagenette.pt",
        map_location="cpu"
    )

    # 兼容numpy保存格式
    if isinstance(bins, np.ndarray):
        bins = torch.from_numpy(bins)

    bins = bins.float()



    # =========================
    # malicious model
    # =========================


    malicious_weight=(
        1/(128*10*10)
        *
        torch.ones(
            2048,
            128*10*10
        )
    ).to(device).float()



    malicious_layer2_weight=(
        1/2048
        *
        torch.ones(
            1024,
            2048
        )
    ).to(device).float()



    malicious_layer2_bias=(
        1/1024
        *
        torch.ones(
            1024
        )
    ).to(device).float()



    malicious_layer3_weight=torch.rand(
        200,
        1024
    ).to(device).float()



    malicious_layer3_bias=torch.rand(
        200
    ).to(device).float()



    malicious_bias=(
        -bins
    ).to(device).float()



    number=0


    mse_score=torch.zeros(
        test_rounds*all_epoch
    )


    PSNR_score=torch.zeros(
        test_rounds*all_epoch
    )



    start_time=time.time()



    for epoch in range(all_epoch):


        for idx,(train_x,train_label) in enumerate(test_loader):


            if idx>=test_rounds:
                break



            aggregated_weight=torch.zeros(
                2048,
                128*10*10
            )


            aggregated_bias=torch.zeros(
                2048,
                128*10*10
            )


            gradient_weight=None

            l1output=None



            for j in range(client_num):


                model=Classifier().to(device)



                # copy encoder

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



                sgd=SGD(
                    model.parameters(),
                    lr=0.1
                )


                loss_fn=CrossEntropyLoss()



                if j==0:


                    handle=model.fc1.register_forward_hook(
                        get_activation("fc1")
                    )


                    predict_y=model(
                        train_x.float().to(device)
                    )


                    l1output=activation["fc1"].view(
                        batch_size,
                        -1
                    )


                    handle.remove()



                    loss=loss_fn(
                        predict_y,
                        train_label.long().to(device)
                    )


                    loss.backward()



                    gradient_weight=model.fc1.weight.grad.data

                    gradient_bias=model.fc1.bias.grad.data.view(
                        2048,
                        1
                    ).expand(
                        2048,
                        128*10*10
                    )



                for k in range(local_epoch):


                    x_local=train_x[
                        j*client_size:
                        (j+1)*client_size
                    ].to(device)



                    y_local=train_label[
                        j*client_size:
                        (j+1)*client_size
                    ].to(device)



                    sgd.zero_grad()


                    pred=model(
                        x_local.float()
                    )


                    loss=loss_fn(
                        pred,
                        y_local.long()
                    )


                    loss.backward()

                    sgd.step()



                aggregated_weight += (
                    model.fc1.weight.data.cpu()
                    /
                    client_num
                )



                aggregated_bias += (
                    model.fc1.bias.data.cpu()
                    .view(2048,1)
                    .expand(
                        2048,
                        128*10*10
                    )
                    /
                    client_num
                )



            estimated_weight_gradient=(
                -(aggregated_weight
                -
                1/(128*10*10)
                *
                torch.ones(
                    2048,
                    128*10*10
                ))
                /
                (local_epoch*0.1)
            )


            estimated_bias_gradient=(
                -(aggregated_bias
                +
                bins.view(
                    2048,1
                ).expand(
                    2048,
                    128*10*10
                ))
                /
                (local_epoch*0.1)
            )



            original_data=train_x.to(device)


            # [DEBUG] whole-matrix mean grad diff (reference only)
            if args.debug_grad_diff and idx == 0:
                est_w = estimated_weight_gradient.to(device)
                est_b = estimated_bias_gradient.to(device)
                diff_w = (est_w - gradient_weight).abs()
                diff_b = (est_b - gradient_bias).abs()
                true_w_mean = gradient_weight.abs().mean().item()
                true_b_mean = gradient_bias.abs().mean().item()
                print(
                    f"  [debug] (whole-matrix, reference only) weight grad diff mean="
                    f"{diff_w.mean().item():.6e}, true |grad_w| mean={true_w_mean:.6e}, "
                    f"relative error={diff_w.mean().item()/max(true_w_mean,1e-12):.3f}"
                )
                print(
                    f"  [debug] (whole-matrix, reference only) bias grad diff mean="
                    f"{diff_b.mean().item():.6e}, true |grad_b| mean={true_b_mean:.6e}, "
                    f"relative error={diff_b.mean().item()/max(true_b_mean,1e-12):.3f}"
                )


            # [DEBUG] optionally bypass the estimated gradient entirely and
            # feed the true gradient straight into attack(), for A/B testing.
            if args.use_true_grad:
                weight_for_attack = gradient_weight.to(device)
                bias_for_attack = gradient_bias.to(device)
            else:
                weight_for_attack = estimated_weight_gradient.to(device)
                bias_for_attack = estimated_bias_gradient.to(device)


            result=attack(

                weight_for_attack,

                bias_for_attack,

                original_data,

                l1output,

                batch_size,

                idx,

                true_weight=gradient_weight.to(device),

                true_bias=gradient_bias.to(device),

                debug_latent_diff=args.debug_latent_diff

            )


            number += result[0].item()


            mse_score[
                epoch*test_rounds+idx
            ]=result[1].item()


            PSNR_score[
                epoch*test_rounds+idx
            ]=result[2].item()



    end_time=time.time()



    print(
        "number",
        number/(test_rounds*batch_size*all_epoch)
    )


    print(
        "mse score",
        torch.nansum(mse_score)
        /
        torch.nansum(mse_score<0.03)
    )


    print(
        "PSNR score",
        torch.nansum(PSNR_score)
        /
        torch.nansum(PSNR_score>1)
    )


    print(
        "attack time",
        (end_time-start_time)
        /
        (all_epoch*test_rounds)
    )