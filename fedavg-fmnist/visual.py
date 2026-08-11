import matplotlib.pyplot as plt 
import torch


batch_size=64
dim=28
channel=1


input_data=torch.load(
    f"data/data_batch_{batch_size}.pt",
    map_location="cpu"
)


recovered_data=torch.load(
    f"data/recoved_{batch_size}.pt",
    map_location="cpu"
)


sorted_list=torch.load(
    f"data/list_{batch_size}.pt",
    map_location="cpu"
)


input_data=input_data.reshape(
    batch_size,
    channel,
    dim,
    dim
)


recovered_data=recovered_data.reshape(
    batch_size,
    channel,
    dim,
    dim
)



# =====================
# 原始图片
# =====================

plt.figure(figsize=(10,10))


for i in range(batch_size):

    img=input_data[
        sorted_list[batch_size-1-i]
    ][0]


    plt.subplot(8,8,i+1)
    plt.imshow(
        img.numpy(),
        cmap="gray"
    )
    plt.axis("off")


plt.savefig(
    "figs/original_sample.png",
    dpi=300
)


# =====================
# 重构图片
# =====================

plt.figure(figsize=(10,10))


for i in range(batch_size):

    img=recovered_data[i][0]


    plt.subplot(8,8,i+1)

    plt.imshow(
        img.detach().numpy(),
        cmap="gray"
    )

    plt.axis("off")


plt.savefig(
    "figs/reconstructed_sample.png",
    dpi=300
)


plt.show()