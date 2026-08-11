# Scale-MIA: Reproduced & Extended

This repository is an **extended reproduction** of the NDSS'25 paper *"Scale-MIA: A Scalable Model Inversion Attack against Secure Federated Learning via Latent Space Reconstruction."*

- **Original Author Repo**: [unknown123489/Scale-MIA](https://github.com/unknown123489/Scale-MIA)
- **Paper**: [arXiv 2311.05808](https://arxiv.org/abs/2311.05808)

> **🌏 中文版说明见下方 [中文版 README](#中文版readme)（完整中文文档在 `README_zh.md`）**
> **English README continues below; a full Chinese version is also available in [`README_zh.md`](README_zh.md).**

---

# 中文版README

> **Scale-MIA 复现与扩展（中文说明）**

本项目是对 NDSS'25 论文 *"Scale-MIA: A Scalable Model Inversion Attack against Secure Federated Learning via Latent Space Reconstruction"*（arXiv: 2311.05808）的**扩展复现**。

- 原论文作者仓库：[unknown123489/Scale-MIA](https://github.com/unknown123489/Scale-MIA)
- 论文地址：[arXiv 2311.05808](https://arxiv.org/abs/2311.05808)

---

## 一、本仓库与原论文/原仓库的主要区别

### 1. 新增了原论文没有的数据集与实验（扩展内容）

| 新增内容 | 位置 | 说明 |
|---------|------|------|
| **ImageNette 数据集** | `fedavg-imagenette/` | ImageNet 的 10 类子集（94×94），中分辨率、更具挑战性的基准，原论文只用了 4 个数据集 |
| **CelebA 人脸数据集** | `fedavg-celeba/` | 大规模人脸属性数据集（178×218），验证 Scale-MIA 在人脸图像上的攻击效果 |
| **Table IV 扩展为 6 个数据集** | 汇总见 `FedAVG实验数据汇总.xlsx` | 原论文 Table IV 只有 4 个数据集（CIFAR-10/FMNIST/HMNIST/TinyImageNet），本仓库新增 ImageNette、CelebA 两个数据集并给出「复现 vs 论文」的 Δ 差值表 |
| **Table VIII（数据偏斜）完整复现** | `fedavg-tinyimagenet/` | 原仓库只有论文数值，本仓库**完整复现**了 data skew 实验：monarch butterfly（帝王蝶，辅助数据）→ sulfur butterfly（硫黄蝶，intra 类）/ bullfrog（牛蛙，inter 类），含严格模式（500 张辅助图）全流程 |
| **Rob 探索性变体** | `fedavg-cifar/Rob-*.py` | 原论文没有的 CIFAR-10 鲁棒性变体探索代码 |

**Table VIII 复现流程（本仓库新增脚本）：**

| 脚本 | 作用 |
|------|------|
| `monarch-adv-train.py` | 从全量预训练权重出发，仅用帝王蝶数据微调自编码器（2000 epoch + 数据增强），使 decoder 专精蝴蝶、部分"遗忘"牛蛙 |
| `monarch-para-gen.py` | 用微调后的 encoder 在帝王蝶数据上统计 relu3 层输出分布，生成 2048 个 latent bins |
| `attack-skew.py` | 执行数据偏斜攻击：`--skew same`（自洽性检查）/ `intra`（硫黄蝶）/ `inter`（牛蛙） |

```bash
cd fedavg-tinyimagenet
python monarch-adv-train.py
python monarch-para-gen.py
python attack-skew.py --skew same  --batch_size 16   # sanity check
python attack-skew.py --skew intra --batch_size 16   # 硫黄蝶
python attack-skew.py --skew inter --batch_size 16   # 牛蛙
# 然后重复 batch_size = 32 / 64 / 128 / 256
```

**复现结果（Table VIII）：**

| Batch | Intra Rate | Intra PSNR | Inter Rate | Inter PSNR |
|-------|-----------|------------|------------|------------|
| 16 | 0.9395 | 23.5367 | 0.9435 | 22.8295 |
| 32 | 0.8938 | 23.4400 | 0.9250 | 22.8096 |
| 64 | 0.8638 | 23.4740 | 0.8750 | 22.4771 |
| 128 | 0.7214 | 23.6927 | 0.8411 | 22.1383 |
| 256 | 0.7227 | 23.6377 | 0.7422 | 22.2338 |

> inter PSNR 系统性比 intra 低 0.6–1.5 dB，说明微调后 decoder 确实专精蝴蝶、对牛蛙重建质量更差。inter Rate 仍高于论文是因为 encoder 保留了对牛蛙的特征提取能力。

### 2. 根据本机设备修改/适配的代码

原论文在 Linux 服务器（Intel i7-8700K + 双 RTX 2080 Ti）上运行，本仓库针对 **Windows + 单 GPU / 无 GPU** 环境做了如下适配：

| 适配项 | 修改内容 |
|--------|---------|
| **设备自适应** | 所有脚本将硬编码的 `cuda` 改为 `device = 'cuda:0' if torch.cuda.is_available() else 'cpu'`，无 GPU 也能跑（速度较慢） |
| **模型加载兼容** | `torch.load(..., map_location=device)` 统一处理，避免在无 GPU 机器上加载 `.pkl` 权重时报错 |
| **数据集获取改为 HuggingFace** | TinyImageNet 等数据集改用 `datasets.load_dataset('Maysee/tiny-imagenet')`，无需手动下载 245MB 压缩包，支持断点缓存（HF 缓存目录） |
| **新增 CelebA / ImageNette 训练-参数-攻击三件套** | 为两个新数据集编写了新的 `adv-train / para-gen / recover-attack` 脚本，复用了原论文 3 步 pipeline 结构 |
| **实验数据汇总 Excel** | 新增 `FedAVG实验数据汇总.xlsx`，一键汇总全部表格和「复现 vs 论文」Δ 差值，方便写报告 |

---

## 二、Table 对应关系

| 表格 | 内容 | 位置 |
|------|------|------|
| **Table IV** | 重建率 & PSNR（FedSGD / FedAVG×3 / FedAVG×5，batch 64–1024） | 6 个数据集（含新增 ImageNette、CelebA） |
| **Table V** | 不同模型架构（CNN/ResNet/VGGNet/AlexNet/ViT） | `fedavg-cifar/` — `multimodel-*` 脚本 |
| **Table VI** | 数据不足（1%/3%/10%/100% 辅助数据） | `fedavg-cifar/` — `--aux=N` 参数 |
| **Table VII** | 数据偏差（1/2/3/10 类） | `fedavg-cifar/` — `targeted-*` 脚本 |
| **Table VIII** | 数据偏斜（TinyImageNet intra/inter 类） | `fedavg-tinyimagenet/` — **本仓库完整复现** |
| **Table IX** | 差分隐私防御（DP） | `dp-cifar/` — `dp-recover-attack.py` |

## 三、快速开始

```bash
pip install -r requirements.txt

# 例：复现 TinyImageNet 的 Table IV
cd fedavg-tinyimagenet
python fedavg-recover-attack.py --batch_size=64 --test_round=5
```

> 预训练模型与 bins 已包含在 `models/` 和 `data/` 目录，可直接运行攻击脚本，无需重新训练。

## 四、复现笔记

- **Table IV intra Rate** 比论文略低（−0.05 ~ −0.13），PSNR 基本持平或略高；
- **HMNIST** 部分 batch 偏差较大，源于医学图像与自然图像的域差距；
- **ImageNette** 重建率低于 CIFAR，符合论文"分辨率越高攻击越难"的观察；
- **CelebA** 人脸重建率较高（0.91→0.55），验证了 Scale-MIA 对人脸数据的有效性；
- **Table VIII** 复现难点是自编码器泛化——直接用全量预训练 AE 会让 decoder 对蝴蝶和牛蛙重建同样好，本仓库通过"仅帝王蝶微调"制造 inter/intra 差距。

---

*继续阅读英文完整文档 ↓*

---

## What's Different from the Original Repo

| Aspect | Original Repo | This Repo |
|--------|--------------|-----------|
| Datasets | CIFAR-10, FMNIST, HMNIST, TinyImageNet (4) | + ImageNette, + CelebA (6 total) |
| Table IV | 4 datasets, paper results only | 6 datasets, **复现 vs 论文 with Δ deltas** |
| Table VIII (data skew) | Paper results only | **Fully reproduced** with monarch/sulfur butterfly & bullfrog |
| Excel summary | Not provided | `FedAVG实验数据汇总.xlsx` with all tables |
| Exploratory code | None | `Rob-para-gen.py`, `Rob-recover-attack.py` (CIFAR-10 robustness variants) |

### Dataset Additions

| Dataset | Directory | Description |
|---------|-----------|-------------|
| **ImageNette** | `fedavg-imagenette/` | 10-class subset of ImageNet (94×94), a more challenging mid-resolution benchmark |
| **CelebA** | `fedavg-celeba/` | Large-scale face attribute dataset (178×218), testing Scale-MIA on face images |

Both follow the same 3-step Scale-MIA pipeline: `fedavg-adv-train.py` → `fedavg-para-gen.py` → `fedavg-recover-attack.py`.

### Table VIII (Data Skew) — Reproduced

The original paper's Table VIII was fully reproduced under the **strict mode** setting (500 monarch butterfly auxiliary images). Additional scripts in `fedavg-tinyimagenet/`:

| Script | Purpose |
|--------|---------|
| `monarch-adv-train.py` | Fine-tune autoencoder on monarch butterfly only (from full pre-trained weights, 2000 epochs with augmentation) |
| `monarch-para-gen.py` | Generate 2048 latent space bins from monarch-tuned encoder |
| `attack-skew.py` | Execute data skew attack: same-class (sanity), intra-class (sulfur butterfly), inter-class (bullfrog) |

**Execution:**

```bash
cd fedavg-tinyimagenet

# Step 1: Fine-tune autoencoder on monarch butterfly
python monarch-adv-train.py

# Step 2: Generate monarch-specific bins
python monarch-para-gen.py

# Step 3: Sanity check (same-class)
python attack-skew.py --skew same --batch_size 16

# Step 4: Intra-class skew (sulfur butterfly)
python attack-skew.py --skew intra --batch_size 16

# Step 5: Inter-class skew (bullfrog)
python attack-skew.py --skew inter --batch_size 16

# Repeat for batch sizes 32, 64, 128, 256
```

**Reproduced Results (Table VIII):**

| Batch | Intra Rate | Intra PSNR | Inter Rate | Inter PSNR |
|-------|-----------|------------|------------|------------|
| 16 | 0.9395 | 23.5367 | 0.9435 | 22.8295 |
| 32 | 0.8938 | 23.4400 | 0.9250 | 22.8096 |
| 64 | 0.8638 | 23.4740 | 0.8750 | 22.4771 |
| 128 | 0.7214 | 23.6927 | 0.8411 | 22.1383 |
| 256 | 0.7227 | 23.6377 | 0.7422 | 22.2338 |

> Inter PSNR is consistently 0.6–1.5 dB lower than intra, confirming the decoder specializes in butterflies after fine-tuning. Inter Rate remains higher than the paper because the encoder retains pre-trained feature extraction on bullfrog.

---

## Paper Table Mapping

| Table | Content | Location |
|-------|---------|----------|
| **Table IV** | Reconstruction rate & PSNR (FedSGD / FedAVG×3 / FedAVG×5, batch 64–1024) | 6 datasets: CIFAR-10, FMNIST, HMNIST, TinyImageNet, ImageNette, CelebA |
| **Table V** | Different model architectures (CNN, ResNet, VGGNet, AlexNet, ViT) | `fedavg-cifar/` — `multimodel-*` scripts |
| **Table VI** | Data deficiency (1%/3%/10%/100% auxiliary data) | `fedavg-cifar/` — `fedavg-recover-attack.py --aux=N` |
| **Table VII** | Data bias (1/2/3/10 classes) | `fedavg-cifar/` — `targeted-*` scripts |
| **Table VIII** | Data skew (intra/inter class on TinyImageNet) | `fedavg-tinyimagenet/` — `attack-skew.py` (this repo: fully reproduced) |
| **Table IX** | Differential privacy defense | `dp-cifar/` — `dp-recover-attack.py` |

### Combined Excel (`FedAVG实验数据汇总.xlsx`)

Contains all tables from the paper in one file:
- **Table IV 汇总** sheet: All 6 datasets side by side, with 复现 Rate/PSNR, 论文 Rate/PSNR, and Δ deltas (green = better than paper, orange = worse)
- Table V–IX sheets: Paper reference data preserved as-is

---

## Abstract

Federated learning (FL) has been widely regarded as a privacy-preserving distributed learning paradigm. However, in this work, we propose a novel model inversion attack named Scale-MIA that breaks this property to have the malicious server reconstruct private local samples hold by the clients, even when the FL system is protected by the state-of-the-art secure aggregation mechanism. To achieve this, the attacker modifies the global model's parameters before sending it to the clients. The clients, without finding the stealthy and unnoticeable modifications, will train the local models according to the adversarial global model and send their model updates back. Then the attacker, i.e. the server, receives the aggregation of these model updates when the secure aggregation mechanism is in place, and can reconstruct local training samples from them with an efficient analytical method.

## Get Started

### Hardware Requirements

We run all the experiments on a server equipped with an Intel Core i7-8700K CPU @ 3.70GHz×12, two GeForce RTX 2080 Ti GPUs, and Ubuntu 18.04.3 LTS. (Also tested successfully on Windows with a single NVIDIA GPU.)

### Software Requirements

This implementation is PyTorch-based. Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch torchvision numpy matplotlib opacus einops datasets scipy tqdm pillow
```

### Quick Start

```bash
git clone https://github.com/weimeng09280929/Scale-MIA.git
cd Scale-MIA

# Run a single dataset (e.g., TinyImageNet Table IV)
cd fedavg-tinyimagenet
python fedavg-recover-attack.py --batch_size=64 --test_round=5
```

> **Note**: Pre-trained models and attack parameters (bins) are provided in `models/` and `data/` folders. You can directly run `fedavg-recover-attack.py` without training.

---

## Repository Structure

```
Scale-MIA-main/
├── fedavg-cifar/          # CIFAR-10: Table IV, V, VI, VII + multi-model + targeted attacks
├── fedavg-fmnist/         # FashionMNIST: Table IV
├── fedavg-hmnist/         # HMNIST (histopathology): Table IV
├── fedavg-tinyimagenet/   # TinyImageNet: Table IV + Table VIII (data skew)
│   ├── fedavg-adv-train.py
│   ├── fedavg-para-gen.py
│   ├── fedavg-recover-attack.py
│   ├── monarch-adv-train.py      # ★ NEW: fine-tune AE on monarch butterfly
│   ├── monarch-para-gen.py       # ★ NEW: generate monarch-specific bins
│   └── attack-skew.py            # ★ NEW: data skew attack (same/intra/inter)
├── fedavg-imagenette/     # ★ NEW: ImageNette dataset
├── fedavg-celeba/         # ★ NEW: CelebA dataset
├── dp-cifar/              # CIFAR-10 with DP defense: Table IX
├── README.md              # Bilingual (中文+English)
├── README_zh.md           # ★ Chinese full documentation
├── requirements.txt
├── FedAVG实验数据汇总.xlsx  # ★ NEW: combined experiment results with paper deltas
└── LICENSE
```

For each dataset folder, the core 3-step pipeline:

1. **`fedavg-adv-train.py`** — Train surrogate autoencoder (optional, pre-trained weights provided)
2. **`fedavg-para-gen.py`** — Generate attack parameters / latent space bins (optional, pre-generated bins provided)
3. **`fedavg-recover-attack.py`** — Main attack execution (can run directly with pre-trained models)

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 128 | Reconstruction batch size (64, 128, 256, 512, 1024) |
| `--test_rounds` | 10 | Number of reconstructed batches |
| `--client_num` | 8 | Number of FL clients (4, 8, 16) |
| `--local_epoch` | 1 | Local training epochs per client |

---

## Extended Experiments

### Data Skew on TinyImageNet (Table VIII)

```bash
cd fedavg-tinyimagenet
python monarch-adv-train.py          # Fine-tune autoencoder
python monarch-para-gen.py            # Generate monarch bins
python attack-skew.py --skew same --batch_size 16   # Sanity check
python attack-skew.py --skew intra --batch_size 16  # Intra: sulfur butterfly
python attack-skew.py --skew inter --batch_size 16  # Inter: bullfrog
```

### ImageNette & CelebA (Extended Table IV)

```bash
# ImageNette
cd fedavg-imagenette
python fedavg-recover-attack.py --batch_size=64

# CelebA
cd fedavg-celeba
python fedavg-recover-attack-celeba.py --batch_size=64
```

### Different Model Architectures (Table V)

```bash
cd fedavg-cifar
python multimodel-recover-attack.py --model_name=Resnet
# Options: CNN, VGGNet, AlexNet, ResNet, ViT
```

### Data Deficiency (Table VI)

```bash
cd fedavg-cifar
python fedavg-recover-attack.py --aux=10   # 10% auxiliary data
# Options: --aux=1, 3, 10, 100
```

### Data Bias (Table VII)

```bash
cd fedavg-cifar
python targeted-recover-attack.py
```

### Differential Privacy (Table IX)

```bash
cd dp-cifar
python dp-recover-attack.py --delta=1e-4 --epsilon=1
```

---

## Reproduction Notes

### Table IV — Key Differences from Paper

- **Intra Rate**: Slightly lower than paper (−0.05 to −0.13 across datasets), PSNR generally comparable or slightly higher.
- **Inter-class**: HMNIST shows larger deviation at some batch sizes due to the medical image domain gap.
- **ImageNette**: Added as a more challenging mid-resolution benchmark; reconstruction rates are lower (CIFAR < ImageNette < TinyImageNet scale), consistent with the paper's observation that higher resolution hurts attack performance.
- **CelebA**: Face reconstruction rates are relatively high (0.91→0.55 across batch sizes), demonstrating Scale-MIA's effectiveness on face datasets.

### Table VIII — Data Skew Reproduction Notes

The key challenge in reproducing Table VIII is the autoencoder's generalization. Using a fully pre-trained autoencoder (trained on all 200 TinyImageNet classes) causes the decoder to reconstruct both butterflies and bullfrogs equally well, failing to show the inter-class degradation.

Our approach: fine-tune the pre-trained autoencoder on **only** monarch butterfly data (2000 epochs with augmentation), allowing the decoder to specialize in butterflies and partially forget non-butterfly classes. This produces the expected inter < intra PSNR gap (0.6–1.5 dB), though the inter Rate remains higher than the paper due to the encoder's retained general-purpose feature extraction.
