# Scale-MIA: Reproduced & Extended

<p align="center">
  <b>English</b> · <a href="README_zh.md">简体中文</a>
</p>

This repository is an **extended reproduction** of the NDSS'25 paper *"Scale-MIA: A Scalable Model Inversion Attack against Secure Federated Learning via Latent Space Reconstruction."*

- **Original Author Repo**: [unknown123489/Scale-MIA](https://github.com/unknown123489/Scale-MIA)
- **Paper**: [arXiv 2311.05808](https://arxiv.org/abs/2311.05808)

```bibtex
@misc{shi2023scalemiascalablemodelinversion,
      title={Scale-MIA: A Scalable Model Inversion Attack against Secure Federated Learning via Latent Space Reconstruction},
      author={Shanghao Shi and Ning Wang and Yang Xiao and Chaoyu Zhang and Yi Shi and Y. Thomas Hou and Wenjing Lou},
      year={2023},
      eprint={2311.05808},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2311.05808},
}
```

---

## What's Different from the Original Repo

| Aspect | Original Repo | This Repo |
|--------|--------------|-----------|
| Datasets | CIFAR-10, FMNIST, HMNIST, TinyImageNet (4) | + ImageNette, + CelebA (6 total) |
| Table IV | 4 datasets, paper results only | 6 datasets, **reproduction vs paper with Δ deltas** |
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
Scale-MIA/
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
├── README.md              # English documentation
├── README_zh.md           # ★ Chinese documentation
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
