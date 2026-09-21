# Scale-MIA 复现与扩展（中文文档）

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

> 本项目是对 NDSS'25 论文 *"Scale-MIA: A Scalable Model Inversion Attack against Secure Federated Learning via Latent Space Reconstruction"*（arXiv: 2311.05808）的**扩展复现**。

- **原论文作者仓库**：[unknown123489/Scale-MIA](https://github.com/unknown123489/Scale-MIA)
- **论文地址**：[arXiv 2311.05808](https://arxiv.org/abs/2311.05808)
- **本仓库**：[weimeng09280929/Scale-MIA](https://github.com/weimeng09280929/Scale-MIA)

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

## 一、本仓库与原论文/原仓库的主要区别

### 1. 新增了原论文没有的数据集与实验（扩展内容）

| 新增内容 | 位置 | 说明 |
|---------|------|------|
| **ImageNette 数据集** | `fedavg-imagenette/` | ImageNet 的 10 类子集（94×94），中分辨率、更具挑战性的基准，原论文只用了 4 个数据集 |
| **CelebA 人脸数据集** | `fedavg-celeba/` | 大规模人脸属性数据集（178×218），验证 Scale-MIA 在人脸图像上的攻击效果 |
| **Table IV 扩展为 6 个数据集** | 汇总见 `FedAVG实验数据汇总.xlsx` | 原论文 Table IV 只有 4 个数据集（CIFAR-10/FMNIST/HMNIST/TinyImageNet），本仓库新增 ImageNette、CelebA 两个数据集并给出「复现 vs 论文」的 Δ 差值表 |
| **Table VIII（数据偏斜）完整复现** | `fedavg-tinyimagenet/` | 原仓库只有论文数值，本仓库**完整复现**了 data skew 实验：monarch butterfly（帝王蝶，辅助数据）→ sulfur butterfly（硫黄蝶，intra 类）/ bullfrog（牛蛙，inter 类），含严格模式（500 张辅助图）全流程 |
| **Rob 探索性变体** | `fedavg-cifar/Rob-*.py` | 原论文没有的 CIFAR-10 鲁棒性变体探索代码 |

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

### Table VIII 复现流程（本仓库新增脚本）

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


---

## 三、安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：

```bash
pip install torch torchvision numpy matplotlib opacus einops datasets scipy tqdm pillow
```

> 原论文环境为 Linux + 双 GPU，本仓库已在 **Windows 单 GPU / CPU** 环境验证可运行。

## 四、快速开始

```bash
git clone https://github.com/weimeng09280929/Scale-MIA.git
cd Scale-MIA

# 例：复现 TinyImageNet 的 Table IV
cd fedavg-tinyimagenet
python fedavg-recover-attack.py --batch_size=64 --test_round=5
```

> 预训练模型与 bins 已包含在 `models/` 和 `data/` 目录，可直接运行攻击脚本，无需重新训练。

## 五、仓库结构

```
Scale-MIA/
├── fedavg-cifar/          # CIFAR-10：Table IV, V, VI, VII + 多模型 + targeted 攻击 + Rob 变体
├── fedavg-fmnist/         # FashionMNIST：Table IV
├── fedavg-hmnist/         # HMNIST（病理图像）：Table IV
├── fedavg-tinyimagenet/   # TinyImageNet：Table IV + Table VIII（数据偏斜）
│   ├── fedavg-adv-train.py
│   ├── fedavg-para-gen.py
│   ├── fedavg-recover-attack.py
│   ├── monarch-adv-train.py      # ★ 新增：仅帝王蝶微调自编码器
│   ├── monarch-para-gen.py       # ★ 新增：生成帝王蝶专属 bins
│   └── attack-skew.py            # ★ 新增：数据偏斜攻击（same/intra/inter）
├── fedavg-imagenette/     # ★ 新增：ImageNette 数据集
├── fedavg-celeba/         # ★ 新增：CelebA 数据集
├── dp-cifar/              # CIFAR-10 + DP 防御：Table IX
├── README.md              # 中英双语
├── README_zh.md           # ★ 中文文档
├── requirements.txt
├── FedAVG实验数据汇总.xlsx  # ★ 新增：实验数据汇总（含复现 vs 论文差值）
└── LICENSE
```

每个数据集目录的核心 3 步流程：

1. **`fedavg-adv-train.py`** — 训练替代自编码器（可选，已提供预训练权重）
2. **`fedavg-para-gen.py`** — 生成攻击参数 / latent bins（可选，已提供预生成 bins）
3. **`fedavg-recover-attack.py`** — 主攻击脚本（可直接用预训练模型运行）

## 六、扩展实验

### Table VIII（TinyImageNet 数据偏斜）

```bash
cd fedavg-tinyimagenet
python monarch-adv-train.py          # 微调自编码器
python monarch-para-gen.py            # 生成帝王蝶 bins
python attack-skew.py --skew same --batch_size 16   # 自洽性检查
python attack-skew.py --skew intra --batch_size 16  # intra：硫黄蝶
python attack-skew.py --skew inter --batch_size 16  # inter：牛蛙
```

### 扩展 Table IV（ImageNette & CelebA）

```bash
# ImageNette
cd fedavg-imagenette
python fedavg-recover-attack.py --batch_size=64

# CelebA
cd fedavg-celeba
python fedavg-recover-attack-celeba.py --batch_size=64
```

### 不同模型架构（Table V）

```bash
cd fedavg-cifar
python multimodel-recover-attack.py --model_name=Resnet
# 可选：CNN, VGGNet, AlexNet, ResNet, ViT
```

### 数据不足（Table VI）

```bash
cd fedavg-cifar
python fedavg-recover-attack.py --aux=10   # 10% 辅助数据
# 可选：--aux=1, 3, 10, 100
```

### 数据偏差（Table VII）

```bash
cd fedavg-cifar
python targeted-recover-attack.py
```

### 差分隐私防御（Table IX）

```bash
cd dp-cifar
python dp-recover-attack.py --delta=1e-4 --epsilon=1
```
