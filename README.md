# UEKD：多模态虚假新闻检测的单模态事件无关知识蒸馏（实验代码）

本仓库是论文 **"Uni-Modal Event-Agnostic Knowledge Distillation for Multimodal
Fake News Detection"** (Liu et al., IEEE TKDE 2024, vol. 36, no. 12) 的 PyTorch
实验复现框架。UEKD 是一个即插即用模块：通过模态掩码分离多模态模型中的单模态
预测通道，用跨域验证得到的"事件无关"教师标签对其做预测级蒸馏，并用 Shapley
值动态平衡各模态的蒸馏权重，缓解多模态联合训练中的模态失衡（弱模态欠优化）
问题。

## 论文方法 → 代码位置对照

| 论文内容 | 公式/算法 | 代码位置 |
|---|---|---|
| 统一骨干抽象 `ŷ = σ(f(ϕ_t, ϕ_v, ψ))` | Eq. 1 | `uekd/models/backbone.py` |
| 高斯模态掩码（批内均值/方差采样，融合阶段之前） | Eq. 8–9 | `uekd/models/masking.py` |
| Late-fusion 骨干（SpotFake+ 风格简化版） | Table I | `uekd/models/late_fusion.py` |
| Co-attention 骨干（MCAN/HMCAN 风格简化版，含 ψ） | Table I | `uekd/models/attn_fusion.py` |
| 跨域验证教师：逐域训练、域外预测、中性化精炼 | Alg. 1 | `uekd/framework/teacher.py` |
| KNN/聚类域划分（GossipCop 协议） | Sec. IV-B3 | `uekd/data/domains.py` |
| 蒸馏损失 `L_KD` (MSE)、`L_GT` (BCE)、`L^m = L_KD + β·L_GT` | Eq. 10–12 | `uekd/framework/distill.py` |
| Shapley 值 `φ^m`、欠优化度 `γ^m`、归一化权重 `λ^m` | Eq. 13–17 | `uekd/framework/distill.py` |
| 总目标 `L = α·L_bce + Σ λ^m·L^m` | Eq. 18 | `uekd/framework/distill.py` |
| 模态内通道评估（掩码下测试，Table IV / Fig.2 协议） | Sec. III-B | `uekd/framework/evaluate.py` |
| 冻结 BERT / CLIP-ResNet50 特征抽取 | Sec. V-A4 | `uekd/extract/` |

骨干模型按论文结构做了**简化实现**：`late_fusion` 对应 SpotFake+ 类的晚期融合，
`co_attention` 用投影 token + 自注意力/交叉注意力近似 MCAN/HMCAN 的上下文注意力
融合，并带 CAFE 风格的跨模态一致性特征 ψ。接口与论文 Eq. 1 的通用框架一致，
替换为你自己的骨干只需实现 `encode()` / `fuse()` 两个方法。

## 项目结构

```
uekd/
├── train_teacher.py        # 阶段1：交叉验证训练单模态教师 → 事件无关标签
├── train_student.py        # 阶段2：UEKD 蒸馏训练多模态学生
├── evaluate.py             # 独立评估（多模态指标 + 单模态通道精度）
├── smoke_test.py           # 合成数据端到端冒烟测试（CPU 可跑）
├── requirements.txt
└── uekd/
    ├── config.py           # 数据集预设与超参（论文 Sec. V-A4）
    ├── runtime.py          # 设备解析/后端优化/AMP/非阻塞迁移（CUDA、MPS、CPU）
    ├── data/               # 特征数据集、域划分、合成数据
    ├── models/             # 骨干 + 高斯掩码 + UEKD 包装器
    ├── framework/          # 教师/蒸馏损失/Shapley 权重/训练器/评估
    └── extract/            # 冻结 BERT、CLIP-ResNet50 特征抽取脚本
```

## 快速开始

### 0. 安装

```bash
pip install -r requirements.txt
```

### 1. 冒烟测试（无需任何真实数据，CPU 几分钟）

```bash
python smoke_test.py                 # 两种骨干全跑
python smoke_test.py --backbone co_attention
```

合成数据模拟了论文研究的核心现象：事件特异性噪声、单模态篡改（约 45% 仅文本、
45% 仅图像被篡改）、事件不相交的训练/测试划分。

### 2. 真实数据集完整流程（以 Weibo21 为例）

**a) 准备原始数据**，放在 `data/weibo21/`：

```
raw_texts.jsonl    # 每行 {"text": "..."}，顺序与样本一一对应
images.list        # 每行一个图片路径
labels.pt          # (N,) long, 1=fake 0=real
domains.pt         # (N,) long 官方域标签（Weibo21=9 域, Twitter=17 事件；
                   # GossipCop 无需提供，代码自动聚类为 10 域）
split.json         # {"train": [idx...], "test": [idx...]}
```

**b) 冻结编码器特征抽取**（论文设置：Weibo21 用 bert-base-chinese / 长度 120，
Twitter 用 twhin-bert-base / 170，GossipCop 用 bert-base-uncased / 200，图像统一
CLIP-ResNet50，全部冻结）：

```bash
python -m uekd.extract.extract_text --dataset weibo21 \
    --texts data/weibo21/raw_texts.jsonl --output data/weibo21/text_feats.pt

python -m uekd.extract.extract_image --dataset weibo21 \
    --images data/weibo21/images.list --output data/weibo21/image_feats.pt
```

**c) 阶段1：跨域教师（Alg.1）**，输出 `teacher_preds_t.pt / teacher_preds_v.pt`：

```bash
python train_teacher.py --dataset weibo21 --backbone co_attention \
    --output-dir ./checkpoints/weibo21 --teacher-epochs 30
```

**d) 阶段2：UEKD 学生蒸馏**（论文超参：batch 32、Adam lr 2e-4、150 epochs、
早停 patience 30、α=β=0.25，均为默认值）：

```bash
python train_student.py --dataset weibo21 --backbone co_attention \
    --teacher-dir ./checkpoints/weibo21 --output-dir ./checkpoints/weibo21
```

**e) 评估**（accuracy / precision / recall / F1，以及掩码下的单模态通道精度，
对应论文 Table III 与 Table IV）：

```bash
python evaluate.py --dataset weibo21 --backbone co_attention \
    --checkpoint ./checkpoints/weibo21/student_co_attention.pt
```

Twitter / GossipCop 流程相同，只需把 `--dataset` 换成 `twitter` / `gossipcop`。
GossipCop 记得先按论文做正负样本平衡下采样（保留全部 fake，对 real 下采样至
1:1）。

## 运行环境：CUDA / MPS / CPU，Windows / Linux

运行层集中在 `uekd/runtime.py`，所有入口脚本（`train_teacher.py`、
`train_student.py`、`evaluate.py`、`smoke_test.py`）共用同一套设备解析与后端
配置，无需改动训练代码即可在 CPU、NVIDIA GPU（CUDA）、Apple Silicon（MPS）
以及 Windows/Linux 之间迁移。

**设备选择（`--device`）**

| 取值 | 行为 |
|---|---|
| `auto`（默认） | 按 CUDA → MPS → CPU 顺序自动挑选可用设备 |
| `cpu` | 强制 CPU |
| `cuda` / `cuda:0` / `cuda:1` | 指定 GPU；无可用 GPU 时直接报错而非静默回退 |
| `mps` | Apple Silicon GPU |

**自动启用 / 自动关闭的能力**

- **混合精度（AMP）**：仅在 CUDA 上自动启用（`torch.amp.autocast` +
  `GradScaler`），CPU/MPS 上自动关闭；加 `--no-amp` 可强制关闭。
- **cuDNN 优化**：CUDA 上自动开启 `cudnn.benchmark` 与 TF32
  （matmul/conv），CPU 上不做任何改动。
- **DataLoader 并行（`--num-workers`）**：默认 `-1` 自适应——Linux 上取
  `min(4, 可用核数)` 并启用 `persistent_workers`；Windows 上取 0（主进程加载，
  规避 spawn 开销与句柄问题）。也可显式指定任意值。CUDA 上自动启用
  `pin_memory`，数据搬运用 `non_blocking` 传输。
- **可复现性**：worker 进程用 `torch.initial_seed()` 派生种子；加
  `--deterministic` 会在 CUDA 上强制确定性算法（变慢但结果可复现）。

**典型用法**

```bash
# Linux 服务器，第 0 张 GPU，4 个 DataLoader worker
python train_teacher.py --dataset weibo21 --backbone co_attention \
    --device cuda:0 --num-workers 4 --output-dir ./checkpoints/weibo21

python train_student.py --dataset weibo21 --backbone co_attention \
    --device cuda:0 --num-workers 4 \
    --teacher-dir ./checkpoints/weibo21 --output-dir ./checkpoints/weibo21

# CPU 调试 / 可复现运行
python train_student.py --dataset synthetic --device cpu --deterministic
```

GPU 版 PyTorch 安装（Linux）：`pip install torch --index-url
https://download.pytorch.org/whl/cu121`（按实际 CUDA 版本选择）。本仓库代码
本身不依赖 CUDA——无 GPU 时所有脚本在 CPU 上同样可运行。

## 与论文的实现差异说明

1. **骨干为简化版**：未逐层还原 SpotFake+/MCAN/HMCAN/CAFE 原始实现，但保留了
   三类融合策略的结构特征与 Eq. 1 的统一接口，UEKD 的插入点（融合阶段之前的
   掩码）与论文完全一致。
2. **教师早停**：论文称教师"训练至收敛"，本实现用 D−d_i 内部 10% 留出做早停，
   上限 `--teacher-epochs`（默认 30），与论文"单模态训练收敛更快"的描述一致。
3. **早停监控**：论文用测试精度早停（默认 `--monitor test`）；如需更严谨的
   协议可加 `--monitor val` 并通过代码传入验证集。
4. **GossipCop 域聚类**：论文用 Sentence-BERT 句向量聚类；`uekd/data/domains.py`
   默认用冻结文本特征的均值池化 + KMeans（等价输入源，无需额外下载 SBERT；如需
   完全对齐可自行替换 `knn_cluster_domains` 的输入向量）。
5. **Shapley 数值保护**：`φ^m` 可能非正，代码用 `max(φ, 1e-3)` 钳制后再求 `γ^m`
   （`--shapley-eps`），避免除零/负权重。

## 主要超参（默认值即论文最优设置）

| 超参 | 默认 | 说明 |
|---|---|---|
| `--batch-size` | 32 | Sec. V-A4 |
| `--lr` | 2e-4 | Adam, Sec. V-A4 |
| `--epochs` | 150 | Sec. V-A4 |
| `--patience` | 30 | 早停, Sec. V-A4 |
| `--alpha` | 0.25 | Eq. 18，Fig. 9 最优 |
| `--beta` | 0.25 | Eq. 12，Fig. 9 最优 |
| `--teacher-epochs` | 30 | 教师收敛更快, Sec. VI-F |
| `--hidden-dim` | 128 | 骨干投影维度 |
| `--device` | `auto` | `cpu` / `cuda:N` / `mps`，见上文运行环境一节 |
| `--num-workers` | -1（自适应） | Linux>0、Windows=0 |
| `--no-amp` | 关 | 强制关闭 CUDA 混合精度 |
| `--deterministic` | 关 | CUDA 确定性内核（更慢、可复现） |
