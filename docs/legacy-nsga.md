# 历史实现：NSGA-II + 连续值 Liquid Transformer

本页归档 `3babf5d` 时的说明，供复查旧实验协议。当前主线已切换到 PPO 与 LIF FFN，
请以 [项目 README](../README.md) 为准。旧入口 `train/evolve.py` 仍保留用于回归对照；
其数据目录默认值现已统一到用户目录下的 `dataset/`，历史示例中的环境名不再适用。

面向 CIFAR-100 的模块化 Vision Transformer 训练工程，提供 **DeiT-Tiny**、
**Swin-Tiny** solid 基线，以及用 NSGA-II 协同进化准确率和小世界拓扑的 liquid 模型。

## 现有实验结果与结论

截至 2026-09-10，核查的代码版本为 `52e0e03`。当前本地 `train-log/`
（包括 Git 忽略的文件）和该版本 GitHub `main` 中均只有说明文件与占位文件，
未发现训练日志、checkpoint、`population.csv`、`evolution.jsonl` 或 `finalists.json`。
这表示当前工作副本没有可核验的实验结果，不代表其他环境中没有运行过实验。

| 实验 | 可核验的准确率 / loss | 当前可得结论 |
| --- | --- | --- |
| Solid DeiT-Tiny / Swin-Tiny | 暂无记录 | 无法比较两种主干的收敛与泛化表现 |
| Liquid 粗搜索 / 精搜索 | 暂无记录 | 无法判断 Pareto 前沿是否改善，或拓扑评分与精度是否相关 |
| Liquid 最终候选 | 暂无记录 | 无法报告测试精度、相对 solid 的提升、耗时或显存开销 |

下面的模型结构、训练预算和输出格式来自代码核对，属于实现说明，不是实测结果。
测试中的 `80.0 / 75.0 / 70.0` 是 NSGA-II 选择测试的人造输入，不是 CIFAR-100 实验精度。

### 评估口径与比较限制

| 阶段 | 默认训练数据 | 评估数据与记录含义 | 权重选择 |
| --- | --- | --- | --- |
| Solid | 官方训练集 50,000 张 | 每轮评估官方测试集 10,000 张；日志虽命名 `val_acc` / `best_val_accuracy`，实际是测试准确率 | `-best.pt` 按历轮测试准确率选择；`-last.pt` 保存最后一轮 |
| Liquid 粗搜索 | 45,000 张搜索训练样本的 50%，即 22,500 张 | 从官方训练集留出的 5,000 张验证样本；`accuracy` 为最后一轮验证准确率（%） | 每次候选评估重新初始化 |
| Liquid 精搜索 / 最终训练 | 同一批 45,000 张搜索训练样本 | 验证集保持不变；最终训练结束才调用官方测试集评估，写入 `test_accuracy`（%） | 最终 checkpoint 保存最后一轮权重 |

Liquid 的划分受 `--seed`、`--validation-size` 控制，各阶段训练比例可配置；
最终训练沿用 `--refine-train-fraction`，不会自动合并训练集和验证集。
训练 DataLoader 均使用 `drop_last=True`，上表为数据子集大小。

当前 solid 与 liquid 的训练样本数、最佳/最后轮权重选择规则不同。
将当前 `train/train.py` 生成的 solid checkpoint 用于 liquid 初始化时，
该权重已经在完整训练集上训练过，也就见过 liquid 后续划出的验证样本；
若选用 `-best.pt`，还引入了测试集选轮的影响。因此不能将这种设置直接视为
严格独立的验证/测试对照。
正式比较需要统一数据划分、初始化来源、训练预算和权重选择规则，报告多随机种子的
均值与波动，并计入 solid 预训练及拓扑搜索成本；当前代码尚未统一这些评估协议。

## 项目结构

```text
Liquid-Transformer/
├── model-solid/          # DeiT-Tiny、Swin-Tiny 及共享组件
├── model-liquid/         # 两个 Liquid ViT 与可微小世界 FFN
├── dataset/              # CIFAR-100 下载位置与数据加载代码
├── train/evolution/      # NSGA-II、图指标与可视化
├── train/                # solid 训练与双 GPU liquid 进化入口
├── train-log/            # 日志、图表、基因和 checkpoint
└── tests/                # 模型、反向传播、拓扑和 NSGA-II 测试
```

## 模型配置

| 模型 | CIFAR patch/window | Tiny 主干配置 | 分类数 |
| --- | --- | --- | --- |
| DeiT-Tiny | patch 4×4 | dim 192，12 blocks，3 heads | 100 |
| Swin-Tiny | patch 4×4，window 4×4 | dim 96，depths [2,2,6,2]，heads [3,6,12,24] | 100 |

两个模型直接使用 32×32 图像，不将 CIFAR-100 人工放大到 224×224。
DeiT 使用 64 个 patch token 加 1 个分类 token，当前训练入口没有蒸馏教师或蒸馏损失。
Swin 四个 stage 的网格为 8×8、4×4、2×2、1×1，实际窗口边长依次为 4、4、2、1；
当网格边长不大于配置窗口时，代码关闭 shifted window。

## Liquid 小世界 FFN

Liquid 模型将每个 Transformer block 的规则 FFN 替换为：

```text
输入投影 → 隐藏神经元 → 稀疏小世界消息传播 × 2 → 输出投影
```

默认隐藏节点数仍为 `4 × embed_dim`，初始图使用平均度 8 的连通
Watts–Strogatz 网络。图结构由 NSGA-II 交叉和变异，边权与模型连续参数由 AdamW
反向传播训练。两个目标均最大化：候选最后一轮验证准确率，以及各 block
小世界系数的稳定变换：

```text
sigma = (C / C_random) / (L / L_random)
small_world_score = mean(log(1 + sigma))
```

同时记录聚类系数、平均路径长度、Louvain 模块度、社区数量和枢纽节点数量。
指标基于无向、无权的二值拓扑计算，不随训练中的边权变化而更新。
路径长度默认从 64 个源节点采样；随机图参照采用
`C_random = mean_degree / (nodes - 1)`、
`L_random = log(nodes) / log(mean_degree)` 的解析近似，未生成随机对照图集合。
`small_world_score` 是搜索代理指标，分数升高本身不能证明分类能力提升。

每条无向边对应两个可独立训练的有向边权；两轮消息传播共享边权，分别使用
LayerNorm 和可训练残差缩放。实现通过 FP32 的 `torch.sparse.mm` 传播消息。
原有稠密输入/输出投影仍保留，liquid 还增加了边权等参数；“稀疏消息传播”不意味着
总参数量减少，也不构成已验证的加速结论。

## 安装

项目提供 Python 3.12 的 Conda 环境定义，环境名为 `liquid`：

```bash
conda env create -f environment.yml
conda activate liquid
```

`requirements.txt` 未指定 PyTorch wheel 索引，也未锁定依赖版本。GPU 环境需要安装
与本机驱动匹配的 CUDA 版 PyTorch；项目不自动选择 CUDA 版本。

也可以使用 Python 3.10 或更高版本创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

安装后可用下面命令确认 CUDA 可用：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

## Solid 训练

从项目根目录运行统一训练脚本。首次运行会把 CIFAR-100 自动下载到 `dataset/`。
CUDA 可用时默认使用 `--devices 0,1`（`nn.DataParallel`）；CPU 调试可传
`--devices cpu`，单卡必须显式传 `--devices 0`，否则默认请求第二张卡会报错。
`--batch-size` 为全局 batch size。兼容参数 `--device` 会覆盖 `--devices`。

```bash
# Solid DeiT-Tiny；通过参数手动设定 epoch，范围为 1～100（省略时为 100）
python train/train.py --model deit-tiny --epochs 100 --batch-size 128 --devices 0,1

# Solid Swin-Tiny
python train/train.py --model swin-tiny --epochs 100 --batch-size 128 --devices 0,1
```

也可以使用快捷脚本，后面的参数会原样传给统一入口：

```bash
bash train/train_deit_tiny.sh --epochs 100 --batch-size 128 --devices 0,1
bash train/train_swin_tiny.sh --epochs 100 --batch-size 128 --devices 0,1
```

默认训练策略为 AdamW、100 epochs、10 epochs warmup、cosine decay、label smoothing、
RandAugment 和 CUDA FP16 AMP。默认学习率 `5e-4`、最低学习率 `1e-6`、
weight decay `0.05`、label smoothing `0.1`、梯度裁剪 `1.0`、随机种子 `42`。
数据增强还包括随机裁剪、水平翻转和 RandomErasing。完整参数可通过以下命令查看：

```bash
python train/train.py --help
```

日志示例为 `train-log/20260712-120000-swin-tiny-solid.log`。同一次运行的最优与
最近 checkpoint 使用相同前缀，可用 `--resume <checkpoint>` 恢复训练。

## 双 GPU NSGA-II 拓扑进化

可用对应 solid checkpoint 初始化所有候选的兼容参数，包括原有 FFN 的
`fc1` / `fc2`；新增边权、消息缩放和消息 LayerNorm 保持初始化值。
此方式有上述数据划分限制；不传 `--solid-checkpoint` 时从头训练。

```bash
# 两张卡分担同一代中的不同候选，不对单个小模型使用 DDP
python train/evolve.py \
  --model deit-tiny \
  --devices 0,1 \
  --solid-checkpoint train-log/<deit-solid-best.pt> \
  --coarse-epochs 2 \
  --refine-epochs 5 \
  --final-epochs 100

python train/evolve.py \
  --model swin-tiny \
  --devices 0,1 \
  --solid-checkpoint train-log/<swin-solid-best.pt> \
  --coarse-epochs 2 \
  --refine-epochs 5 \
  --final-epochs 100
```

默认多保真预算如下；每个阶段的代数包含初始种群评估，后续各代评估新子代，
再从父代与子代合并的种群中选择幸存者。

| 阶段 | 种群 / 代数 | 每次候选训练 | 训练子集 | 候选评估次数 |
| --- | --- | --- | --- | --- |
| 粗搜索 | 16 / 10 | 2 epochs | 搜索训练集的 50% | 160 |
| 精搜索 | 8 / 10 | 5 epochs | 搜索训练集的 100% | 80 |
| 最终训练 | 2 个入选候选 | 100 epochs | 同精搜索 | 2 |

入选候选按 NSGA-II 的非支配排序和拥挤距离选择；第一前沿不足指定数量时，
会从后续前沿补足，因此不保证所有 finalists 都来自第一 Pareto 前沿。
默认共进行 242 次候选训练，合计 920 个候选 epoch；考虑粗搜索的半量数据后，
约相当于 760 个完整搜索训练集 epoch（忽略 `drop_last`），不含 solid 预训练、
验证和图指标计算。双卡只是并行分担候选，不减少总训练工作量。

粗搜索转精搜索、子代评估和最终训练都会重新创建模型与优化器；传入 checkpoint
时重载同一份 solid 权重，否则重新随机初始化，不继承已训练候选的权重。
训练使用 AdamW、无 warmup 的 cosine 调度，CUDA 下使用 BF16 autocast，
稀疏传播仍使用 FP32；这与 solid 的 warmup + FP16 AMP 不同。

`--coarse-epochs`、`--refine-epochs` 为 1～100；`--final-epochs` 为 0～100，
设为 0 可跳过最终训练与测试。单卡使用 `--devices 0`；CPU 调试使用 `--cpu`。
进化入口默认不下载数据，首次运行需加 `--download`，或先准备好 CIFAR-100。

快速验证完整流程时可缩小预算：

```bash
python train/evolve.py \
  --model deit-tiny --devices 0,1 --download \
  --coarse-population 4 --coarse-generations 2 --coarse-epochs 1 \
  --refine-population 2 --refine-generations 1 --refine-epochs 1 \
  --finalists 1 --final-epochs 1
```

该缩小预算示例仍使用默认训练数据比例，不是少量样本的单元测试。
完整参数可通过 `python train/evolve.py --help` 查看。

### 训练记录与可视化

每次搜索创建 `train-log/YYYYMMDD-HHMMSS-模型-liquid-nsga/`，持续输出：

- TensorBoard：每代 accuracy、loss、small-world score、聚类、路径和模块度；
- 候选 TensorBoard/JSONL：训练期间逐 epoch 实时写入，无需等待一代结束；
- Pareto PNG：验证准确率与小世界评分散点图；
- 训练 PNG：每代最佳准确率候选的 train/validation accuracy 与 loss；
- 拓扑 PNG：首层、中层和末层的社区压缩图，节点大小表示社区规模；
- block 热力图：逐代展示聚类、路径、sigma、模块度、社区数和边变化率；
- `config.json`：数据划分、初始化来源和搜索配置；
- `population.csv`、`evolution.jsonl`：每代选择后的幸存个体记录，可能重复出现同一候选；
- `genomes/`：每代 Pareto 前沿的完整边集合；
- `state.json`：最近一代的完整种群状态；
- `finalists.json` 与 `checkpoints/`：最终验证/测试结果及模型权重。

`state.json` 仅保存状态，当前进化入口没有 `--resume`，也不读取此文件来恢复搜索；
`--run-dir` 必须指向尚不存在的目录。跳过最终训练时，`finalists.json` 没有测试指标。
逐候选记录和统计口径详见 [训练日志说明](../train-log/README.md)。

查看实时曲线：

```bash
tensorboard --logdir train-log --port 6006
```

## 测试

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

2026-09-10 本地检查：现有 9 项测试通过，Ruff 检查通过；测试产生一条 PyTorch
稀疏张量 invariant 提示。solid 的 `--help` 可正常运行。
这次检查使用现有 Python 3.13.5 环境，不是新建的 `liquid` 环境，也未运行完整训练。
现有 TensorBoard 2.11.2 低于项目要求的 `>=2.16`，与 protobuf 6.33.6 组合时，
进化入口导入失败（`Descriptors cannot be created directly`）。以下临时设置已验证
可让帮助命令运行，但不代表完整训练已验证；复现应先安装项目声明的依赖。

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python train/evolve.py --help
```

## License

本项目采用 [MIT License](../LICENSE)。
