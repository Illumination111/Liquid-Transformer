# Liquid-Transformer

面向 CIFAR-100 的模块化 Vision Transformer 训练工程，提供 **DeiT-Tiny**、
**Swin-Tiny** solid 基线，以及用 NSGA-II 协同进化准确率和小世界拓扑的 liquid 模型。

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

## Liquid 小世界 FFN

Liquid 模型将每个 Transformer block 的规则 FFN 替换为：

```text
输入投影 → 隐藏神经元 → 稀疏小世界消息传播 × 2 → 输出投影
```

默认隐藏节点数仍为 `4 × embed_dim`，初始图使用平均度 8 的连通
Watts–Strogatz 网络。图结构由 NSGA-II 交叉和变异，边权与模型连续参数由 AdamW
反向传播训练。第二优化目标为各 block 小世界系数的稳定变换：

```text
sigma = (C / C_random) / (L / L_random)
small_world_score = mean(log(1 + sigma))
```

同时记录聚类系数、平均路径长度、Louvain 模块度、社区数量和枢纽节点数量。

## 安装

项目提供 Python 3.12 的 Conda 环境定义，环境名为 `liquid`：

```bash
conda env create -f environment.yml
conda activate liquid
```

`requirements.txt` 默认可能装到 CPU 版 PyTorch。若要用 GPU，请改装 CUDA 版
（Driver 支持 CUDA 12.8+ 时可用 `cu128`；也可按
[PyTorch 官网](https://pytorch.org/get-started/locally/)选择对应索引）：

```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

也可以使用 Python 3.10 或更高版本创建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

安装后可用下面命令确认 CUDA 可用：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

## Solid 训练

从项目根目录运行统一训练脚本。首次运行会把 CIFAR-100 自动下载到 `dataset/`。
CUDA 可用时默认使用 `--devices 0,1`（`nn.DataParallel`）；无 CPU 调试可传
`--devices cpu`，单卡可传 `--devices 0`。

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
RandAugment 和 CUDA AMP。完整参数可通过以下命令查看：

```bash
python train/train.py --help
```

日志示例为 `train-log/20260712-120000-swin-tiny-solid.log`。同一次运行的最优与
最近 checkpoint 使用相同前缀，可用 `--resume <checkpoint>` 恢复训练。

## 双 GPU NSGA-II 拓扑进化

建议先训练对应 solid baseline，随后把最优 checkpoint 作为所有候选的共同初始化，
以降低短训练准确率的噪声：

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

默认采用多保真搜索：粗搜索为 16 个个体、10 代、每个候选 2 epochs、50% 训练子集；
精搜索为 8 个个体、10 代、每个候选 5 epochs、完整训练集；最终选择 2 个 Pareto
候选分别完整训练最多 100 epochs。`--coarse-epochs`、`--refine-epochs` 和
`--final-epochs` 都可手动设置，但均不得超过 100。CIFAR-100 原训练集固定划分为
45,000 个搜索训练样本和 5,000 个验证样本，官方测试集只在最终候选训练完成后使用。

快速验证完整流程时可缩小预算：

```bash
python train/evolve.py \
  --model deit-tiny --devices 0,1 \
  --coarse-population 4 --coarse-generations 2 --coarse-epochs 1 \
  --refine-population 2 --refine-generations 1 --refine-epochs 1 \
  --finalists 1 --final-epochs 1
```

### 训练记录与可视化

每次搜索创建 `train-log/YYYYMMDD-HHMMSS-模型-liquid-nsga/`，持续输出：

- TensorBoard：每代 accuracy、loss、small-world score、聚类、路径和模块度；
- 候选 TensorBoard/JSONL：训练期间逐 epoch 实时写入，无需等待一代结束；
- Pareto PNG：验证准确率与小世界评分散点图；
- 训练 PNG：每代最佳准确率候选的 train/validation accuracy 与 loss；
- 拓扑 PNG：首层、中层和末层的社区压缩图，节点大小表示社区规模；
- block 热力图：逐代展示聚类、路径、sigma、模块度、社区数和边变化率；
- `population.csv`、`evolution.jsonl`：可用于后续统计分析的逐个体记录；
- `genomes/`：每代 Pareto 前沿的完整边集合；
- `state.json`：最近一代的完整种群状态；
- `finalists.json` 与 `checkpoints/`：最终验证/测试结果及模型权重。

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

## License

本项目采用 [MIT License](LICENSE)。
