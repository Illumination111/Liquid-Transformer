# Liquid-Transformer

面向 CIFAR-100 的模块化 Vision Transformer 训练工程，提供从零实现的
**DeiT-Tiny** 和 **Swin-Tiny** solid 基线，以及统一、可复现的训练入口。

## 项目结构

```text
Liquid-Transformer/
├── model-solid/          # DeiT-Tiny、Swin-Tiny 及共享组件
├── model-liqiud/         # 预留的 liquid 模型空目录
├── dataset/              # CIFAR-100 下载位置与数据加载代码
├── train/                # 统一训练脚本和快捷启动脚本
├── train-log/            # 按“时间-模型-solid”命名的日志与 checkpoint
└── tests/                # 两个模型的前向传播烟雾测试
```

> `model-liqiud` 沿用项目要求中的目录拼写。

## 模型配置

| 模型 | CIFAR patch/window | Tiny 主干配置 | 分类数 |
| --- | --- | --- | --- |
| DeiT-Tiny | patch 4×4 | dim 192，12 blocks，3 heads | 100 |
| Swin-Tiny | patch 4×4，window 4×4 | dim 96，depths [2,2,6,2]，heads [3,6,12,24] | 100 |

两个模型直接使用 32×32 图像，不将 CIFAR-100 人工放大到 224×224。

## 安装

项目提供 Python 3.12 的 Conda 环境定义（环境名按要求为 `liqiud`）：

```bash
conda env create -f environment.yml
conda activate liqiud
```

也可以使用 Python 3.10 或更高版本创建虚拟环境，并按机器的 CUDA 环境从
[PyTorch 官网](https://pytorch.org/get-started/locally/)安装 PyTorch：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 训练

从项目根目录运行统一训练脚本。首次运行会把 CIFAR-100 自动下载到 `dataset/`：

```bash
# DeiT-Tiny
python train/train.py --model deit-tiny

# Swin-Tiny
python train/train.py --model swin-tiny
```

也可以使用快捷脚本，后面的参数会原样传给统一入口：

```bash
bash train/train_deit_tiny.sh --epochs 300 --batch-size 128
bash train/train_swin_tiny.sh --epochs 300 --batch-size 128
```

默认训练策略为 AdamW、300 epochs、10 epochs warmup、cosine decay、label smoothing、
RandAugment 和 CUDA AMP。完整参数可通过以下命令查看：

```bash
python train/train.py --help
```

日志示例为 `train-log/20260712-120000-swin-tiny-solid.log`。同一次运行的最优与
最近 checkpoint 使用相同前缀，可用 `--resume <checkpoint>` 恢复训练。

## 测试

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

## License

本项目采用 [MIT License](LICENSE)。
