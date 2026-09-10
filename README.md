# Liquid-Transformer

基于 DeiT-Tiny 的混合 ANN/SNN Transformer，使用多时间步 LIF 神经元构建小世界 FFN，
并通过 PPO 搜索网络拓扑。面向 CIFAR-100，Attention 与输入、输出投影保留连续值计算。

## 特性

- 保留 DeiT-Tiny 主干，支持原生 32×32 图像。
- 基于 [SpikingJelly](https://github.com/fangwei123456/spikingjelly) 的 LIF FFN，支持替代梯度训练与可学习稀疏边权。
- PPO 按层调整拓扑，保持图的连通性和边数。
- 提供连续值 DeiT/Swin 基线、历史 NSGA-II 实现及合成数据检查。

## 安装

```bash
git clone https://github.com/Illumination111/Liquid-Transformer.git
cd Liquid-Transformer
conda env create -f environment.yml
conda activate liquid-transformer
```

环境使用 Python 3.12。GPU 运行需要与驱动兼容的 CUDA 版 PyTorch。

## 快速开始

下载 CIFAR-100，默认保存到 `~/dataset`：

```bash
python -m dataset.download
```

运行 PPO 拓扑搜索：

```bash
python train/search.py --device cuda:0
```

可通过 `--data-dir` 指定数据目录。默认只搜索；使用 `--final-epochs 100` 开启
最终训练与测试。候选使用训练集划出的验证集评分，官方测试集仅用于最终评估。
日志与 checkpoint 保存到 `train-log/`。

使用合成数据快速检查流程，无需下载数据：

```bash
python train/search.py --smoke-test --device cpu
```

完整参数见 `python train/search.py --help`。

## 项目结构

```text
model-liquid/    # LIF FFN、小世界拓扑与混合 DeiT
model-solid/     # 连续值 DeiT/Swin 基线
train/           # 训练入口与 PPO 搜索
dataset/         # 数据下载与加载
tests/           # 单元测试与集成测试
train-log/       # 运行产物（默认不提交）
```

## 开发与测试

```bash
pip install -r requirements-dev.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 pytest -q
ruff check .
```

## License

[MIT](LICENSE)
