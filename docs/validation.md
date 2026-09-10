# PPO + LIF FFN 改造验证

本记录对应 2026-09-10 的功能改造，不是 CIFAR-100 实验报告。
模型范围为连续值 Attention + LIF FFN，RL 算法为 PPO。

环境使用 Conda `liquid-transformer`（Python 3.12），通过 `requirements-dev.txt`
安装运行与测试依赖。实际安装版本为 PyTorch 2.14.0+cu130、torchvision 0.29.0+cu130、
TensorBoard 2.21.0、protobuf 7.36.1、NumPy 2.5.3、NetworkX 3.6.1、
pytest 9.1.1、Ruff 0.16.6。

## 检查结果

30 项 pytest 用例通过，Ruff 检查通过。另在 `cuda:3`（NVIDIA A100-SXM4-80GB）
使用 `--smoke-test` 完成合成数据的候选训练、PPO 更新、最终 checkpoint 保存/重载
与评估。GPU 运行记录保存在本地 `train-log/20260910-lif-ppo-cuda-smoke/`，未提交权重。

CIFAR-100 已下载、解压并校验到 `/data/user25101366/dataset`，
训练集 50,000 张、测试集 10,000 张；压缩包 MD5 为
`eb9058c3a382ffc7106e4002c42a8d85`。此次只准备数据，没有对真实样本进行训练或评估。

## 验证范围

- LIF 的积分、泄漏、阈值、reset、跨时间步梯度及非法参数检查。
- 混合 DeiT 的输出形状、时间步响应、batch 状态隔离；稠密投影、边权、Attention
  与输入投影均获得有限且非零的梯度；Attention 中没有 LIF。
- 默认 12 层、192 维 DeiT 前向检查。
- PPO 的 GAE 终止/引导值处理、奖励动作的概率更新、独立采样随机数流。
- 重连的确定性、固定边数与连通性；数据划分的互斥、复现与测试集隔离。
- 合成数据的完整候选评估 → 图动作 → PPO 更新 → 最终权重保存/重载/评估。
- 最佳验证轮 checkpoint 选择和关闭最终训练时不加载测试集。
- 原连续值模型及 NSGA-II 的既有回归用例。

运行命令：

```bash
conda activate liquid-transformer
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 pytest -q
ruff check .
```

尚未运行真实数据训练、多随机种子对照或完整拓扑搜索，不能据此推断准确率提升、
搜索效率或能耗收益。合成数据使用缩小模型和随机标签，其指标不是实验结果。
