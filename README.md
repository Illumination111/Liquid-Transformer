# Liquid-Transformer

面向 CIFAR-100 的 **PPO 小世界拓扑搜索 + LIF FFN DeiT-Tiny**。
当前版本只将 FFN 改为多时间步脉冲计算，Attention、patch embedding、位置编码、
残差与分类读出保留连续值，因此是混合 ANN/SNN Transformer。

新搜索入口为 `train/search.py`。原 NSGA-II 入口和连续值 DeiT/Swin 保留作历史对照，
不参与 PPO 流程，旧协议见 [历史说明](docs/legacy-nsga.md)。

## 当前状态

本轮完成代码改造与功能验证，未启动 CIFAR-100 训练或完整搜索实验。
合成数据检查中的准确率只用于验证流程，不能当作分类性能、拓扑收益或能耗结果。
测试范围和环境记录见 [验证记录](docs/validation.md)。
当前 30 项测试与 Ruff 检查通过，并通过 A100 上的合成数据 GPU 集成检查。

## 模型：仅 FFN 使用 LIF

默认输入为 32×32，patch 4×4，主干保持 DeiT-Tiny 的 dim=192、12 blocks、3 heads，
64 个 patch token 加 1 个分类 token，输出 100 类 logits。
静态图像的 patch 特征重复到 `T=4` 个时间步，Attention 在每个时间步独立计算。
最后对各时间步归一化后的分类 token 取平均，再做线性分类。

```text
图像 → 连续值 patch/位置编码 → 重复 T 个时间步
     → [连续值 Attention + residual → LIF 小世界 FFN + residual] × 12
     → LayerNorm → 分类 token 时间平均 → 连续值分类头

FFN：Linear(d, 4d) → LayerNorm → LIF
     → [FP32 稀疏消息传播 → LayerNorm → LIF → 缩放残差] × 2
     → Linear(4d, d)
```

LIF 对 `[time, batch, tokens, channels]` 的时间维积分：

```text
u[t] = beta * v[t-1] + current[t]
s[t] = H(u[t] - threshold)
v[t] = u[t] - stop_gradient(s[t]) * threshold
```

默认 `beta=0.5`、阈值 `1.0`，硬阈值产生 0/1 脉冲；反向使用
`1 / (1 + slope * abs(u - threshold))²` 的替代梯度，默认 `slope=5`。
reset 的脉冲分支停止梯度，膜电位积分保留跨时间步梯度。
膜电位是每次 forward 的局部变量，batch 之间不会继承状态，checkpoint 也不保存膜电位。

小世界图默认每层 768 个隐藏节点、平均度 8，图保持连通；每条无向边对应两个独立
可训练有向边权。消息传播轮次与仿真时间步是两个独立维度。
LIF 输出是脉冲，但 FFN 的消息残差与输出投影仍可为连续值。该实现不等同于
全脉冲 Attention，也未宣称事件驱动硬件加速或能耗下降。

## 搜索：PPO 替换主线 NSGA-II

策略为带价值头的两层 MLP；每次动作同时为每个 block 选择一档重连比例：
`0 / 1% / 5% / 10%`。0 表示保持原图，其余档位按比例尝试重连，保留总边数和连通性；
受重连约束与尝试次数限制，实际变化边数可能小于目标。

| 项目 | 实现 |
| --- | --- |
| 观测 | 当前验证准确率、剩余步数比例，以及各层聚类系数、归一化路径长度与 log-sigma |
| 质量分数 | `Q = validation_accuracy / 100 + topology_weight * small_world_score`，权重默认 0.01 |
| 奖励 | `Q(新候选) - Q(当前候选)` |
| 优化 | PPO clipped objective + GAE + value loss + entropy bonus；默认 clip=0.2、gamma=0.99、lambda=0.95 |
| 轨迹 | 每次 iteration 从相同初始图重新开始，依次执行 `rollout-steps` 次动作；末步为终止状态 |
| 候选初始化 | 每次用同一训练种子重新构建模型、优化器和数据加载器，不继承上一个候选的权重 |
| 最终拓扑 | 从初始图及所有已评估候选中选择质量分数最高者；相同时保留较早候选 |

策略采样与拓扑重连使用各自的随机数生成器，避免候选训练重设种子后反复采到相同动作。
PPO 优化一个加权标量目标，不输出 NSGA-II 的 Pareto 前沿。
观测是图统计摘要，不包含完整邻接矩阵；目前属于有限观测的拓扑策略原型。
奖励权重、搜索效率和分类收益都需要后续真实实验评估。

小世界评分沿用 `mean(log(1 + sigma))`；sigma 基于无权二值图计算，路径长度默认
采样 64 个源节点，随机图参照为解析近似。该评分不衡量已训练边权的功能或分类能力。

## Conda 环境与数据

从项目根目录执行：

```bash
conda env create -f environment.yml
conda activate liquid-transformer
pip install -r requirements-dev.txt
```

环境名为 `liquid-transformer`，Python 3.12。依赖安装在该 Conda 环境内，
`requirements.txt` 保留版本下限；本次实际版本见验证记录。GPU 使用需匹配本机驱动的
PyTorch wheel，可用以下命令检查：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

所有训练入口默认数据根目录为 `~/dataset`，本机即 `/data/user25101366/dataset`。
可用 `LIQUID_DATA_DIR` 或显式 `--data-dir` 覆盖。下载与训练分离：

```bash
python -m dataset.download --data-dir /data/user25101366/dataset
```

下载器使用 torchvision 的官方 CIFAR-100 下载与完整性校验。数据位于
`/data/user25101366/dataset/cifar-100-python/`，不放进 Git 仓库。

## 功能验证与后续运行

只运行功能测试：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 pytest -q
ruff check .
```

不下载、也不读取真实数据的端到端检查：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python train/search.py --smoke-test --device cpu
```

`--smoke-test` 强制使用 2 blocks、dim=24、2 时间步、1 轮消息传播、合成的
8/4/4 张训练/验证/测试图像。流程包含初始候选、2 次拓扑动作、一次 PPO 更新和
最终候选的保存、重载与评估；产物标记 `synthetic: true`。

下面是后续真实搜索的运行方式，本轮未执行：

```bash
python train/search.py \
  --device cuda:0 --data-dir /data/user25101366/dataset \
  --iterations 4 --rollout-steps 4 --candidate-epochs 2 \
  --time-steps 4 --final-epochs 0
```

这组参数训练 1 个初始候选及 16 个动作候选，共 34 个候选 epoch。
`--iterations 0` 可用于仅评估初始固定拓扑。需要最终重训和测试时显式设置
`--final-epochs 1..100`；默认 0，只搜索。当前候选训练使用单设备 FP32 AdamW、
无 warmup 的 cosine 调度、label smoothing 和梯度裁剪，不自动占用多卡。
`bash train/search_deit_tiny.sh ...` 等价于上述新入口；完整参数见 `--help`。

### 数据与权重选择

CIFAR-100 官方训练集按种子划分为默认 45,000 张训练样本和 5,000 张验证样本。
`--train-fraction` 仅缩减训练子集；所有候选与最终训练使用相同划分和比例。
PPO 的 `accuracy` 是候选训练期间最高验证准确率，`loss` 为对应轮的验证 loss；
最终重训也保存最佳验证轮，并只在该权重与拓扑固定后加载官方测试集评估一次。
训练使用 `drop_last=True`；子集小于 batch size 时立即报错。

新入口不加载旧 solid checkpoint，避免旧权重提前见过搜索验证样本。
原 solid 入口仍每轮评估官方测试集，不能直接作为严格匹配的泛化对照。
后续对照实验应统一划分、训练预算与选择规则，并报告多种子结果及完整搜索成本。

### 输出与恢复范围

每次运行创建独立的 `train-log/*-deit-snn-ppo/`；也可指定尚不存在的 `--run-dir`。

| 文件 | 含义 |
| --- | --- |
| `config.json` | 有效参数、模型设置、代码提交/dirty 标记、合成数据标记 |
| `candidates.jsonl` | 初始图及每次动作候选的验证结果、训练历史、拓扑指标与耗时 |
| `transitions.jsonl` | PPO 观测、各层动作、奖励及对应候选 ID |
| `ppo.jsonl` | 每次策略更新的策略损失、价值损失、熵与平均奖励 |
| `policy.pt` | 策略、优化器、策略采样 RNG 与配置；当前 CLI 尚不支持恢复搜索 |
| `selected.json` | 按质量分数选中的候选与完整拓扑 |
| `final.json` / `checkpoints/final-best.pt` | 显式最终训练后的指标及可重建 SNN 模型的权重、拓扑、LIF 配置 |

日志和权重默认被 Git 忽略。PPO 暂未接入旧 NSGA-II 的 Pareto 图与 TensorBoard 面板。
产物详情见 [日志说明](train-log/README.md)。

## 代码位置

```text
model-liquid/lif.py             # LIF 积分与替代梯度
model-liquid/deit_tiny_snn.py    # 连续值 Attention + LIF 小世界 FFN
train/search.py                # PPO 搜索入口
train/rl/                      # 策略、图动作与候选评估
dataset/                       # 数据加载与独立下载命令
tests/                         # 新功能验证及旧实现回归
model-solid/                   # 原始 DeiT/Swin 对照模型
train/evolve.py                # 历史 NSGA-II 入口
```

算法背景：[PPO 原论文](https://arxiv.org/abs/1707.06347)。
本项目的 FFN-only LIF 混合模型不是某篇全脉冲 Transformer 论文的复现。

本项目采用 [MIT License](LICENSE)。
