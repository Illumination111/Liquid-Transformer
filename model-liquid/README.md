# Liquid model architecture

当前主线为 `SpikingDeiTTiny`（`deit_tiny_snn.py`）：只将 FFN 改为多时间步 LIF，
Attention、输入投影与读出保留连续值。神经元实现位于 `lif.py`，膜电位在每次 forward
内部初始化，通过时间反向传播训练；不在 batch 之间共享状态。

```text
static image → analog patches repeated over T steps
  → [analog attention + residual → LIF small-world FFN + residual] × depth
  → temporal average of class features → analog classifier
```

FFN 隐藏层使用 `LayerNorm → LIF` 替代 GELU；每轮稀疏消息传播后也使用 LIF。
输入/输出稠密投影与消息残差保持连续值。PPO 调整各层二值边集合，AdamW 训练
网络参数与有向边权；详见 [主 README](../README.md)。

## 历史连续值模型

`LiquidDeiTTiny` 与 `LiquidSwinTiny` 保留各自的 Attention、patch embedding 和层级结构，
只将每个 Transformer block 的规则 FFN 替换为可进化的小世界 FFN：

```text
x → Linear(d, 4d) → GELU → sparse small-world message passing × T
  → Linear(4d, d) → Transformer residual
```

图的二值边集合在单个候选训练期间保持固定，由 NSGA-II 在候选之间执行交叉、重连和
变异；边权、消息缩放、LayerNorm、输入/输出投影以及模型其余参数均通过反向传播训练。

默认图为平均度 8 的连通 Watts–Strogatz 网络，包含两轮稀疏消息传播。实现使用
`torch.sparse.mm`，不会显式创建 `[batch, token, edge]` 消息张量。
