# Liquid model architecture

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
