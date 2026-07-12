# 训练日志

训练日志命名格式为 `YYYYMMDD-HHMMSS-模型-solid.log`，例如：

```text
20260712-120000-deit-tiny-solid.log
20260712-120000-deit-tiny-solid-best.pt
20260712-120000-deit-tiny-solid-last.pt
```

日志文件与模型 checkpoint 默认不提交到 Git。

液态拓扑搜索使用目录 `YYYYMMDD-HHMMSS-模型-liquid-nsga/`，其中包含：

- `evolution.log`：完整文本日志；
- `population.csv` 与 `evolution.jsonl`：每代个体指标；
- `tensorboard/`：accuracy、loss、小世界评分和拓扑指标；
- `candidates/*/tensorboard/`：训练过程中实时写入的逐 epoch 候选曲线；
- `figures/pareto-*.png`：准确率与小世界评分的 Pareto 分布；
- `figures/training/`：每代最佳准确率候选的训练/验证 accuracy 与 loss；
- `figures/topology/`：社区级拓扑演化快照；
- `figures/block-topology-evolution.png`：各 block 的拓扑指标与边变化率热力图；
- `genomes/`：每代 Pareto 前沿的完整图基因；
- `checkpoints/`：最终候选权重。
