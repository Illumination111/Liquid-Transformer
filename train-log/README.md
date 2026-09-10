# 训练日志

训练日志和模型权重默认保存在本目录，运行产物不提交到 Git。

## 当前 PPO / LIF 记录

新入口 `train/search.py` 默认创建 `*-deit-snn-ppo/`：

- `config.json`：有效参数、模型/LIF 配置、代码版本与 dirty 标记、`synthetic` 标记；
- `candidates.jsonl`：所有搜索候选，包括初始图、未入选候选、最佳验证轮和训练历史；
- `transitions.jsonl`：观测、动作、奖励、iteration/step 与候选 ID；
- `ppo.jsonl`：策略更新损失、价值损失、熵和平均奖励；
- `policy.pt`：策略、优化器及策略 RNG 快照；当前无恢复搜索入口；
- `selected.json`：质量分数最高候选及完整二值拓扑；
- `final.json`：显式最终训练的验证/测试结果；
- `checkpoints/final-best.pt`：按验证准确率选出的权重、拓扑与完整模型构建参数。

PPO 记录中的 `accuracy` 为最佳验证轮准确率（%），`loss` 为同轮 loss。
`test_accuracy` 只出现在显式最终评估的 `final.json`。
`--smoke-test` 的产物均由有效配置及汇总文件中的 `synthetic: true` 标识。
运行子目录与权重默认不入 Git。

## 历史 solid / NSGA-II 记录

训练日志命名格式为 `YYYYMMDD-HHMMSS-模型-solid.log`，例如：

```text
20260712-120000-deit-tiny-solid.log
20260712-120000-deit-tiny-solid-best.pt
20260712-120000-deit-tiny-solid-last.pt
```

日志文件与模型 checkpoint 默认不提交到 Git。
solid 日志中的 `val_acc` / `best_val_accuracy` 来自官方测试集，
`-best.pt` 按历轮测试准确率选择，`-last.pt` 为最后一轮权重。

液态拓扑搜索使用目录 `YYYYMMDD-HHMMSS-模型-liquid-nsga/`，其中包含：

- `evolution.log`：完整文本日志；
- `config.json`：运行配置、数据划分种子、预算与 solid checkpoint 路径；
- `population.csv` 与 `evolution.jsonl`：每代选择后的幸存个体指标；
- `tensorboard/`：accuracy、loss、小世界评分和拓扑指标；
- `candidates/*/tensorboard/`：训练过程中实时写入的逐 epoch 候选曲线；
- `candidates/<阶段-代数>-<候选ID>-e<epochs>/epochs.jsonl`：每轮训练与验证的
  loss、accuracy、学习率、耗时和峰值显存，包括未幸存候选的记录；
- `figures/pareto-*.png`：准确率与小世界评分的 Pareto 分布；
- `figures/training/`：每代最佳准确率候选的训练/验证 accuracy 与 loss；
- `figures/topology/`：社区级拓扑演化快照；
- `figures/block-topology-evolution.png`：各 block 的拓扑指标与边变化率热力图；
- `genomes/`：每代 Pareto 前沿的完整图基因；
- `state.json`：最近一代搜索种群状态；当前入口尚不支持恢复搜索；
- `finalists.json`：入选候选的拓扑、验证结果、训练历史，完成最终训练时还含
  `test_accuracy` 和 `test_loss`；
- `checkpoints/`：最终候选最后一轮权重及拓扑；`--final-epochs 0` 时不生成。

## 分析口径

Liquid 的 `accuracy` 与 `validation_accuracy` 是验证准确率，`test_accuracy`
才是官方测试准确率，数值单位均为百分比。loss 使用带 label smoothing 的交叉熵。
最终训练仍使用搜索训练子集，保存最后一轮权重，未按验证集最佳轮次回滚。

`population.csv` / `evolution.jsonl` 记录幸存种群，不是所有已评估候选。
同一候选可能连续存活多代，因此不能直接将这些行的 `runtime_seconds` 求和
作为搜索总耗时。总墙钟时间应取 `evolution.log` 的完成记录；
候选成本分析应结合完整日志、候选目录及各阶段重新评估的记录。
`peak_memory_gb` 为候选所在设备上 PyTorch 的峰值已分配显存（以 1024³ 字节换算），
不是整机显存或驱动报告的占用。

比较 Pareto 前沿时应在相同 phase、训练预算与数据比例下进行，粗搜索切换到
精搜索或最终训练后的精度变化不能全部归因于拓扑进化。block 热力图追踪每代
准确率最佳候选；边变化率比较相邻代最佳候选的边集合，不一定是同一个体的变异轨迹。

后续归档结果时，建议同时保留运行配置、代码提交、种子、初始化来源、最佳/最后轮
选择规则，以及每个最终候选的验证精度、测试精度、拓扑评分、耗时和显存。
搜索子目录整体被 `.gitignore` 忽略；需要公开实验结论时，应另行提交可追溯的
汇总及必要证据文件。
