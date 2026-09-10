# CIFAR-100 数据目录

数据存放在仓库外。默认根目录为 `~/dataset`，本机为
`/data/user25101366/dataset`；`LIQUID_DATA_DIR` 或 `--data-dir` 可覆盖默认值。
在项目根目录使用 `liquid-transformer` Conda 环境独立下载与校验：

```bash
python -m dataset.download --data-dir /data/user25101366/dataset
```

解压后的数据位于：

```text
/data/user25101366/dataset/cifar-100-python/
```

原始数据约 169 MB，不纳入 Git 版本控制。若训练节点不能访问网络，可手动将官方
`cifar-100-python.tar.gz` 放在数据根目录，或将已解压的 `cifar-100-python` 目录复制过去。
PPO 搜索只构建训练/验证数据加载器，最终权重选定后才加载官方测试集。
