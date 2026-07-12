# CIFAR-100 数据目录

统一训练脚本首次运行时会通过 `torchvision` 自动下载并校验 CIFAR-100，解压后的数据位于：

```text
dataset/cifar-100-python/
```

原始数据约 169 MB，不纳入 Git 版本控制。若训练节点不能访问网络，可手动将官方
`cifar-100-python.tar.gz` 放在本目录，或将已解压的 `cifar-100-python` 目录复制到这里。
