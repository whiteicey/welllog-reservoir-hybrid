# 测井储层参数预测：纯深度学习

`deep_learning` 分支记录仅使用神经网络预测逐深度 **POR、PERM、SW** 的方法、复现入口和实验路径。历史树–序列混合方案见 [README.hybrid.md](README.hybrid.md)，其中的 A 榜成绩不属于本分支纯 DL 方法。

**当前最佳完整五折 OOF：81.713233**，此前为 81.502720；目标 82 尚未达成。最后的后验路由第五折为 **80.250894**，未过新门槛 80.4。按用户要求，训练已结束，不再自动探索。测试井无标签，不把提交预测视为精度验证。

## 从这里开始

| 内容 | 入口 |
|---|---|
| 方法、公式、特征、损失与解码 | [METHOD.md](docs/deep_learning/METHOD.md) |
| 有效路径、负结果和最后一项实验 | [EXPERIMENTS.md](docs/deep_learning/EXPERIMENTS.md) |
| 配置数据路径并复现固定第五折 | [训练入口](tools/deep_learning/README.md) |
| 独立推理源码与权重放置方式 | [package_dl](package_dl/README.md) |
| 五折结果、哈希与精简实验索引 | [results/deep_learning](results/deep_learning/README.md) |
| 已完成事项及当前运行状态 | [PROJECTS.md](PROJECTS.md) |

## 方法概览

从 GR、PE、SP、CAL、AC、DEN、CNL、RXO、RT 九条曲线构造固定 206 维特征。两个多尺度一维 U-Net 分别读取完整特征和 68 维精简特征；精简分支使用井内相对特征，对 31 条数值特征进行逐点尺度与偏移校正。

骨干结合局部点注意力和全局 patch 余弦注意力。每个输入窗口 256 个深度点，监督与输出中心 128 点。每个目标预测默认状态概率与连续混合分布，最后按官方容差评分的期望收益选取预测值。五折推理等权融合完整后验，不能先解码再平均数值。

训练使用默认状态 BCE 与非默认连续 NLL，辅助监督逐步退场。覆盖所有训练点的全局表对持续冲突的疑似坏样本降低最终损失权重，保留恢复机制；单凭损失大不把困难样本判为异常。方法不使用树预测、预训练权重、伪标签、测试标签或事后拟合的 OOF 融合。

## 结果与口径

| 固定折 | 最佳轮次 | 验证分数 |
|---:|---:|---:|
| 1 | 55 | 82.599332 |
| 2 | 61 | 81.295686 |
| 3 | 67 | 82.782849 |
| 4 | 58 | 81.685044 |
| 5 | 52 | 80.200358 |
| 完整 OOF | 各折最佳 checkpoint | **81.713233** |

80 口训练井采用固定 GroupKFold 五折，共 723,556 个三目标有效行。OOF 总分由完整预测按原评分器计算，不以某一折代替。验证集参与最佳 epoch 选择，反复使用第五折也存在选择偏差，不能视为未参与开发的测试成绩。

最佳交付对应冻结的 75 轮、patience=18 实验；之后的探索为上限 80 轮、patience=18。第五折晋级门槛后来提高到严格大于 80.4，不能用历史 80.2 的晋级记录宣称达到新门槛。原始复现脚本保留历史配置，不静默改写已记录实验。

路由机制的同预算对照：单侧校准 80.095785 → 输入路由 80.171202；在输入路由上分别尝试局部上下文（80.221861）或后验描述量（80.250894），两者没有叠加。后验路由读取默认概率、分量权重熵、条件均值和总标准差，不直接把自信视为正确。这条路线有小幅收益，但尚未过新门槛，也不是多随机种子的稳定增益证明。更多结果见实验路径文档。

## 推理

本分支**不上传原始数据、模型权重、逐行标签/预测或提交 ZIP**。先把另行取得的五个 `fold_1.pt` 至 `fold_5.pt` 放入 `package_dl/models/`，再执行：

```bash
python -m pip install -r package_dl/requirements.txt
python package_dl/predict.py --data_dir /path/to/data --output /path/to/result.json --device cuda
```

输入可以是包含 `test/` 的数据根目录，或直接包含井 TXT 的目录。输出为严格的 `resultData → logId/predictions → DEPTH/POR/PERM/SW` JSON，并生成只含根目录 `result.json` 的 `result.zip`。没有模型文件时不能仅凭 Git 克隆执行推理。

已交付版本在 Windows、normal 环境的 GPU PyTorch 上从新解压目录运行两次：10 口井、95,948 行，JSON 字节完全一致，峰值显存约 401 MiB。仅核验运行、结构与数值范围，不说明测试精度；Linux 与本版本 CPU 推理未实测。

## 仓库布局

```text
docs/deep_learning/       方法和精简实验路径
tools/deep_learning/      可配置数据路径的训练入口
package_dl/              独立纯 DL 推理源码（不附权重）
results/deep_learning/   精简分数、权重哈希与核验证据
experiments/pure_dl/     冻结的历史实验源码；由入口和文档定位
experiments/simple_cnn/  早期神经网络基线源码
src/, package/, history/ 历史混合方案，参见 README.hybrid.md
outputs/                本地实验产物，Git 忽略
```

本地数据、模型、历史记录及各版本交付目录保留在原位置；新版交付单独建目录，不覆盖既有版本。用户已要求最后一次训练结束后停止，不应根据本 README 自动启动实验或续训。
