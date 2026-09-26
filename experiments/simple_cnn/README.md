# 简单一维 CNN 五折基线

目标：只测 80 口训练井的按井五折交叉验证水平，不读取 test。

模型：13 个原始输入（GR/RXO/RT 做 log1p）按训练折中位数和 IQR 归一化，加 13 个缺失标记和相对深度，共 27 通道。Conv1d(27,64,k5) 后接四个残差块，每块两个 k5 卷积，扩张率 1/2/4/8，末端 1x1 卷积输出六个通道。共 173,446 参数，局部感受野 245 点（约 24.4 米跨度）。无树模型、无预训练、无多种子融合。

输出：三个默认值 logits 与三个非默认数值回归值。目标变换为 log1p(POR)、log10(PERM)、SW/100，并按训练折非默认样本均值/标准差归一化。损失为三目标平均 BCE 加三目标平均 SmoothL1(beta=0.2)，回归只使用非默认且完整标签有效的点。门控阈值固定 0.5，输出裁剪到原包范围。

训练：normal GPU，AdamW(lr=0.001, weight_decay=0.0001)，cosine LR 到 0.00001；40 epoch，每轮 80 step，batch=16，随机井内 crop=512。只在末轮评估外折，不根据外折挑 checkpoint。fold seed=20260923+fold。

划分：使用与仓库相同的有序井文件、三个标签同时有效掩码和 GroupKFold(5)。插值在各井内部进行，填补和归一化统计仅拟合训练井。不读取 test；不使用发布模型、伪标签、验证阈值调优或指标解码。

从仓库根目录运行：

```powershell
& 'C:\Users\z5364\.conda\envs\normal\python.exe' experiments\simple_cnn\train_cv.py
```

输出位于 outputs/simple_cnn_cv：配置、五折 checkpoint、OOF（truth/pred/gate/continuous/fold/well）、逐折进度及最终 report.json。目标得分统一按 0–100 展示；TOTAL 使用官方 0.30/0.35/0.35 权重，整体 OOF 分数按有效点汇总，与五折简单均值略有不同。

重要：单次固定配置基线不是穷尽调参后的神经网络上限。与仓库报告的 v3 结果比较时，应标明 v3 是既有报告，本次未重训 v3。

## CNN + MLP 变体

`--model cnn_mlp` 增加逐点全连接分支 Linear(27,128) -> GELU -> Linear(128,64) -> GELU，与原 CNN 的 64 通道上下文拼成 128 通道，以 1x1 卷积降至 64 通道，再用两个独立的 64->32->3 头预测门控和标准化数值。共 197,510 参数。输入、基础清洗、损失、优化器、五折与训练预算保持不变，不使用额外异常处理。

```powershell
& 'C:\Users\z5364\.conda\envs\normal\python.exe' experiments\simple_cnn\train_cv.py --model cnn_mlp --out outputs\cnn_mlp_cv
& 'C:\Users\z5364\.conda\envs\normal\python.exe' experiments\simple_cnn\verify_results.py --out outputs\cnn_mlp_cv
& 'C:\Users\z5364\.conda\envs\normal\python.exe' experiments\simple_cnn\compare_mlp.py
```

这是 MLP 分支、特征融合及分离预测头的联合架构变体，未单独消融各部件。持续在相同 CV 上开发可能带来选择偏差，不能把开发后的分数当作独立盲测结果。
