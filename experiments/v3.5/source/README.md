# v3 方案：树-序列混合 + Bayes 效用解码 + 嵌套融合

复刻已验证的 A 榜 83.26 方案架构（其线下嵌套 CV 81.63，A 榜有约 +1.6 稳定偏移）。

## 组件

1. **CatBoost 分布模型**（每目标，仅非默认样本，双种子 × 5 折）：
   POR log1p 空间 80 分位箱 / PERM log10 空间 60 箱 / SW 原值 60 箱，
   MultiClass 输出分布 → 在 800 点动作网格上做 **Bayes 效用解码**
   （对每个候选输出值算官方核期望得分，取最大）。
2. **CatBoost 门控**（每目标二分类）+ **4 状态联合模型**
   （0=全默认 1=全非默认 2=SW默认水层 3=PERM默认致密，后验边缘化
   得到每目标默认概率）。
3. **WellNet**：整井扩张卷积（12 残差块，k=5，dilation 1..32×2，SE 门 +
   GroupNorm + 整井上下文池化回拼），每目标 1 门控 logit + 分箱头。
   两组配置：`gate_b`（33 通道，bin_weight 0.3，3 种子，偏门控）、
   `base_c`（拼 113 维树特征，bin_weight 1.0，2 种子）。
4. **低自由度嵌套融合**：连续 = 各源分布在效用层加权混合后解码；
   门控 = 概率加权混合 + 单阈值精确扫描。权重取 {0,0.25,0.5,0.75,1} 单纯形，
   嵌套 5 折（内层选配置、外折评估）报告无泄漏分数。

## 运行顺序（依赖 GPU；CatBoost 加 --gpu 需 GPU 版）

```bash
pip install -r requirements.txt

# 1) 树模型（CatBoost，~1-2h GPU / 更久 CPU）
python train_trees.py --data_dir ../data_1789566691922/train --out_dir ./outputs --gpu

# 2) 两组 WellNet（各 ~2-4h GPU，可并行在两张卡或顺序跑）
python train_nn.py --data_dir ../data_1789566691922/train --out_dir ./outputs --cfg gate_b
python train_nn.py --data_dir ../data_1789566691922/train --out_dir ./outputs --cfg base_c

# 3) 嵌套融合（CPU 数分钟，打印嵌套 OOF Total_Score）
python assemble.py --out_dir ./outputs

# 4) 推理
python predict.py --data_dir ../data_1789566691922/test --out_dir ./outputs --output ./result.json
```

分阶段验证：跑完 (1) 即可看树单源 OOF 分（预期 ≈81 上下）；
(2) 各自打印 NN 单源分（预期 77~79，正常，NN 是门控源不是主力）；
(3) 的嵌套分数是最终线下指标（目标 ≥81.5）。

## 与参考方案的已知差异

- 树模型用组内复刻的 base/extra 特征（口径同参考：曲线变换、井级统计、
  滚动 9/51/201、滞后差分、60/150m 长窗、井顶底距离），非逐列一致。
- NN base_c 拼接的树特征取方差 top-113 而非参考的固定 113 维清单。
- CatBoost 无 GPU 时退回默认 bootstrap（Bayesian 为 GPU 优化选项）。
```
v3_solution/
├── train_trees.py   # 分布 + 门控 + 4 状态
├── train_nn.py      # WellNet gate_b / base_c
├── assemble.py      # 嵌套融合 → decision.json
├── predict.py       # 最终推理
└── src/             # config / data / features / scoring / wellnet / nn_input
```
