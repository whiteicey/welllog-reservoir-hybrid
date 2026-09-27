# 异构四模型投票提交（2026-09-27）

## 结果

| 项 | 值 |
|---|---|
| 完整 OOF（4 模型 20 折等权投票） | **81.99647883414204** |
| 成员 OOF | posterior-seed0 81.6420 / posterior-seed1 81.6988 / posterior-seed2 81.6444 / local-router 81.6917 |
| 历史交付（v3） | OOF 81.7132 / A 榜 83.6915 |
| 三种子投票（昨日） | OOF 81.9842 / A 榜 83.6709 |

## 方法

- posterior-router × 3 种子（seed 20261006 + 0/1/2）
- local-router × 1 种子（路由隐层加零初始化 depthwise conv 残差）
- 每成员内部：两骨干完整后验等权融合 → 期望效用
- 跨 4 成员 × 5 折 = 20 个 fold 级模型，期望效用等权平均后取 Bayes 最优动作

## 目录

- `code.zip`：推理代码 + 4 成员 × 5 折权重（20 个），解压后根为 `package_dl/`
- `result.zip`：仅根目录 `result.json`（10 口井，95,948 行）
- `result.json` / `result.inference.json`：未压缩结果与推理元信息

## 复现

```bash
python package_dl/predict_heterogeneous.py \
  --data_dir /path/to/data \
  --model_root package_dl/models_posterior \
  --members seed0 seed1 seed2 local --folds 1 2 3 4 5 \
  --output result.json --device cuda
```

## 完整 OOF 复算

```bash
python tools/deep_learning/vote_heterogeneous.py \
  outputs/full_seed0 outputs/full_seed1 outputs/full_seed2 \
  outputs/full_local_router outputs/full_disagreement_router
```
