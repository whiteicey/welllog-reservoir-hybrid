# 基于测井曲线的储层参数预测 · CatBoost 两阶段方案

## 算法名称

CatBoost two-stage reservoir parameter prediction

## 方法简介

本方案将任务拆成两层：

1. **默认值判别层**：分别判断 `POR`、`PERM`、`SW` 是否等于训练集中的高占比默认值。
2. **非默认值回归层**：对判别为非默认值的深度点，使用 CatBoost 回归器预测连续储层参数。

特征工程包含原始测井曲线、对数变换、方位角三角变换、井眼几何特征、物理派生比值、缺失值标记、井内统计量以及多尺度滚动窗口特征。模型训练采用按井分组的交叉验证，避免同一口井同时出现在训练集和验证集。

## 运行环境

- Python 3.10.12
- CUDA 驱动：580.126.20
- GPU：NVIDIA H20

依赖版本见 `requirements.txt`。

## 训练命令

```bash
python train_catboost.py \
  --data-dir ../data \
  --output-dir ../outputs \
  --folds 3 \
  --iterations 500 \
  --depth 8 \
  --learning-rate 0.05
```

训练产物：

```text
outputs/
├── models/
│   ├── POR_classifier.cbm
│   ├── POR_regressor.cbm
│   ├── PERM_classifier.cbm
│   ├── PERM_regressor.cbm
│   ├── SW_classifier.cbm
│   └── SW_regressor.cbm
└── model_meta.json
```

## 推理命令

```bash
python predict_catboost.py \
  --data-dir ../data \
  --model-dir ../outputs/models \
  --metadata ../outputs/model_meta.json \
  --output ../outputs/result.json
```

## 结果校验命令

```bash
python validate_result.py \
  --data-dir ../data \
  --result ../outputs/result.json
```

校验内容包括：

- 顶层字段是否为 `modelId`、`modelName`、`version`、`resultData`
- 测试井数量是否为 10
- 每口井的 `logId` 是否与测试文件名一致
- 预测行数是否与测试数据行数一致
- `depth` 是否与测试数据一致
- 每行字段是否为 `depth`、`POR`、`PERM`、`SW`
- 数值是否有限且在有效范围内

## 交叉验证结果

3 折井级交叉验证的 OOF 结果：

| 目标 | 单项分数 | 最优阈值 | 缩放 | 偏移 |
| --- | ---: | ---: | ---: | ---: |
| POR | 0.7501 | 0.20 | 1.00 | 0.00 |
| PERM | 0.8917 | 0.35 | 1.00 | 0.00 |
| SW | 0.7403 | 0.10 | 1.05 | -2.00 |

按官方权重 `0.3 / 0.35 / 0.35` 计算，综合得分约为 **79.62**。

## 文件说明

```text
solution/
├── README.md
├── requirements.txt
├── train_catboost.py
├── predict_catboost.py
├── validate_result.py
└── src/
    ├── data.py
    ├── features.py
    ├── metrics.py
    └── forest.py
```

`forest.py` 是早期 NumPy 随机森林实验实现，当前最终提交使用 `train_catboost.py` 中的 CatBoost 两阶段方案。

