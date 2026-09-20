# 基于测井曲线的储层参数预测 · Target-Specialized Ensemble v2

## 方法概览

该版本不再让 POR、PERM、SW 共用同一种回归结构，而是根据官方指标和训练集分布分别建模：

- **POR**：二分类器判断默认值 `0.1`；对非默认样本训练 80 档分位数分类器，以类别中位数作为连续预测。该方法显著降低普通回归在 POR 相对误差指标下的偏差。
- **PERM**：二分类器判断默认值 `0.01`；非默认样本在 `log10(PERM)` 空间使用 MAE 回归，与官方对数比值指标一致。
- **SW**：二分类器判断默认值 `99.9`；非默认样本使用增强的井内统计、局部趋势及岩石物理代理特征，并采用 RMSE 回归。
- **集成**：默认训练两个随机种子并平均概率/回归输出，降低单模型方差。

严格按井 5 折 OOF 验证：

| 方案 | POR | PERM | SW | Total |
| --- | ---: | ---: | ---: | ---: |
| 原两阶段 CatBoost（同一 5 折） | 0.75280 | 0.89318 | 0.74114 | 79.7853 |
| v2 目标专用方案 | 0.77972 | 0.89435 | 0.74824 | 80.8823 |
| 提升 | +0.02692 | +0.00117 | +0.00711 | **+1.0970** |

## 环境

- Python 3.10
- CUDA GPU（推荐 H20）
- 依赖版本见 `requirements.txt`

## 训练

```bash
python train.py \
  --data-dir /path/to/data \
  --output-dir /path/to/outputs \
  --seeds 42,2026
```

如无 CUDA，可追加 `--cpu`，但训练时间会明显增加。

## 推理

```bash
python predict.py \
  --data-dir /path/to/data \
  --model-dir /path/to/outputs/models \
  --metadata /path/to/outputs/model_meta.json \
  --output /path/to/outputs/result.json
```

## 校验

```bash
python validate_result.py \
  --data-dir /path/to/data \
  --result /path/to/outputs/result.json
```

## 打包

```bash
python package_submission.py \
  --solution-dir /path/to/solution_v2 \
  --outputs-dir /path/to/outputs
```

生成：

- `result.zip`
- `submission_code.zip`

## 说明

- 推理阶段需要读取训练集输入曲线，以重建与训练一致的全局缺失值填充值；不会读取训练标签参与测试预测。
- 训练和推理均按文件内部深度顺序构造中心滚动特征；测试时整口井曲线均已给定，因此不构成未来信息泄漏。
- 原始 `solution/` 和 `submission/` 未修改，本版本独立存放。
