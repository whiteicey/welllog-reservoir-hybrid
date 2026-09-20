# 基于测井曲线的储层参数预测 · 提交代码包（v3）

## 1. 算法名称

指标对齐的树-序列混合储层参数预测模型（Metric-aligned Tree–Sequence Hybrid）

## 2. 方法简介

对每口井的 13 条输入曲线（不使用 CASE）逐深度点预测 POR、PERM、SW。

1. **特征工程**（`src/feature.py`）：`-99999` 视为缺失并按井线性插值；构造 113 维基础特征（曲线变换、井内统计、9/51/201 点滚动统计）、237 维增强特征（井内分位归一化、局部残差、斜率、密度/声波孔隙度、GR 泥质指数、Archie 代理量等）以及 181 维附加特征（±1～3 点滞后差分、3/5 点中值、一阶/二阶差分、密度-中子分离度、60 m / 150 m 地层尺度窗口统计、距井顶/井底距离）。插值用的全局中位数保存在 `models/model_meta.json`，推理不读取训练数据。
2. **树模型**（CatBoost，`src/model.py`）：每个目标一个默认值门控二分类器；每个目标一个分位数分箱 MultiClass 分布模型（POR 80 档 log1p、PERM 60 档 log10、SW 60 档线性），只在非默认样本上训练；一个 4 状态联合储层状态模型（非储层 / 储层 / 水层 SW=99.9 / 致密 PERM=0.01）。均为 2 个随机种子平均。
3. **序列网络**（PyTorch，`src/model.py` 中 `WellNet`）：整井一维扩张残差卷积网络（12 个残差块，dilation 1～32，SE 门控，全井上下文向量），每个目标输出门控 logit 与分箱 logit；两组网络（仅曲线输入 3 种子；曲线 + 113 维特征通道 2 种子）。
4. **决策层**（`src/utils.py`）：连续值取各来源预测分布加权混合后在 800 点动作网格上官方核（POR 8 % / SW 5 % 相对容差三角核，PERM log10 比值核）期望效用最大的动作；门控为各来源默认概率的加权混合与单一阈值比较。权重与阈值由按井 5 折嵌套交叉验证选出，写在 `model_meta.json` 的 `decision` 字段。

按井 5 折交叉验证（80 口训练井）官方总分 81.63（v2 80.88，v1 79.41）；A 榜 83.2579。

## 3. 运行环境

- Python 3.10（服务器实测 3.10.12；3.9～3.12 均可）
- 依赖（`requirements.txt`）：catboost 1.2.10、numpy 2.2.6、pandas 2.3.3、torch ≥ 2.4
- 训练与提交结果在 NVIDIA H20（CUDA 12.6）上生成；推理支持 GPU 或 CPU（`config.yaml` 中 `device: auto`）。CPU 推理 10 口井约 3～5 分钟；GPU 约 1 分钟。序列网络在 GPU 上用 bf16 自动混合精度，CPU 上为 fp32，结果仅有浮点精度级别的微小差异。

## 4. 运行命令

```bash
pip install -r requirements.txt
python predict.py --data_dir ./data --output ./result.json
```

- `--data_dir`：与赛事下发数据结构一致的目录，需包含 `test/` 子目录（推理不使用 `train/`）。
- `--output`：预测结果 `result.json` 的保存路径。
- 可选：`--config config.yaml`（默认）、`--model_dir models`、`--metadata models/model_meta.json`、`--device auto|cuda|cpu`。

结果格式校验（井数、行数、深度顺序、字段、数值范围）：

```bash
python -m src.validate --data_dir ./data --result ./result.json
```

## 5. 文件说明

```text
submission_code.zip
├── README.md                  # 本文件
├── predict.py                 # 主预测脚本：读取测试井 → 特征 → 模型推理 → 决策 → result.json
├── requirements.txt           # Python 依赖
├── config.yaml                # 模型目录、元数据路径、推理设备
├── src/
│   ├── __init__.py
│   ├── data.py                # 读取两行表头的曲线文件，-99999 缺失处理，量程过滤
│   ├── feature.py             # FeatureBuilder（113/237 维）与 181 维附加特征
│   ├── model.py               # CatBoost 加载与种子平均、WellNet 序列网络、二级特征拼装
│   ├── utils.py               # 官方核效用矩阵、动作网格、HMM 平滑、配置读取、结果写出
│   └── validate.py            # 结果文件校验工具
├── models/
│   ├── model_meta.json        # 特征名、分箱中心、训练中位数、网络归一化统计量、决策配置
│   ├── POR_gate_seed{42,2026}.cbm / PERM_gate_* / SW_gate_*      # 默认值门控
│   ├── POR_bins_seed{42,2026}.cbm / PERM_bins_* / SW_bins_*      # 分位数分箱分布模型
│   ├── STATES_seed{42,2026}.cbm                                  # 4 状态联合模型
│   ├── nn_gate_b_seed{1,2,3}.pt                                  # 序列网络（曲线输入）
│   └── nn_base_c_seed{1,2}.pt                                    # 序列网络（曲线 + 113 维特征通道）
└── examples/
    └── result_example.json    # 输出格式样例（1 口井前 3 行）
```

随机种子固定（CatBoost 42/2026，网络 1/2/3）；本包内的权重即生成提交结果所用的权重，从本包独立解压重新推理得到的 `result.json` 与提交文件逐字节一致（服务器 GPU 环境实测）。
