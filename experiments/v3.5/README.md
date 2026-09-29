# v3.5 · 已提交源码归档

本目录保存用户提供的 `v3.5_solution.zip` 解包源码。A 榜 **83.2960** 为用户反馈；提交日期、实际训练环境与生成配置未随包提供。

## 目录与版本边界

```text
experiments/v3.5/
├── README.md                 本归档说明
└── source/                   原包 v3_solution/，13 个文件逐字节保留
    ├── README.md             原始说明，含未经本次验证的历史预期
    ├── requirements.txt
    ├── train_trees.py
    ├── train_nn.py
    ├── assemble.py
    ├── predict.py
    └── src/                  config / data / features / nn_input / scoring / wellnet
```

- 原始 ZIP（包括 11 个 `.pyc`）保存在 [solution.original.zip](../../submissions/v3.5/solution.original.zip)；解包源码排除 `__pycache__/`，没有修改任何 `.py`。
- 原始结果保存在 [result.zip](../../submissions/v3.5/result.zip)；与收到的 `v3.5_result.zip` 逐字节一致。
- 外部归档版本是 **v3.5**，源码目录仍称 `v3_solution`，结果内仍是 `modelName=tree-seq-hybrid-v3`、`version=3.0`。这是已提交文件的历史元数据，不为统一名称而重写。
- 本版本是树–序列路线的独立复刻，**不是原 v3 的同权重或同配置发布**。原 v3 的 CV 81.6324、全量模型和独立复现证明不能直接继承。
- [manifest.json](../../submissions/v3.5/manifest.json) 记录原包、结果和各源码的 SHA256、榜单来源与证据缺口。

## 当前能验证什么

本机仅使用 uv 管理的 Python 标准库：原包/源码一致性、Python 语法、结果 JSON 格式、10 口测试井的 95,948 行深度对齐与数值范围均通过。

**未执行训练或模型推理，未连接 SSH，未独立验证榜单。** 原包没有权重、`bins.json`、`decision.json`、模型元数据或 OOF；无法证明仅用当前源码可重生成该结果。

在仓库根目录运行，无需安装 CatBoost、PyTorch 或下载数据：

```bash
uv run --no-project --offline --python 3.14 python tools/audit_v35.py --data-zip ../data.zip --check
uv run --no-project --offline --python 3.14 python -m unittest discover -s tools -p "test_audit_v35.py" -v
```

`--check` 只读比较已归档报告；不加该选项会重生成 `results/v3.5/local_audit.json` 与逐井差异 CSV。数据包不在 Git 中，`../data.zip` 是本次克隆目录外的本地文件，可替换为自己的路径。

## 已知问题：先修复、另立版本，再训练

1. **异常标签清洗**：`src/data.py` 仅以 `v > -99999/2` 判有效，本数据有 128 行三目标负异常值会被误收（其中 126 行为 `-9999`）；原 v3 物理范围过滤会排除它们。
2. **NN 分箱语义**：`train_nn.py` 按每口井自己的验证折边界预构造标签，再供不同外折训练，导致一个分类头混用多套边界，并使其他折边界的拟合接触外层验证标签。
3. **NN 推理支持错配**：`predict.py` 先跨折平均分箱概率，再将平均概率配给各折支持点；应保持每个模型概率与本折支持点一一对应，在公共动作网格上平均期望效用。
4. **验证边界**：部分预处理在所有训练井上拟合；融合的“嵌套”是固定 OOF 上的决策交叉拟合，不是每个内折重新训练完整模型。两者都不宜描述为完全无偏验证。

这些问题是源码/数据审查结论，**没有据此推算 A 榜损失或收益**。本归档不悄悄修正历史源码；修复版需新目录、新配置与新结果。详细证据见 [综合报告](../../docs/report_v35_and_deep_learning.md)。

## 恢复原推理还缺哪些文件

至少需要本次实际运行所用的：

- `trees_meta.json`、`bins.json`、`decision.json`；
- 分布、二分类门控与四状态树模型：默认配置共 **70 个 `.cbm`**；
- `nn_gate_b` / `nn_base_c` 的各折权重及 `nn_*_meta.json`：默认两组共 **25 个 `.pt`**；若启用 `edge_b` 还需对应额外权重；
- 核对训练/CV还需 `trees_oof.npz`、`nn_*_oof.npz`、每行井号/深度/折号、训练日志、精确依赖与硬件/精度配置。

原推理入口为 `source/predict.py --data_dir <测试txt所在目录> --out_dir <上述模型产物目录> --output <结果路径>`。它不是只有源代码即可运行的完整比赛交付包；不要把 v3.0 Release 的权重与本版本混用。
