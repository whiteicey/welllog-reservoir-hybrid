# Well-log reservoir-property prediction

本仓库归档测井储层参数预测的方法、实验和论文。当前唯一现役论文位于 [`paper/current/`](paper/current/)，其 LaTeX、PDF、图件、公开复核脚本和新增受控实验按同一版本归档。

## 当前论文

**Tolerance-Aware Reservoir Property Prediction from Wireline Logs: Representation Complementarity, Decision Sensitivity, and Offline Transfer**

- [英文主文 PDF](paper/current/main.pdf)：23 页；
- [英文补充材料 PDF](paper/current/supplement.pdf)：7 页；
- [LaTeX 与图件](paper/current/)；
- [审稿意见与受控实验说明](docs/reviewer_response.md)。

研究使用真实测井曲线和解释标签开展离线、回顾性基准评价，不将离线得分解释为现场物理验证或在线部署性能。

## 公开复现材料

| 路径 | 内容 |
|---|---|
| `paper/current/` | 当前主文、补充材料、参考文献、全部图件和文件哈希 |
| `reproducibility/public_volve/` | 公开 Volve/PDDA 复核入口、指标、环境和公开图件 |
| `experiments/public_v4/` | 完整 v4 公开从头训练管线、阶段指标和报告 |
| `experiments/reviewer_response/` | 融合 LOMO、交互、阈值、外部权重、分箱数敏感性脚本与汇总结果 |
| `tools/` | 图件生成、审计和公开前去敏感化工具 |
| `docs/history/` | 旧论文与竞赛过程记录；不是当前论文入口 |
| `experiments/`、`src/`、`package/` | 各历史模型线的源码和复现说明 |

## 公开 Volve 快速开始

1. 从 [SPWLA PDDA SIG](https://github.com/pddasig/Machine-Learning-Competition-2021) 获取公开任务数据，按该仓库说明准备 `train.csv`、`test.csv` 和公开标签。
2. 将三个 CSV 放在本仓库的 `external/pdda2021/` 下；数据文件被 `.gitignore` 排除。
3. 安装 `requirements.txt` 中的依赖，然后运行：

```bash
python reproducibility/public_volve/reproduce_public_validation.py
python reproducibility/public_volve/public_gate_sensitivity.py
```

输出写入 `reproducibility/public_volve/output/`。可用 `WELLLOG_ROOT` 指向包含 `package/` 和 `external/pdda2021/` 的另一份源码工作区。

## 内部受控实验

内部原始曲线、标签、模型权重和大型 OOF 数组**不在仓库内**。`experiments/reviewer_response/` 中的内部实验脚本用于复核已有工作区，需要用户自行提供合法获得的 `cache/`、`outputs/` 和模型参数，并设置：

```bash
export WELLLOG_ROOT=/path/to/internal/workspace
```

仓库只提供聚合指标、方法、环境记录和哈希，不提供内部原始数据或隐藏标签。历史提交文件是模型预测结果，不是隐藏真值。

## 一致性与公开审计

- `paper/current/manifest.json`：当前论文文件哈希；
- `RELEASE_MANIFEST.json`：发布目录、脚本和聚合结果哈希；
- `RELEASE_AUDIT.md`：数据、凭据、历史和构建核验；
- `docs/reviewer_response.md`：四项审稿意见与新增实验。

公开前去敏感化工具：

```bash
python tools/sanitize_before_public.py
```

历史开发文档和模型线保留作审计轨迹，不应与当前论文的方法、数值或命名混用。
