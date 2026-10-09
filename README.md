# 基于测井曲线的储层参数预测 · Tree–Sequence Hybrid (v1 → v6c)

> 昆仑大模型比赛「基于测井曲线的储层参数预测」的实验与提交仓库。
> 当前 A 榜最强：**DL v97 83.9329**；融合/替换候选 v4 83.7159 / v5 83.6394 / **v6 83.9326**；v370 家族 83.55~83.61。
>
> 仓库为 private；**赛事数据不在仓库内**。模型权重不在 git：v3 的 404 MB 带权重包见 Release **v3.0**，DL 线（v97/v370）权重随提交 ZIP 与服务器 runtime 留存。

## 本次更新（2026-09-29）

- [v3.5 源码与复现边界](experiments/v3.5/README.md) · [原提交包与 SHA256](submissions/v3.5/README.md) · [本地审计结果](results/v3.5/README.md)
- [v3.5 与 deep_learning 综合评估](docs/report_v35_and_deep_learning.md) · [队友分支证据快照](results/comparison/README.md)
- **[后续优化执行计划](docs/optimization_plan_v36.md)**：先对齐已有 OOF、拆分连续/门控融合实验，并行准备 v3.6 正确性修复；含阶段依赖、验收、晋级与算力边界，尚未实施新实验。
- **优先处理**：v3.5 源码会接受 128 行异常负标签，NN 训练/推理存在分箱支持错配；归档保留原源码不改，修复版需独立实验。
- **借鉴方向**：队友的逐点条件输入校准、完整后验融合；不要把其第五折 router 分数混为完整五折或 83.6915 的提交方案。

## 方法与结果

原 v3 主线：树模型负责逐点条件分布（CatBoost 分位数分箱），整井一维扩张卷积网络负责储层/非储层判层（门控），
4 状态联合模型提供默认概率，决策层按官方容差核取 Bayes 动作，并在固定按井 OOF 上交叉拟合低自由度融合。

| 版本 / 分支 | 方法 | 线下记录（口径见下） | A 榜（5 口盲井） |
| --- | --- | ---: | ---: |
| v1 | 统一两阶段 CatBoost | 79.407，同折复现 | 81.0052（第 237 名） |
| v2 | 目标专用树模型 | 80.884，同折复现 | 82.2718 |
| v3 | 树–序列混合 + 指标对齐 + 4 状态门控 | 81.632，OOF 决策交叉拟合 | 83.2579 |
| **v3.5** | 树–序列复刻、保留五折模型集成 | **未提供对应 OOF** | **83.2960**（用户反馈） |
| **deep_learning（v97 基线）** | 校准双 U-Net 骨干 · 18 阶段链 | **81.891**，完整五折 OOF | **83.9329**（用户反馈） |
| deep_learning（v370 双种子等权） | 同族 · 8 阶段链 × 双种子 | 82.174，完整 OOF | 83.6076 |
| **v4**（树序列×双骨架·后验融合） | v3×v370 混值 + 仲裁，α 嵌套选定 | **82.2106**（嵌套） | 83.7159 |
| **v5**（双支DL级联·后验融合） | v97×v370 混值 + 仲裁 | **82.1906**（嵌套） | 83.6394 |
| **v6**（锐利 U-Net 融合） | v97 基座 + 三模型默认态二票否决 | **82.0361**，无拟合参数 | **83.9326** |
| v6b（原种子双骨架 U-Net） | v370 · 单 original 种子 | 81.9896 | 83.5547 |
| v6c（折间调参双骨架 U-Net） | v370 · per-fold 比率诊断版 | 82.1947 | 83.6044 |
| **v7**（漂移增强门控移植） | 骨干基座 + 新训漂移增强 SW 门控（B′) | SW 头 干净/伪漂移 75.31/75.29 | **83.8506** |

**七连终审（2026-10-09 定稿）**：七次提交流程上到——一次 v6 打平、其他六次全部低于 v97。OOF 侧取得的增益等级从 +0.05 到 +0.58，每次都在 A 榜 5 口盲井上蒸发。**基座预测在那 5 口井上已接近运气上限，一切改动期望为负**；改动小才持有平（v6 2.23%），改动大掉得最狠。后续方向只剩两条：**新锚模型**（从源头就是新的）、或 83.9329 收束转攻 B 榜。详见 [漂移增强门控移植实验](docs/drift_augmented_gate_transplant_20261009.md)。

v3.5 比 v3 的 A 榜高 0.0381，队友方案比 v3.5 高 0.3955。**线下记录不是统一新消融**：v3 的“嵌套”限于固定 OOF 的决策选择；队友使用各验证折选最佳 epoch，并曾复用第五折筛选架构。不能从这些差值直接推导因果增益、显著性或融合收益。详见[证据边界](docs/report_v35_and_deep_learning.md)。

原 v3 A 榜消融（历史同日提交）：只换门控约 +0.720、只换连续值约 +0.266；去掉序列网络约 −0.455。旧实验结论保留为 v3 的证据，不迁移到新版本。

## 仓库结构

```text
.
├── README.md            本文件
├── ROADMAP.md           队友必读：已验证有效 / 已验证无效 / 值得尝试 / 不必再试
├── CHANGELOG.md         版本时间线、归档记录与分数来源
├── experiments/v3.5/    新提交源码（原样保存）及复现边界
├── tools/               无训练依赖的本地归档审计脚本与测试
├── requirements.txt     原 v3 服务器 venv 的精确版本
├── src/                 v3 实验代码（诊断、CV、变体、序列网络、融合、最终训练/推理/打包）+ src/README.md
├── package/             提交代码包源码（官方目录模板；模型权重见 Release）
├── history/             v1、v2 的训练/推理源码（只含代码）
├── submissions/         v1–v3.5 提交文件、校验和、v3 消融；v3.5 原代码 ZIP
├── results/             v3 历史实验、v3.5 本地审计、队友分支证据快照与数据字典
├── docs/
│   ├── optimization_plan_v36.md   后续优化执行计划、产物契约、实验矩阵与验收规则
│   ├── report_v35_and_deep_learning.md  新版本归档、代码风险与队友方案评估
│   ├── report_v3_optimization.md   原 v3 历史实验报告（冻结，不回写新版本结果）
│   ├── supplement/                 自动生成的补充材料：实验登记表、逐井/逐折得分、特征清单、算力预算
│   ├── figures/                    论文图 1–8（PNG 300 dpi + PDF + 底层 CSV）
│   ├── paper_draft.md              论文 Markdown 草稿
│   ├── competition/                赛题信息整理稿、第二轮审计任务书
│   └── history/                    前两轮的审计报告
├── paper/               Elsevier elsarticle LaTeX 稿（main.tex、refs.bib、figures/）与已编译的 main.pdf
└── external/pdda2021/   SPWLA PDDA 2021（Volve）迁移验证的适配与实验脚本（数据从上游仓库获取）
```

## 本地审计（uv，无训练依赖）

在仓库根目录运行，数据包路径按实际位置替换：

```bash
uv run --no-project --offline --python 3.14 python tools/audit_v35.py --data-zip ../data.zip --check
uv run --no-project --offline --python 3.14 python -m unittest discover -s tools -p "test_audit_v35.py" -v
```

只读取原数据 ZIP 和已归档文件，不连接 SSH、不加载权重；`--check` 不改报告。已通过 10 井/95,948 行校验及源码字节一致性检查。缺少模型与 OOF 时，只能核对产物，不能重算榜分。

## 原 v3 复现路径（历史服务器流程，不属于本次本地操作）

H20 环境：`/home/ubuntu/kunlun_welllog_v3_20260919`。以下仅适用于原 v3；服务器仍有其他实验时请勿直接启动。

1. 准备 `data/train`、`data/test`（赛事原始数据，两行表头，`-99999` 缺失）。
2. `python src/cv_v2.py` 复现 v2 的 5 折 OOF（所有后续实验的共同基准，≈45 min）。
3. 按 `src/README.md` 的顺序运行诊断（`diag_*.py`）、变体（`exp_variant.py`）、序列网络（`nn_seq.py`）、状态模型（`exp_states.py`）。
4. `python src/assemble.py --config src/assemble_cfg_D.json --out outputs/assembly_D` 得到嵌套融合与最终决策配置。
5. `python src/train_final.py --config src/final_cfg_v3.json --out-dir outputs/final_v3` 全量训练；`python src/build_submission_v3.py` 打包并从 zip 独立复现校验。
6. 推理只需 `package/`：`python predict.py --data_dir ./data --output ./result.json`（不读训练数据）。

原 v3 历史记录：从带权重代码包独立解压重新推理，`result.json` 与提交文件逐字节一致（SHA256 `afcaad76…`）。本次没有重跑；这一证明不适用于 v3.5。

## 原 v3 的三条经验（限定于当时实验，详见历史报告 §3、§4）

1. **连续标签预测困难**：当时同井上半段训练、下半段测试，POR 约 0.18、SW 0.11（跨井 0.32 / 0.13）；线性滞后诊断仅解释部分逐点变化。这是诊断结果，不是所有模型的可达上限证明。
2. **当时最有效的分工是树回归、网络判层**：旧序列模型单独约 77.6，但门控融合贡献显著（A 榜门控消融约 +0.72）。队友新 DL 方案已取得完整 OOF 81.713，不能继续把旧模型负结果推广为纯 DL 整体无效。
3. **决策层要跟指标走**：容差核用 Bayes 动作（PERM +0.35 分），RMSE 用均值（Volve 迁移实验复现）。

## 论文

目标期刊 Geodata and AI（Elsevier）。`paper/main.pdf` 为当前编译稿（23 页，preprint 单栏）；作者信息、数据可用性链接、参考文献 DOI 需在投稿前补齐（`paper/main.tex` 中已标注）。

## 许可

开源时间待定；开源前请勿外传。开源时建议 MIT（代码）+ CC BY 4.0（文档/图表）。
