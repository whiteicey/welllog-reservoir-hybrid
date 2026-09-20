# 基于测井曲线的储层参数预测 · Tree–Sequence Hybrid (v1 → v2 → v3)

> 昆仑大模型比赛「基于测井曲线的储层参数预测」赛题的完整工作仓库：三代方案的代码、按井 5 折实验数据、A 榜提交产物、
> 实验报告与论文初稿。当前最终版本 **v3**：A 榜 **83.2579**（v2 82.2718，v1 81.0052）。
>
> 仓库现为 private；比赛结束、论文接收后开源。**赛事数据不在仓库内**（属主办方），模型权重（404 MB）以 Release 附件提供。

## 一句话方法

树模型负责逐点条件分布（CatBoost 分位数分箱），整井一维扩张卷积网络负责储层/非储层判层（门控），
4 状态联合储层状态模型统一三个目标的默认值判断，决策层按官方容差核取 Bayes 最优动作，
按井嵌套交叉验证做低自由度融合。三代结果：

| 代际 | 方法 | 同折 5 折 CV | A 榜（5 口盲井） |
| --- | --- | ---: | ---: |
| v1 | 统一两阶段 CatBoost（默认值分类 + MAE 回归） | 79.407 | 81.0052（第 237 名） |
| v2 | 目标专用：POR 80 档分箱、PERM log MAE、SW 237 维 RMSE | 80.884 | 82.2718 |
| **v3** | 树-序列混合 + 指标对齐解码 + 4 状态门控 + 嵌套融合 | **81.632** | **83.2579** |

A 榜消融（同日提交）：只换门控 +0.720、只换连续值 +0.266（两者之和 = 完整版 +0.986，严格可加）；去掉序列网络 −0.455。

## 仓库结构

```text
.
├── README.md            本文件
├── ROADMAP.md           队友必读：已验证有效 / 已验证无效 / 值得尝试 / 不必再试
├── CHANGELOG.md         三代方案时间线与分数
├── requirements.txt     服务器 venv 的精确版本
├── src/                 v3 实验代码（诊断、CV、变体、序列网络、融合、最终训练/推理/打包）+ src/README.md
├── package/             提交代码包源码（官方目录模板；模型权重见 Release）
├── history/             v1、v2 的训练/推理源码（只含代码）
├── submissions/         三代 result.zip、model_meta.json、校验和、v3 消融提交文件
├── results/             全部实验结果 JSON（56 个）、井间相似度矩阵、汇总表 + results/README.md 数据字典
├── docs/
│   ├── report_v3_optimization.md   完整实验报告（中文，20 节，含每个实验的数字与结论）
│   ├── supplement/                 自动生成的补充材料：实验登记表、逐井/逐折得分、特征清单、算力预算
│   ├── figures/                    论文图 1–8（PNG 300 dpi + PDF + 底层 CSV）
│   ├── paper_draft.md              论文 Markdown 草稿
│   ├── competition/                赛题信息整理稿、第二轮审计任务书
│   └── history/                    前两轮的审计报告
├── paper/               Elsevier elsarticle LaTeX 稿（main.tex、refs.bib、figures/）与已编译的 main.pdf
└── external/pdda2021/   SPWLA PDDA 2021（Volve）迁移验证的适配与实验脚本（数据从上游仓库获取）
```

## 复现路径（服务器 H20，`/home/ubuntu/kunlun_welllog_v3_20260919`）

1. 准备 `data/train`、`data/test`（赛事原始数据，两行表头，`-99999` 缺失）。
2. `python src/cv_v2.py` 复现 v2 的 5 折 OOF（所有后续实验的共同基准，≈45 min）。
3. 按 `src/README.md` 的顺序运行诊断（`diag_*.py`）、变体（`exp_variant.py`）、序列网络（`nn_seq.py`）、状态模型（`exp_states.py`）。
4. `python src/assemble.py --config src/assemble_cfg_D.json --out outputs/assembly_D` 得到嵌套融合与最终决策配置。
5. `python src/train_final.py --config src/final_cfg_v3.json --out-dir outputs/final_v3` 全量训练；`python src/build_submission_v3.py` 打包并从 zip 独立复现校验。
6. 推理只需 `package/`：`python predict.py --data_dir ./data --output ./result.json`（不读训练数据）。

已验证：从提交代码包独立解压重新推理，`result.json` 与 A 榜提交文件逐字节一致（SHA256 `afcaad76…`）。

## 最重要的三条结论（详见 docs/report_v3_optimization.md §3、§4）

1. **标签不是曲线的确定函数**：同井上半段训练、下半段测试，POR 只有 0.18、SW 0.11（跨井 0.32 / 0.13）；逐点变化只有 36～42% 能被曲线解释。任何只用这 13 条曲线的模型都到不了连续段上限。
2. **树回归、网络判层**：序列网络单独用只有 77.6，但门控准确率 SW 0.867→0.923、POR 0.917→0.934；作为门控源融合后是最大增益来源（A 榜 +0.72 来自门控）。
3. **决策层要跟指标走**：容差核用 Bayes 动作（PERM +0.35 分），RMSE 用均值（Volve 迁移实验复现）。

## 论文

目标期刊 Geodata and AI（Elsevier）。`paper/main.pdf` 为当前编译稿（23 页，preprint 单栏）；作者信息、数据可用性链接、参考文献 DOI 需在投稿前补齐（`paper/main.tex` 中已标注）。

## 许可

开源时间待定；开源前请勿外传。开源时建议 MIT（代码）+ CC BY 4.0（文档/图表）。
