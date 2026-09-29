# results · 数据字典

本目录根层的原 v3 文件由历史服务器实验生成，保持不变。对应完整 OOF/测试输出（`oof.npz`、`test.npz`，约 13 GB）按历史记录保存在服务器 `outputs/`，这里只放报告级 JSON 与小型矩阵。

## 新增归档与来源边界

| 目录 | 来源 / 含义 |
| --- | --- |
| [v3.5/](v3.5/README.md) | 本地标准库审计：原源码哈希、提交格式/深度、预测差异、异常标签计数；**不是训练结果/CV** |
| [comparison/](comparison/README.md) | 固定 `deep_learning` commit 的紧凑证据快照；**没有在本次重跑 OOF 或推理** |

v3.5 的 CV 未提供，队友完整 OOF 81.7132 与后验 router 的第五折 80.2509 需分开。新材料不回填旧 v3 生成脚本的 registry 或论文数据表，避免混合样本、验证协议与来源。完整解释见 [综合报告](../docs/report_v35_and_deep_learning.md)。

## 基准与生成

| 文件 | 内容 |
| --- | --- |
| `cv_v2_report.json` | v2 在统一 5 折下的复现：逐目标 micro/macro/最差 5 井/默认段/非默认段/逐折/逐井 |
| `cv_v1_report.json` | v1（提交估计器）同折复现 |
| `assembly_{A,B,C,D,E,F0,F,G,H0,H,I}_report.json` | 各融合组合的嵌套结果、每折选择、全折选择、配对 bootstrap；`assembly_D_decision.json` = 提交决策配置 |

## 诊断

| 文件 | 内容 |
| --- | --- |
| `diag_wells.json` | 同井半井 / 交错块 / 跨井 / 最近邻井 / 每井最优缩放 |
| `diag_runs.json` | 状态段长分布、转移矩阵、v2 OOF 的碎片化统计 |
| `diag_gate.json` | 门控错误随边界距离分布、薄层召回、非边界错误修复上限 |
| `diag_hf.json` | 逐点变化被曲线滞后差分线性解释的比例 |
| `diag_wellscale.json` | 每井最优缩放及其可预测性（LOO） |
| `diag_typewell.json` | 4 种选井criterion × K=12/30/加权 vs 全井 |
| `exp_decisions.json`、`exp_calib_joint.json` | 解码规则、平滑、嵌套阈值、校准 + 联合解码 |
| `dtw_align_report.json`、`sim_{median,kl,dtw_gr,dtw_comp}.npy`、`well_order.json` | 井间相似度（90×90，顺序见 well_order）、对齐统计、标签复制基线 |

## 模型变体（`<name>_report.json`）

`por_bins80_base_extra`、`perm_bins60_base`、`sw_bins60_enh`、`*_aln`、`gate_*`、`states_base_extra[_aln]`、`stack_states`、
`nn_seq_a`、`nn_gate_b`、`nn_base_c`、`nn_gate_d`、`nn_gate_b_pl`、`nn_base_c_pl` 等：每个含 `args`（完整配置）、逐折耗时、
非默认段得分、全量得分（v2 门控）、联合解码得分与配对 CI；网络含门控准确率。

## 迁移验证

`transfer_pdda2021_report.json`（教程基线、留一井 CV、Volve 可辨识性审计、盲井）、`transfer_pdda2021_pass2.json`（回归+门控组合、范围规则、逐盲井）。

## 汇总

`summary_all.json`、`summary_variants.md`：`collect_results.py` 生成的全部实验汇总。
