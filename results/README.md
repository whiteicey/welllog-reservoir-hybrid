# results · 数据字典

全部文件由服务器实验直接生成，未手工修改。每个实验的完整 OOF/测试输出（`oof.npz`、`test.npz`，共 13 GB）保存在服务器 `outputs/`，
这里只放报告级 JSON 与小型矩阵。

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
