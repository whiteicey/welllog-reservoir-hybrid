# results/dl_fusion — DL 方向融合与悖论证据数据（2026-10-08~09）

与文档 [docs/paradox_dl_fusion_20261008.md](../../docs/paradox_dl_fusion_20261008.md) 配套的机读证据。
全部产物在无标签、无新一轮训练的条件下生成；A 榜分数均为用户报告，平台独立核验未做。

## 文件清单

| 文件 | 内容 | 关键结论 |
|---|---|---|
| `fusion_E1E2E3.json` | v3 基座 × DL(v97/v370) 的连续行混值(E1)/默认门控仲裁(E2)/组合(E3) OOF 网格 | E3-v370 OOF 82.2133；α 逐目标最优：POR 1.0 / PERM 0.75 / SW 0.85 |
| `fusion_nested.json` | E3 融合按井 5 折嵌套验证（4 折选 α、留出折评估） | 82.2106 vs 全-OOF 选择 82.2133，差 0.003（无选择偏置）；分折 alpha 稳定 |
| `v97_x_v370.json` | v97 基座 单头/成对/融合替换 OOF | 换 POR +0.164、全换 82.2084（E3 形式）；全换嵌套 82.1906 |
| `vote_fusion.json` | 树-序列混合 v3 × DL v97 × DL v370 三模型默认态表决（V1–V5）与二票否决分解 | vote-V3 82.2112（+0.320 vs v97，全折正）；vote-V4 82.0361（+0.145，全折正）；v97 形态最接近的 vote-V1 +0.113 |
| `drift/inputs.json` | 10 口测试井 vs 80 口训练井池：9 条曲线 KS 统计、中位偏移（训练 IQR 单位）、缺失率、井均值 | 全部测试井存在 KS 0.114–0.240 的漂移；最重 d328607c（meanKS 0.240，最近训练井 z 距离 2.552） |
| `drift/clusters.json` | robust-z 中位特征上 k=6 聚类：每口测试井所属簇与包含的训练井数 | 测试井分布与普通训练井混居（无独立簇），漂移是连续而非离群异常 |

## 分数链（A 榜，用户报告）

v97 83.9329 > E3 融合(v3×v370) 83.7159 > v97×v370 83.6394 > v370 双种子等权 83.6076 > v370 调参 83.6044 > v370 原种子 83.5547 > calibrated dual-backbone 83.6915 > v3.5 83.2960 > v3 83.2579。
投票表决链 vote-V1 +0.113 / vote-V4 +0.145（OOF 增益，A 榜未提交；vote-V3 +0.320 OOF 也仅提交结构更实的辨析上报为 83.9326 的融合条目之外）。

数据生成方与脚本：服务器 `/home/ubuntu/kunlun_welllog_v3_20260919/`（harness.py、fusion_*.py、vote_fusion.py、diag_divergence_inputs.py、compare_four_submissions.py）。
