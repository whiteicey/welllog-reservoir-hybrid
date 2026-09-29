# v3.5 · 提交产物

| 文件 | 内容 |
| --- | --- |
| [result.zip](result.zip) | 原 `v3.5_result.zip`，完整保留；内含根目录 `result.json` |
| [solution.original.zip](solution.original.zip) | 原 `v3.5_solution.zip`，完整保留，包括原 Python 字节码；非完整带权重推理包 |
| [manifest.json](manifest.json) | 版本、用户反馈分数、校验和、复现状态与证据链接 |

- **A 榜：83.2960**，用户提交后反馈；本次未访问赛事平台独立核验。归档日期为 2026-09-29，不代表实际提交日期。
- 结果 ZIP：487,994 字节；SHA256 `719577259ca317a2dac32f2479eae01e7afb82e2f346d3d1e670329e8b41f27d`。
- 原代码 ZIP：83,378 字节；SHA256 `4e267c6eff30569b4d87553ad4d285d3f974b17e6db36438abdaaeb49887b37f`。
- 本地已检查 10 口井、95,948 行，字段、深度逐行对齐与数值范围通过。格式校验不证明预测准确率。
- 未提供对应模型权重、融合决策或 OOF，未执行独立推理；不能将原 v3 的复现证明迁移至 v3.5。
- 结果内部版本仍是 `3.0`；为保持已提交文件哈希不变，未修改。

源码入口：[experiments/v3.5/](../../experiments/v3.5/README.md)。本地审计：[local_audit.json](../../results/v3.5/local_audit.json)。方法、风险及队友方案对比：[综合报告](../../docs/report_v35_and_deep_learning.md)。
