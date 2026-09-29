# v3.5 · 本地审计结果

本目录由 [tools/audit_v35.py](../../tools/audit_v35.py) 生成，**不是服务器训练报告，也不包含重新计算的 CV/A 榜得分**。

| 文件 | 含义 |
| --- | --- |
| [local_audit.json](local_audit.json) | 输入包 SHA256、原源码一致性/语法校验、提交格式与深度校验、总体预测差异、训练标签有效性对照 |
| [prediction_diff_by_well.csv](prediction_diff_by_well.csv) | 10 口测试井 × 3 目标的预测差异与默认状态变化；left=v3，right=v3.5 |

`changed_count` 按两版输出绝对差 >1e-9 计数，包含浮点保存精度差异。`default_disagreement_count` 按 POR≤0.1 / PERM≤0.01 / SW≥99.9 判定默认状态后比较。`mean_absolute_prediction_difference` 是两组预测之差，**不是相对于真值的 MAE**。

异常标签审计分别计算原 v3 物理范围策略和 v3.5 `v > MISSING / 2` 策略，不训练模型、不导入训练源码。结果为 v3 有效三标签行 723,556，v3.5 策略接收 723,684，差值 128。

复核命令（仓库根目录，替换数据包路径）：

```bash
uv run --no-project --offline --python 3.14 python tools/audit_v35.py --data-zip ../data.zip --check
```

生成时使用 uv 管理的 CPython 3.14.5，仅依赖标准库；该环境不是原模型训练环境。缺少匹配模型产物，未执行重新推理，也没有计算两模型预测误差互补性。

报告先写临时文件再替换，JSON 最后发布并记录配套 CSV 的 SHA256。两个文件不是事务式同时替换；如写入中断，`--check` 或 JSON 内 CSV 哈希可发现不一致。输出目录禁止指向原源码、提交产物、package 或 `.git`。
