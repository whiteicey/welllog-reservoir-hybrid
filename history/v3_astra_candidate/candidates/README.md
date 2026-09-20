# 未验收候选

`por_utility_v2_gate.json` 使用原 v2 的 12 个模型和历史门控参数，只将 POR 连续值的 argmax 改为官方三角核效用决策。95,948 行已通过格式检查；9,106 行 POR 改变，PERM/SW 未改变。

尚无该候选的同口径 OOF 或榜分证据，不建议替代现有提交。新 v3 默认训练/CV 固定 POR gate=0.5，与该历史 gate 消融不同。

复现：把 `submission_v2/submission_code.zip` 的模型解压到新目录，在项目根目录运行：

```bash
python solution_v3/predict.py --data-dir data --model-dir /tmp/welllog-v2-reference/models --metadata submission_v2/model_meta.json --por-decision utility --output /tmp/por_utility_candidate.json
```
