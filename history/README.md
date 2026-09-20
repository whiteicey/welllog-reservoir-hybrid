# history

- `v1_solution/`: 首轮提交源码（统一两阶段 CatBoost）。注意本地 `train_catboost.py` 与提交 ZIP 内脚本曾有漂移，以 `submissions/v1/model_meta.json` 记录的配置为准。
- `v2_solution/`: 目标专用方案源码（A 榜 82.2718）。`src/` 的数据读取与特征代码被 v3 的 `src/ref/solution_v2/src` 原样复用。
- `v3_astra_candidate/`: 第二轮审计（另一模型）实现的候选代码（POR 效用解码），未提交，其思路在 v3 中重新实现并验证。
