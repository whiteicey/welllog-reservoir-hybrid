# 方案对比 · 证据快照

[deep_learning_snapshot/](deep_learning_snapshot/) 固定于 `deep_learning` 分支提交 `f07a7d7d3be341477d9ef7624510e73eb0168d0e`，避免分支后续变化导致本次对比失去上下文。

- `full_oof_original_verification.json`：原分支保存的完整五折原评分核验记录。
- `leaderboard_result.json`：用户反馈 A 榜 83.6915 及来源声明，明确不属于后续 router。
- `delivery_validation.json` / `standalone_execution_verification.json`：原分支提交包哈希和独立 CLI 执行记录。本次没有再次执行这些推理。
- `final_posterior_router.json`：后续第五折探索，不能与完整五折交付混用。
- `selected_experiments.json`：从原实验索引选出的单成员和校准双成员两个条目；不是全量实验索引。
- `manifest.json`：原始路径、固定 commit、逐文件哈希、摘录来源与证据限制。

前五个 JSON 保留 Git blob 原字节。快照不含数据、权重、测试逐点预测或 OOF 数组，因此不能据此自行重算 CV、预测互补率或融合得分。原分支的“验证通过”是其已保存的记录，不等于本次进行了相同验证。

完整分析：[docs/report_v35_and_deep_learning.md](../../docs/report_v35_and_deep_learning.md)。
