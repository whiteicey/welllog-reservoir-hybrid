# 进展

目标：检验强SW私有骨干上的容差概率质量损失能否改善固定第五折。
已完成：读取旧实验和规则，新损失与训练器已实现；v12两项已结束且无增益。
计划：显式运行两个匹配候选，读取原始评分与误差分解，记录最终恢复。
未完成：实际训练收益和跨井泛化验证。
目标完成后：固定候选后再依授权考虑其他折，当前不补折、不覆盖交付。


## 2026-09-28T16:09:08.183155+08:00 强SW容差质量v13启动
本轮显式启动[('likelihood', 76668), ('tolerance_mass', 91348)]。使用SW私有双专家73.622385同起点，区别仅辅助损失系数0/0.5。此前v12已完成无收益，未重复启动。原数据/分折/评分/预算保持；不恢复旧任务、不补折，真实收益待读取。记录results/deep_learning/independent_sw_tolerance_v13_20260928。


## 2026-09-28T16:19:00.119857+08:00 v13完成
两项强SW私有双专家损失对照完成18轮，均保留第0轮73.622384767129，无新增收益；更新后最佳分别73.47330202477876/73.45786228917683。原三独立模型80.274135337791保留，无新三模型重载宣称、无补折/交付更改。完整报告results/deep_learning/independent_sw_tolerance_v13_20260928/FINAL.md。下一步分组泛化诊断尚未启动，>82目标未完成。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
