

## 当前结果
观察时间：2026-09-28T09:40:23+08:00

|方法|状态|TOTAL|最佳轮次|
|---|---|---:|---|
|linear|最终固定三头共同恢复已验证|80.253777203|[1, 0, 0]|
|cross80|最终固定三头共同恢复已验证|80.255112897|[1, 0, 0]|
|gate_only|最终固定三头共同恢复已验证|80.222820979|[0, 34, 0]|
|density_only|最终固定三头共同恢复已验证|80.256678729|[1, 0, 0]|

已完成：新增线性归因/联合内部专家读出/因素分离实现与实际运行记录。
待完成：所有候选的最终恢复、逐通道配方固定后的共同推理；不存在完整OOF增益证明。
目标：当前双专家可复现优化，第五折>80.5且明确收益；完整OOF>82仍未达成。
达成后：固定方案后评估泛化，当前不自动补折、不改旧交付。
图谱在既有curation生成后运行本脚本再build_atlas；不修改此前被保护的curate.py。


## 2026-09-28T09:40:23+08:00 四项优化与固定通道组合已完成
本轮linear/cross80/gate_only/density_only全部自然结束并共同恢复。最终固定组合TOTAL=80.257638624609，相对原80.221861=+0.035777541569；配方=[{'target': 'POR', 'arm': 'density_only', 'score': 77.39392666356925, 'epoch': 1}, {'target': 'PERM', 'arm': 'gate_only', 'score': 89.3475027344081, 'epoch': 34}, {'target': 'SW', 'arm': 'source', 'score': 73.62238476712884, 'epoch': 52}]。没有达到独立泛化目标，不启动补折或改交付。更细分布读出优化未启动。总结见results/deep_learning/dual8222_round2_20260928/FINAL.md。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
