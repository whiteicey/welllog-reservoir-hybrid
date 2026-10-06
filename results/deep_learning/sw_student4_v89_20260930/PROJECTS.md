# v89 SW Student-t4结果

候选SW 73.779643062735，固定POR/PERM TOTAL 80.377309529801；对Gaussian SW -0.013667889753、TOTAL -0.004783761414。最佳/停止14/32，patience18。采用=False，当前独立路线TOTAL仍为80.382093291214。

探索与验证：同839481参数、raw全模型/输出初始hash、原训练侧先验/输入bank/预算/效用1/40%DropKey对齐历史Gaussian。t4标准差参数化与NLL/CDF同步；初始概率分布不同，但动作选择可能相同，不强行要求动作差异。实际梯度/源/预处理/评分hash、最佳144665恢复、原行序和固定三列评价通过。正候选重载=[]；负项不重复原Q或独立预检。

默认/数值/井诊断：{"default": {"n": 100786, "reference": 97.48140224897922, "candidate": 97.57638028978397, "channel_delta": 0.09497804080474737, "TOTAL_contribution": 0.02315943654091552}, "numeric": {"n": 43879, "reference": 19.383958656215817, "candidate": 19.120741124100434, "channel_delta": -0.2632175321153838, "TOTAL_contribution": -0.02794319795452821}}；9井升/7井降。

收益与不继续：未超同容量Gaussian则关闭当前df4/迁移骨干/输出先验重置/效用1/DropKey0.4组合，不扫df或scale；不把从自身先验提高当超越对照，不泛化所有重尾网络。旧共享t3负证据单独保留，不重启。

成本：{"single_candidate_seconds": 222.34578519998468, "launch_to_completion_seconds": 226.76713, "launch_to_verified_seconds": 826.911832, "peak_training_processes": 1, "peak_allocated_MiB": 1814.59765625, "peak_reserved_MiB": 1918.0, "finalization_seconds": 0.2179382999893278, "gaussian_control_retrained": false, "scope": "candidate data/warmup/train/eval/save included; historical control reused, authoring/atlas separate"}。
清理3个冗余权重；删除前计算的大小/哈希未能成功落盘，无法恢复，清单已明确标为未知，未补造数值；原正源/旧交付不改。

已完成：真实训练、匹配历史比较、诊断、最佳恢复、实际组合、必要重载/清理。未做：其他折和独立泛化；原第五严格>80.5且明确收益后才考虑其余四折，完整原OOF>82仍未完成。目标后：独立井确认和新目录交付。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
