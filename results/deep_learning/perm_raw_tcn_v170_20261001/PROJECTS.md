目标：新269维输入、64维多尺度TCN分布修正编码器；原SOURCE保留冻结先验
已完成：实现及预定义对照。
计划：显式启动第五折，记录真实结果。
未完成：训练、恢复、实际组合、时间与图谱；完整OOF>82未达成。
达标后：严格TOTAL>80.8且有收益才补其余四折。
入口失败：默认GBK读取UTF8 PLAN时报UnicodeDecodeError；未训练、无成绩、不是机制负结果。原源码/日志/配置保持；新v172显式UTF8读取，不续跑失败输出。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
