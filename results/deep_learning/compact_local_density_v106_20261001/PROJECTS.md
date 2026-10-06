

## 2026-10-01 v106密度第5阶段
已完成：v104 local选定SW74.121753、频域消融负；冻结atom一次缓存+私有density实现。现在：直接训练第5阶段，与小POR投影任务全局最多2GPU。待做：实际父增益、固定三通道和包含来源/共享阶段的耗时。目标：原第五>80.8、原完整OOF>82、完整配方≤5阶段；达成后新交付。

### 最终第5阶段
SW=74.121752828158，best0/stop18，实际新增益0，221.038秒，峰值1324.5MiB；冻结atom和最佳恢复通过，预测与父local完全一致。实际固定三通道TOTAL=80.475161980049，未过80.8。执行了含源5训练阶段，选定有效依赖仍是父的4阶段；不把多训练一阶段当增益，保留父local。停止此密度设置，不补其他折；完整原OOF>82未完成。

v106 best0重复模型及恢复状态清理27.99MiB，保留实际选定v104 local父模型、全部预测/源码/配置/日志/结果与清单。v105和spectral负模型此前清理37.04MiB，v101正模型和活跃v107不动。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
