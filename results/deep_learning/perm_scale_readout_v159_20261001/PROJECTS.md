# 进展

目标：扩大PERM有效分布读出收益，完整OOF>82。
已完成：有界原始尺度修正实现、冻结和自身RPS复用。
计划/未完成：等待GPU空位显式启动，最终恢复最佳、实际三头组合、耗时/显存/收益和清理证据。
可选：若尺度修正有效，进一步研究PERM条件分布；不自动补折。


## 已核验实际启动

PID44704，normal GPU，临时SYSTEM_REQUIRED已启用；与POR157双并发。初始PERM89.344760174753与Gaussian原函数一致，84784可训练参数；第2轮快照尚无收益，峰值786.465MiB，无stderr错误。待最终恢复/实际合并验证；演进图谱已更新。

V159 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/18, seconds=178.450, peakMiB=786.465; all144665replay/frozen exact.


## 最终收益和关闭

与正RPS控制差值-0.060804682166，最佳/停止0/18；实际TOTAL80.591193779236，144665行恢复和冻结完整性已验证。无新增收益，不继续扫描本机制；负权重按SHA清单删除，证据保留。训练178.450秒，与其他任务并行；不作独立速度结论。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
