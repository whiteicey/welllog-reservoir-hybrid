# v158 进展

目标：PERM采用独立分布读出机制提高成绩，最终完整OOF>82。

已完成：Laplace16分量/匹配标准差、properNLL/CDF/jointRPS同步实现；PLAN与Gaussian正控制固定。

计划/未完成：显式启动与POR157并行，观察GPU真实状态；恢复最佳、实际合并、收益/时间显存证据、清理与图谱更新。

可选：正收益继续独立PERM结构或优化；负则换机制，不开自动族/系数网格，不自动补折。


## 实际启动快照

已核验PID3580正常训练，normal GPU，TEMPORARY_SYSTEM_REQUIRED=True，与POR157并行。Laplace初始PERM89.332375289863，Gaussian初始89.344760174753，初始函数不同；第11轮仍未改善初始值。训练中，不构成最终机制结论。
未完成：最佳磁盘重载、实际三通道组合、最终差值及耗时记录。

V158 final PERM=89.332375289863, compactcontrolgain=-0.046018921280; actualTOTAL=80.586859069525, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/18, seconds=251.808, peakMiB=786.465; all144665replay/frozen exact.


## 最终结论与关闭

Laplace PERM89.332375289863，Gaussian/RPS对照差值−0.073189567056；第0轮最佳，第18轮早停，无学习改善。实际三通道TOTAL80.58685906952451；磁盘最佳重载和冻结完整性、144665行实际组合已验证。关闭本配置，不泛化所有Laplace机制无效。负权重已按SHA/路径清单删除，来源/配置/预测/日志/结果保留。训练251.808秒，与POR157并发，单项时间不能作串行速度结论。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
