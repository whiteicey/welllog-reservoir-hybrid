# v149 进展

目标：通过结构化分布读出获得第五折真实收益，最终完整OOF>82。

已完成：两组相同容量/初始化分量头实现与预定义PLAN，保持原冻结父模型和五阶段依赖。

计划/未完成：显式双并发训练shared/communicate；观察真实进程、记录初步成绩；恢复最佳并实际三通道合并；报告连接消融与绝对对照；更新图谱并清理负权重。

目标完成后可选：证实POR机制收益后，分别迁移PERM/SW并独立验证；目前不启动其他折或其他机制。

Verified explicit dual launch: shared PID45244 and communicate PID22848, both command lines and actual training epochs confirmed live. Both initial POR77.540454326739 and290610 trainableparameters. Snapshot shared epoch6, communicate epoch4, no improvement yet; peak956.931MiB each, noOOM. Shared epochs ~6.4s and communication ~8.9s under overlap; no completed equal-work wall comparison yet. Final gain/actual assembly pending; do not promote.

V149 shared: restored POR=77.672952461232; delta vs v114=+0.124882966339, vs historical=+0.121015659347; actual fifth TOTAL=80.530276861749;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=17/35, seconds=320.591, peakMiB=956.931.

V149 communicate: restored POR=77.645461857202; delta vs v114=+0.097392362309, vs historical=+0.093525055317; actual fifth TOTAL=80.522029680540;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=17/35, seconds=322.662, peakMiB=956.931.

V149 closed: shared POR77.672952461232/TOTAL80.530276861749; communicate POR77.645461857202/TOTAL80.522029680540. Matched communication delta-0.027490604030; no gain. Both best17/stop35, exact144665 replay/frozen integrity, actual three-target assembly. Negative weights removed with SHAmanifest. Timing/concurrent total wall in timing.json; no serial equal-work benchmark. No further component-attention sweep; v150smoothCDF explicitly launched PID68944 after both terminal, one GPU. Best remains v140POR77.912103/TOTAL80.602022; strict808/fullOOF82 unmet, no otherfolds.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
