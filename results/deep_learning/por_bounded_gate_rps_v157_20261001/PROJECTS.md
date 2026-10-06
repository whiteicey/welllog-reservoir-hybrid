# v157 进展

目标：POR有效读出/RPS1上增补小门控扩大收益；其他头主攻PERM，最终完整OOF>82。

已完成：33输出头/±0.125新门控、旧预测器冻结和现有自身联合损失实现、PLAN与固定PERM154/SW104组合约定。

计划/未完成：显式启动、真实训练观察；恢复最佳/实际组合、收益与时间显存证据、必要清理和图谱更新。

可选：有收益保留门控与RPS；无收益不自动扫界或叠加负模块，不自动补折。


## 实际启动快照

已核验PID25516正常训练，normal GPU，TEMPORARY_SYSTEM_REQUIRED=True，与PERM158并行。第28轮快照：最佳POR77.731336011420（第17轮），尚低于正控制77.926380760724；训练未结束，不构成最终收益。
未完成：最佳磁盘重载、实际三通道组合、最终差值及耗时记录。

V157 joint_rps: restored POR=77.763278807566; delta vs v114=+0.215209312673, vs historical=+0.211342005681; actual fifth TOTAL=80.563544832047;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=30/48, seconds=599.604, peakMiB=924.223.


## 最终收益和关闭

与正RPS控制差值-0.163101953158，最佳/停止30/48；实际TOTAL80.563544832047，144665行恢复和冻结完整性已验证。无新增收益，不继续扫描本机制；负权重按SHA清单删除，证据保留。训练599.604秒，与其他任务并行；不作独立速度结论。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
