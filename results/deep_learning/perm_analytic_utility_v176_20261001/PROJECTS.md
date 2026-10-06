目标：解析效用扩大读出小幅增益。已完成：实现、解析推导、边界说明。计划：174最终恢复后显式启动。未完成：训练、score、恢复、组合、配对差值、耗时及图谱。目标达成后：严格80.8补折，原OOF>82。
v176已显式启动PID66468，normal GPU/临时SYSTEM_REQUIRED，active82720。当前训练快照{'status': 'running', 'pid': 66468, 'stage': 'perm_conditional', 'epoch': 4, 'best': 89.3447601747529, 'best_epoch': 0, 'bad': 4, 'peak_MiB': 786.46533203125}，非最终收益；最多2GPU、完整方法5阶段、其余四折未启动。

V176 final PERM=89.359879687511, compactcontrolgain=-0.018514523632; actualTOTAL=80.596485608701, fivephases inclsource;strict808=False, no otherfolds. best/stop=10/24, seconds=609.569, peakMiB=786.465; all144665replay/frozen exact.

最终解析效用负替换：PERM89.359879687511，相对174 -0.005106233761；actualTOTAL80.596485608701，best/stop10/24；144665磁盘恢复预测及冻结一致，同容量/初始化/预算/Q/解码检查通过。负权重已清理留SHA；保留174分箱效用不替换配方，不扫描积分精度。仅这次训练效用替换负结果，不泛化Gaussian解析积分无效；并发改变计算资源，不能凭单项609.569s声称积分实现更慢。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
