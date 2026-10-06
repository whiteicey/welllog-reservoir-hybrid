# 进展

目标：扩大PERM收益，最终完整OOF>82。
已完成：默认门控诊断驱动方案/实际可微门控/两组LR/14耐心值。
计划/未完成：显式启动，真实训练、最佳重载/实际组合、收益/耗时/显存记录和必要清理。
可选：有收益研究PERM门控/表示；无收益换思路，不自动补折。


## 首轮启动核验

PID50992，normal GPU/临时SYSTEM_REQUIRED，与POR160双并发。初始PERM89.344760174753与原函数一致；首轮89.307024687938，暂未增益；82978可训练参数、peak786.465MiB。已完成首轮意味着两组正梯度/冻结签名检查通过，配置80/14确认，stderr无错误。待最终最佳重载/实际三头合并，不将快照算最终收益。图谱已更新。


## 训练中快照

状态running，第9轮，最佳89.344760174753，最佳轮0，连续未改善9。进程已实际核验存活；尚未超过正控制，不构成最终收益。下一项v162局部表示适配已准备，未启动、不占第三GPU任务；本批atlas已更新。

V161 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=203.253, peakMiB=786.465; all144665replay/frozen exact.


## 最终结论

PERM89.344760174753，相对Gaussian/RPS正控制−0.060804682166，初始最佳，无学习改善；best0/stop14，14轮早停规则已实际生效。实际三头TOTAL80.591193779236，144665点最佳恢复/冻结一致。两个梯度组均正，但无成绩收益，关闭本门控更新配置。负权重SHA清单删除，来源/配置/日志/预测/结果保留。训练203.253秒，启动至报告墙钟210.845秒，与POR160并行、包含数据/初始/最终评估，不当作完整五阶段耗时。


## 早停预算适用范围澄清

历史正控制154前15轮未超过初始，第16轮首次改善；pat14会截断这条已知有效轨迹。14轮停于初始只支持当前配置/预算未取得收益，不是所有同类机制无效证据。此诊断不放宽用户pat14，不自动恢复。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
