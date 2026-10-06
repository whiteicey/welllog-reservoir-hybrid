# 进展

目标：主攻PERM，同时检验表示/骨干适配是否突破，最终完整OOF>82。
已完成：有界双私有7点适配器/真实梯度路径/零函数初始化和记录。
计划/未完成：等待两GPU槽位空出后显式启动，无自动排队；核验真实初始/首轮、最佳恢复/实际合并、收益/总墙钟/显存/清理证据。
可选：有收益继续表示适配，负则更换机制，不自动补折。


## 实际启动与首轮验证

PID59780已核验存活，normal GPU、临时SYSTEM_REQUIRED、与POR160双并发；初始PERM89.344760174753与原函数一致，99936活跃参数。已完成第2轮，最佳仍初始，第2轮89.313387736773，peak786.465MiB，stderr无错误。训练的两组正梯度和冻结签名检查已通过，80/14配置确认。尚未证实收益，待最佳磁盘恢复/实际三通道组合；图谱已更新。


## 早停预算适用范围澄清

历史正控制154前15轮未超过初始，第16轮首次改善；pat14会截断这条已知有效轨迹。14轮停于初始只支持当前配置/预算未取得收益，不是所有同类机制无效证据。此诊断不放宽用户pat14，不自动恢复。

V162 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=212.983, peakMiB=786.465; all144665replay/frozen exact.


## 最终结论与关闭

80/14预算下PERM89.344760174753，无初始改善；相对历史有效154−0.060804682166，best0/stop14。实际TOTAL80.591193779236，144665行最佳重载与冻结精确，两梯度组为正。当前配置/预算无收益，历史154需16轮首改善，不能泛化全部适配器无效。关闭本配置，负权重SHA清单清理，其他证据保留。训练212.983秒，启动至报告墙钟219.845秒，数据/初始/最终评估计入，与POR160并行；不是完整五阶段耗时。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
