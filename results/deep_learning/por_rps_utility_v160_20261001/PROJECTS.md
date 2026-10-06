# 进展

目标：扩大POR读出正收益，完整OOF>82。
已完成：同容量效用损失增补、四项记录和预定义正控制。
计划/未完成：等待POR157完成后显式启动、恢复最佳/实际合并/耗时/显存/结论。
可选：有收益保留；负则换机制，不自动补折。


当前已启动PID62172，临时SYSTEM_REQUIRED正常；冻结源码仍18，用户新patience14由专项轮间停止请求执行。原NLL/RPS1+容差效用0.1，等待初始/首轮验证。


## 首轮真实验证

v160初始POR77.540454326739，与同父初始一致，230944可训练参数。已完成第2轮，暂无收益，峰值924.549MiB；stderr无错误。PID62172训练与PID68380非GPU早停监控均核验，监控仅写bad14轮间停止请求，无恢复/队列。图谱curate/build成功更新。


## 训练中快照

状态running，第16轮，最佳77.571541627349，最佳轮12，连续未改善4。进程已实际核验存活；尚未超过正控制，不构成最终收益。下一项v162局部表示适配已准备，未启动、不占第三GPU任务；本批atlas已更新。

V160 joint_rps: restored POR=77.873725032741; delta vs v114=+0.325655537848, vs historical=+0.321788230856; actual fifth TOTAL=80.596678699599;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=30/45, seconds=861.459, peakMiB=924.549.


## 最终结论

POR77.873725032741，正RPS1差值−0.052655727983；实际三头TOTAL80.596678699599。best30/stop45；第44轮bad14触发专项停止，下一轮已开始，实际bad15停；不冒称恰好14轮。144665最佳恢复/冻结精确已验证。关闭效用0.1配置，不否定所有效用；负权重SHA清单删除，其他证据保留。训练861.459秒，启动到报告墙钟869.027秒，含数据/初始/最终评估，与PERM多项串接并行、最多2GPU；不是全五阶段耗时。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
