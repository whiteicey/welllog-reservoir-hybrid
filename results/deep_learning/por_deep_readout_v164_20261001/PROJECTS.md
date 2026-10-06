# 进展

目标：在有效POR读出内部寻找深度增益，最终完整OOF>82。
已完成：同容量linear/nonlinear实现和预定义对照、来源/预算记录。
计划/未完成：空槽显式逐项启动，原始最佳恢复、实际组合、公平差值/耗时显存/清理、图谱。
可选：若有匹配收益保留，无收益换思路；不自动補折。


## 实际双并发启动

linear PID40840、nonlinear PID67592均核验存活、normalGPU/临时SYSTEM_REQUIRED。初始两项POR77.540454326739，与同父函数一致；296864参数、max80/pat14均确认，stderr无错误。快照：{"linear": {"pid": 40840, "epoch": 6, "best": 77.54045432673898, "best_epoch": 0, "bad": 6}, "nonlinear": {"pid": 67592, "epoch": 2, "best": 77.54045432673898, "best_epoch": 0, "bad": 2}}。无最终收益结论，待两项恢复/实际组合/公平差值；图谱已更新。


## 真实中期快照

{"linear": {"status": "running", "epoch": 16, "best": 77.58681772050492, "best_epoch": 12, "bad": 4}, "nonlinear": {"status": "running", "epoch": 11, "best": 77.54045432673898, "best_epoch": 0, "bad": 11}}。两项进程实际核验存活；线性版本已改善初始但仍未超正控制，非线性暂无增益。匹配消融尚未结束，不声称最终机制收益。已准备PERM165 AdamW/SAM0.05同预算两项，未启动、无第三GPU任务。图谱已更新。


## 同轮前缀诊断（非最终消融）

两项共同完成19轮：线性前缀最佳77.808192160603，非线性77.833969698770，非线性−线性+0.025777538167；均未达到正控制77.926380760724。两进程已核验live，无stderr错误。只作训练中快照，未完成不能声称最终机制增益。

V164 linear: restored POR=77.891844743355; delta vs v114=+0.343775248462, vs historical=+0.339907941470; actual fifth TOTAL=80.602114612783;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=30/44, seconds=670.027, peakMiB=925.553.

V164 nonlinear: restored POR=77.885830334051; delta vs v114=+0.337760839158, vs historical=+0.333893532166; actual fifth TOTAL=80.600310289992;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=30/44, seconds=697.551, peakMiB=925.399.


## 最终配对结论

线性POR77.891844743355，非线性77.885830334051；同容量/初始化/80-14非线性−线性-0.006014409304，无非线性收益，两者也低于正RPS控制77.926380760724。两项最佳30/停止44，全144665恢复和实际三头合并、冻结精确。早期前缀正差值不是最终增益，已被最终结果推翻。关闭此65920参数/界0.1残差结构，不泛化全部深头无效。四个负权重按SHA清单删除，证据保留。两项总体启动到最后报告墙钟739.384秒，单项训练/显存见timing.json；错开启动双并发，含数据/预热/评估，不当作完整五阶段训练时长。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
