# 进展

目标：在14轮早停规则内扩大PERM有效读出收益，最终OOF>82。
已完成：历史首增益16轮诊断、低LR1e-4方案。
计划/未完成：空槽显式启动、实际最佳恢复/合并验证、真实增益/时间/显存、权重留存决策。
可选：若有收益继续优化，不自动补折。


## 实际启动核验

PID27280存活，normal GPU/临时SYSTEM_REQUIRED，与POR160双并发。初始89.344760174753，与原函数一致；82720参数，首轮89.327379522056，暂无收益。实际配置max80/pat14/lr1e-4确认；首轮梯度/冻结检查通过，peak786.465MiB，无stderr错误。等待实际最佳恢复/组合/最终增益结论，图谱已更新。

V163 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=184.407, peakMiB=786.465; all144665replay/frozen exact.


## 最终结论

PERM89.344760174753，初始最佳/停止14，无学习改善；正RPS控制差值−0.060804682166，实际TOTAL80.591193779236。144665最佳恢复/冻结精确；低LR1e-4在80/14仍无收益，不继续LR网格。仅限此预算/配置，不否定所有学习率优化。负权重SHA清单清理，证据保留。训练184.407秒，启动到报告墙钟191.893秒，数据/初始/最终评估计入，与POR160并行后短暂独占；不是五阶段总耗时。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
