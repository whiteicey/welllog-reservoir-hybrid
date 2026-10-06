目标：尾部分布稳健化改善PERM。已完成：配套PDF/CDF/utility/decoder及记录。计划：显式第五折。未完成：训练/最终恢复/实际组合/差值/时间/图谱、完整OOF>82；达标后严格80.8有收益补折。
v182已显式启动PID85420、normal GPU、临时SYSTEM_REQUIRED。当前训练快照{'status': 'running', 'pid': 85420, 'stage': 'perm_conditional', 'epoch': 5, 'best': 89.33653883690049, 'best_epoch': 0, 'bad': 5, 'peak_MiB': 786.46533203125}，active82720，非最终收益；max2任务/完整方法5阶段/未补折。
初始PERM89.33653883690049，Gaussian原SOURCE89.3447601747529；CDF变化导致初始函数不同，报告最终score和对各自初始的变化，不能把初始PDF差异当作训练增益。

V182 final PERM=89.336538836900, compactcontrolgain=-0.041855374243; actualTOTAL=80.588316310987, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=327.517, peakMiB=786.465; all144665replay/frozen exact.

最终t5分布无益：PERM89.336538836900，相对174 -0.028447084372；自己初始89.336538836900、学习增益0、best/stop0/14，actualTOTAL80.588316310987。参数/优化/预算/Q一致但初始CDF不同、log分量矩匹配；PDF/CDF/utility/decoder同族，144665恢复一致/冻结通过。负权重清理留SHA；保留Gaussian，不扫df或混合族，结论仅限本df5设置。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
