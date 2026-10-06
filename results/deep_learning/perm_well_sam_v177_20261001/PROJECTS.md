目标：扩大井均衡骨干局部收益。已完成：单因素方案与实现。计划：显式训练第五折。未完成：最终恢复/组合/消融delta/时间/图谱/原OOF>82；达标后按80.8门槛补折。
v177已显式启动PID14800，normal GPU/临时SYSTEM_REQUIRED，active685873。当前训练快照{'status': 'running', 'pid': 14800, 'stage': 'perm_conditional', 'epoch': 12, 'best': 88.34151401360728, 'best_epoch': 6, 'bad': 6, 'peak_MiB': 800.61279296875}，非最终收益；最多2GPU、完整方法5阶段、其余四折未启动。
原私有初始化函数与175一致，完整网络SAM梯度范围与旧独立头不同；CUDA插值/attention反向warn_only非确定性警告如对照，未出现运行错误，最终实际恢复待做。

V177 final PERM=88.970890837572, compactcontrolgain=-0.407503373572; actualTOTAL=80.460339511223, fivephases inclsource;strict808=False, no otherfolds. best/stop=24/38, seconds=551.783, peakMiB=800.613; all144665replay/frozen exact.

最终完整骨干SAM局部正收益：PERM88.970890837572，相对175 +0.077351289783；同模型/初始化/采样/loss/Q/预算/优化步数核验通过；SAM2遍vsAdamW1遍，不等算力。实际TOTAL80.460339511223，best/stop24/38，144665磁盘恢复/冻结一致；保留正结果best/last。仍低于有效154，不替换配方/补折。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
