目标：把采样局部收益迁移至有效读出机制。已完成：实现/固定配对/记录。计划：空槽后显式启动。未完成：训练/恢复/真实组合/delta/时间/图谱、完整OOF>82。达标后：严格80.8有收益才补折。
V180 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=333.678, peakMiB=786.465; all144665replay/frozen exact.

最终井均衡迁移到读出无益：PERM89.344760174753，相对174 -0.020225746520；best/stop0/14，actualTOTAL80.591193779236，同容量/初始化/loss/优化/Q/解码核验通过，144665恢复一致。清理负权重留SHA；保留175骨干原采样正收益，但不跨架构认定采样泛化；不扫比例、读出继续用174原行采样。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
