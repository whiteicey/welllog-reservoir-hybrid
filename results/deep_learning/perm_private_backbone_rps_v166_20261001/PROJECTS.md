# 进展

目标：主攻PERM并检验私有骨干突破，最终OOF>82。
已完成：真实body梯度/capture、冻结输出路由与分组LR、来源/五阶段/80-14记录。
计划/未完成：等待现有训练及预定义165对照释放槽位显式启动，初始/首轮/峰值显存、最佳恢复/实际组合、耗时/收益/清理。
可选：若有收益保留表示微调；无则换机制，不自动补折。

V166 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=249.026, peakMiB=1050.815; all144665replay/frozen exact.


Completed perm_conditional: best/stop=0/14, PERM=89.344760174753, no initial gain, actualTOTAL=80.591193779236; negative best/last deleted with SHA manifest after exact replay. Training=249.026s, launch/report wall=255.981s, peak=1050.815MiB; includes data/initial/final eval, other jobs overlap, not fullfivephase wall. Comparisons retain immutable configs/results/predictions.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
