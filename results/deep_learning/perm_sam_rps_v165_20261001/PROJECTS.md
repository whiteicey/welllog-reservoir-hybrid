# 进展

目标：在有效PERM读出上检验优化器增益，最终OOF>82。
已完成：预定义同容量AdamW/SAM0.05与80/14、原子恢复/精确回滚复用。
计划/未完成：待GPU空槽逐项显式启动、真实最佳恢复/合并、公平差值及两项总墙钟/显存/清理、图谱。
可选：有收益保留优化器，无则换机制，不自动补折。


已完成实际三通道组合/同预算两项差值记录入口record_perm_sam_rps_v165.py；仍未启动，无成绩，等待现有两GPU槽位。


## 显式启动控制项

POR164 linear已完成，暂保留权重至配对比较结束；非线性仍运行。已用空槽显式启动165 AdamW控制，SAM尚未启动，无第三GPU任务或自动队列。私有骨干166已准备但尚无训练/分数，等待现有预定义对照。

V165 adamw: actual PERM=89.344760174753, TOTAL=80.591193779236; delta historicalRPS=-0.060804682166; best/stop=0/14, seconds=131.987, peakMiB=786.465; exact restored144665/frozen,5phases,no otherfolds. Pair gain not claimed until both finish.

V165 sam: actual PERM=89.352248368881, TOTAL=80.593814647181; delta historicalRPS=-0.053316488038; best/stop=10/24, seconds=397.458, peakMiB=786.465; exact restored144665/frozen,5phases,no otherfolds. Pair gain not claimed until both finish.

V165 completed matched80/14 pair: SAM-AdamW PERM=+0.007488194128. Equal updates, unequal compute; original fifth only, not fullOOF.


Completed adamw: best/stop=0/14, PERM=89.344760174753, no initial gain, actualTOTAL=80.591193779236; negative best/last deleted with SHA manifest after exact replay. Training=131.987s, launch/report wall=140.226s, peak=786.465MiB; includes data/initial/final eval, other jobs overlap, not fullfivephase wall. Comparisons retain immutable configs/results/predictions.


## 最终SAM机制收益

同80/14、82720参数、初始化、数据/评分，SAM−AdamW PERM=+0.007488194128；SAM89.352248368881，仍低于历史有效154的89.405564856919。小幅单折单seed实测机制收益，保留SAM最佳和last；不升级为SOTA/完整OOF收益，继续一项门控范围验证。SAM每更新双前后向，控制一次，等更新不等计算；完成两项总墙钟518.049秒、单项耗时/显存见timing.json。错开启动且其他探索穿插，不作纯两任务并发加速结论。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
