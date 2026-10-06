# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v174 条件CDF监督

## 探索重点
保持已证实小幅正收益的SAM0.05、效用0.1、128维Gaussian位置/权重读出，仅把joint RPS改为nondefault conditional RPS1。假设：gate冻结时，默认样本不应把条件连续密度也向默认值拉动。
对照v171PERM89.362085107146（同容量、来源初始化、预算、Q与解码）；绝对对照v15489.405564856919。

## 已探索与验证
已实现，未训练、无成绩。RPS默认样本置零，全行归一化不变，不做类别重新采样/重加权；NLL+BCE和全分布效用保持。自身标签、80/14/min10/b64/64步、cos80*64；最终恢复最佳并实际三头组合。

## 收益
尚未证实。来自原SOURCE而非171 checkpoint；总5阶段不增加。会同时报告局部配对差值和绝对分数，未完整五折。

## 不继续的点
本批只比较joint/conditional监督范围，不同时改结构、gate、scale或扫权重；若无收益关闭此变体。严格TOTAL>80.8有收益后才补折，原OOF>82未完成。

耗时：待实测；数据/初始/最终评价计入，不另做预热；并发按实际记录，训练时间不冒称全配方总墙钟。

v174显式启动PID75048，原SOURCE初始PERM89.344760174753、active82720；当前训练快照{'status': 'running', 'pid': 75048, 'stage': 'perm_conditional', 'epoch': 2, 'best': 89.3447601747529, 'best_epoch': 0, 'bad': 2, 'peak_MiB': 786.46533203125}，尚未证实收益。未启动其他任务或补折。

V174 final PERM=89.364985921273, compactcontrolgain=-0.013408289871; actualTOTAL=80.598272790518, fivephases inclsource;strict808=False, no otherfolds. best/stop=10/24, seconds=528.507, peakMiB=786.465; all144665replay/frozen exact.

最终条件CDF正消融：PERM89.364985921273，相对171 +0.002900814127；同容量/来源初始化/80-14/SAM/效用/Q/解码一致，只有CDF监督范围改变。实际TOTAL80.598272790518，best/stop=10/24，144665最佳恢复预测/冻结一致；保留正结果best/last。仍低于有效154，不替换现有配方、不启动其余折。




