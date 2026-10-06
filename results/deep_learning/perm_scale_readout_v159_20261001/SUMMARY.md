# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# PERM v159：有界分量尺度修正

## 探索重点

Gaussian128位置/权重与RPS1已有效，但分量尺度冻结。新头输出48维，增加16个原始尺度修正，界为±0.5；softplus+.025仍保证尺度正值。门控/骨干冻结，properNLL和原CDF/RPS使用同一修正尺度。

## 已探索内容与验证

实现完成，尚未启动、无分数。与v154同来源/种子/初始化函数/损失/预算，新增2064参数；不是严格同容量消融，不加载已训练v154头。最终固定POR151/SW104实际合并。总五阶段。

## 收益

尚未证实；预定义对照PERM89.405564856919，分别报告自身分数及实际TOTAL。

## 不值得继续的点及证据

Laplace试验仍在训练，不能下最终结论；不继续宽度或RPS系数网格。本项无收益不自动扫描尺度界。


## 已核验实际启动

PID44704，normal GPU，临时SYSTEM_REQUIRED已启用；与POR157双并发。初始PERM89.344760174753与Gaussian原函数一致，84784可训练参数；第2轮快照尚无收益，峰值786.465MiB，无stderr错误。待最终恢复/实际合并验证；演进图谱已更新。

V159 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/18, seconds=178.450, peakMiB=786.465; all144665replay/frozen exact.


## 最终收益和关闭

与正RPS控制差值-0.060804682166，最佳/停止0/18；实际TOTAL80.591193779236，144665行恢复和冻结完整性已验证。无新增收益，不继续扫描本机制；负权重按SHA清单删除，证据保留。训练178.450秒，与其他任务并行；不作独立速度结论。

单项启动到报告总墙钟185.939秒，数据加载/首次预热包括在内；其余正控制/来源阶段复用未重训，不将该时间当成完整五阶段训练耗时或并发总加速证据。




