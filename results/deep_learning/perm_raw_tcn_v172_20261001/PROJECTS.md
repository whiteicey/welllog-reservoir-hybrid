目标：新269维输入、64维多尺度TCN分布修正编码器；原SOURCE保留冻结先验
已完成：实现及预定义对照。
计划：显式启动第五折，记录真实结果。
未完成：训练、恢复、实际组合、时间与图谱；完整OOF>82未达成。
达标后：严格TOTAL>80.8且有收益才补其余四折。
已显式启动PID73544，normal GPU；来源函数初始PERM89.344760174753，可训练参数54176，80/14。当前快照{'status': 'running', 'pid': 73544, 'stage': 'perm_conditional', 'epoch': 3, 'best': 89.3447601747529, 'best_epoch': 0, 'bad': 3, 'peak_MiB': 786.46533203125}；尚未最终恢复，收益尚未证实。
TCN输入269=原206+训练侧63幅值特征；不读取原隐藏特征作新编码器输入。SOURCE冻结分布先验保留；不是整模型冷启动替换；active54176与82720对照不等容量。

V172 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=160.280, peakMiB=786.465; all144665replay/frozen exact.

最终：PERM=89.344760174753，best/stop=0/14，无初始收益；实际TOTAL=80.591193779236。144665重载/冻结校验通过，训练160.280s，峰值786.465MiB；负权重清理留SHA清单。停止本轮仅修正位置/权重的原始特征TCN；不泛化到所有TCN，不扫宽度。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
