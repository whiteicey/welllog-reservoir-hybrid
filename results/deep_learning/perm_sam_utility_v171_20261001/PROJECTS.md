目标：SAM/RPS加PERM log10误差三角效用，系数0.1
已完成：实现及预定义对照。
计划：显式启动第五折，记录真实结果。
未完成：训练、恢复、实际组合、时间与图谱；完整OOF>82未达成。
达标后：严格TOTAL>80.8且有收益才补其余四折。
已显式启动PID39440，normal GPU；来源函数初始PERM89.344760174753，可训练参数82720，80/14。当前快照{'status': 'running', 'pid': 39440, 'stage': 'perm_conditional', 'epoch': 1, 'best': 89.3447601747529, 'best_epoch': 0, 'bad': 1, 'peak_MiB': 786.46533203125}；尚未最终恢复，收益尚未证实。
训练效用系数0.1记录于initialization.train_only_expected_logratio_utility_weight；config.utility_weight是继承的旧SW alpha占位0，本实验实际PERM损失为NLL+BCE+RPS1+0.1效用。SAM半径0.05，2次前后向。

训练快照（非最终）：v171 {'status': 'running', 'pid': 39440, 'stage': 'perm_conditional', 'epoch': 14, 'best': 89.36208510714584, 'best_epoch': 10, 'bad': 4, 'peak_MiB': 786.46533203125}；active82720。尚无最终磁盘恢复和三头组合，不改变当前配方。

V171 final PERM=89.362085107146, compactcontrolgain=-0.016309103998; actualTOTAL=80.597257505573, fivephases inclsource;strict808=False, no otherfolds. best/stop=10/24, seconds=588.786, peakMiB=786.465; all144665replay/frozen exact.

最终局部正收益：PERM89.362085107146，同容量/初始化/80-14/SAM半径/步数/Q/评分对照v165差值+0.009836738265，但比v154 -0.043479749773。实际TOTAL80.597257505573，best/stop=10/24；144665恢复一致与冻结核验通过。保留best/last正结果。额外效用CDF计算使等步数不等计算量；不声称完整OOF收益，不替换有效配方。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
