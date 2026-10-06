目标：含源最多5阶段，原第五>80.8后补原五折OOF>82。
已完成：PERM有限更新范围对照实现，父来源/阶段成本定义。
计划：显式启动两项、按实测内存/墙钟选择并发，实际恢复与固定三通道评分。
未完成：训练/收益/其余四折（不启动）。
达成后：新目录交付。

gate_route最终：PERM=89.378394211144，对父=+0.000000000000；实际TOTAL=80.475161980049，strict808=False。执行5阶段、选定有效依赖4，含源，仅第五，未补其他折。best/stop=0/18，95.216秒，峰值786.5MiB。

mean_coadapt最终：PERM=89.378394211144，对父=+0.000000000000；实际TOTAL=80.475161980049，strict808=False。执行5阶段、选定有效依赖4，含源，仅第五，未补其他折。best/stop=0/18，97.187秒，峰值786.5MiB。

最终更新范围对照：mean-gate_route PERM=+0.000000000000；候选多2064活跃均值变量，非同训练容量。两项从同一个已选PERM函数开始，仅原第五证据，非完整OOF收益。

2026-10-01 v111两项已结束：gate_route和mean_coadapt均PERM89.378394211144，均best0/stop18，匹配增益0，真实重载/冻结参数一致；实际三通道仍TOTAL80.475161980049。执行含源5阶段，选定有效依赖仍4（父函数），不能声称细化有收益。两项单项95.216/97.187秒、峰值均786.47MiB；总墙钟218.387秒，launch至report时间重叠0.000秒，实际模式serial，不能预宣称双并发或加速。重复/负权重和状态清理36.30MiB并留清单，父PERM/current best/交付保留。关闭本3e-5更新范围，不扩大密度尺度/权重或扫LR。无GPU任务仍运行，本批完成，目标原第五>80.8与原OOF>82未完成，后续换机制，不补其他折。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
