目标：以固定消融检验采样泛化。已完成：实现和预定义对照。计划：第五折显式启动。未完成：最终恢复、实际组合、同容量delta、耗时图谱；完整OOF>82。达标后：严格TOTAL>80.8有收益后补折。
已启动PID16724、normal GPU，初始PERM68.383772436967与173一致，685873参数。64训练井锚点最少4379、最多13017，行采样井曝光0.756446%~2.248610%，新混合采样1.159473%~1.905555%。快照{'status': 'running', 'pid': 16724, 'stage': 'perm_conditional', 'epoch': 4, 'best': 88.0536673951041, 'best_epoch': 4, 'bad': 0, 'peak_MiB': 798.00732421875}，非最终成绩。
CUDA插值/attention反向warn_only非确定性警告，同173，未发生运行错误；不声称训练逐步位级一致。

V175 final PERM=88.893539547789, compactcontrolgain=-0.484854663355; actualTOTAL=80.433266559798, fivephases inclsource;strict808=False, no otherfolds. best/stop=21/35, seconds=294.565, peakMiB=798.007; all144665replay/frozen exact.

最终配对正收益：PERM88.893539547789，相对173 +0.045966462643；同容量/初始化/损失/80-14/Q/解码检查通过。初始均68.383772436967，随机窗口序列因采样机制不同，反向存在warn_only非确定性警告，不声称逐步位级一致或跨折稳定收益。实际TOTAL80.433266559798，best/stop21/35，144665磁盘恢复一致和冻结校验通过。低于v154-0.512025309130，保留正消融best/last，当前有效配方不替换、其他折未启动。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
