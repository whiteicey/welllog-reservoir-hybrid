# 进展

目标：扩展已测SAM局部收益到PERM默认门控，完整OOF>82。
已完成：SAM+原gate更新范围方案、预定义对照/来源/五阶段/80-14。
计划/未完成：空槽显式启动，真实最佳恢复/实际组合、三个差值与耗时/显存、清理/图谱。
可选：若实测正收益保留，负则换机制，不自动补折。


## 实际启动与训练验证

normalGPU/临时SYSTEM_REQUIRED，已核验进程存活；原SOURCE初始PERM89.344760174753，活跃参数82978，配置80/14确认。实际快照：{"pid": 29440, "status": "running", "epoch": 11, "best": 89.35085035866882, "best_epoch": 10, "bad": 1}；峰值786.46533203125MiB，训练梯度和冻结签名检查已通过。尚未证实最终增益。骨干路径有PyTorch warn_only线性插值反传非确定性提示，不是运行失败；保持真实记录，不声称训练轨迹严格确定。

V168 final PERM=89.350850358669, compactcontrolgain=-0.027543852475; actualTOTAL=80.593325343606, fivephases inclsource;strict808=False, no otherfolds. best/stop=10/24, seconds=346.955, peakMiB=786.465; all144665replay/frozen exact.

最终结论：v168 PERM=89.350850358669，相对v165 -0.001398010212；实际三头TOTAL=80.593325343606。best/stop=10/24，80/14；144665点磁盘恢复预测及冻结参数核验通过。未超过有效v154，不替换配方、不补折。门控增补未扩大SAM收益；保留v165单独SAM小幅正收益，关闭本轮门控增补，不扫描门控学习率。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
