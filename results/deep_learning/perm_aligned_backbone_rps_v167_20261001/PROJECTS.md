# 进展

目标：PERM表示保留与任务适配兼顾，最终OOF>82。
已完成：训练侧teacher隐藏约束/独立hooks/同活跃容量与80-14对照、来源记录。
计划/未完成：空槽显式启动、初始/首轮真实梯度及显存、最佳重载/实际合并/同预算差值、耗时和清理/图谱。
可选：正收益保留，无则换机制，不自动补折。


## 实际启动与训练验证

normalGPU/临时SYSTEM_REQUIRED，已核验进程存活；原SOURCE初始PERM89.344760174753，活跃参数1355414，配置80/14确认。实际快照：{"pid": 25140, "status": "complete", "epoch": null, "best": null, "best_epoch": 0, "bad": null}；峰值1072.6279296875MiB，训练梯度和冻结签名检查已通过。尚未证实最终增益。骨干路径有PyTorch warn_only线性插值反传非确定性提示，不是运行失败；保持真实记录，不声称训练轨迹严格确定。

V167 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=235.218, peakMiB=1072.628; all144665replay/frozen exact.

最终结论：v167 PERM=89.344760174753，相对v166 +0.000000000000；实际三头TOTAL=80.591193779236。best/stop=0/14，80/14；144665点磁盘恢复预测及冻结参数核验通过。未超过有效v154，不替换配方、不补折。不继续本轮0.1隐藏锚定系数扫描；结论仅限该设置/14轮早停。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
