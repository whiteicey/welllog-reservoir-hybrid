目标：整井输入上下文扩大私有PERM优势。已完成：实现/隔离约束/记录。计划：空槽后显式第五折。未完成：训练/恢复/实际组合/delta/时间/图谱、完整OOF>82；达标后严格80.8有收益补折。
v181已显式启动PID63344、normal GPU、临时SYSTEM_REQUIRED。当前训练快照{'status': 'running', 'pid': 63344, 'stage': 'perm_conditional', 'epoch': 33, 'best': 88.85628937090793, 'best_epoch': 27, 'bad': 6, 'peak_MiB': 826.8720703125}，active839761，非最终收益；max2任务/完整方法5阶段/未补折。
井bank只含原206输入，128深度采样、无label/井ID；额外条件编码器容量改变SAM全局梯度范数，不是等容量消融。初始函数保持，插值/attention反向warn_only警告如177，不声称位级训练复现。

V181 final PERM=88.856289370908, compactcontrolgain=-0.522104840236; actualTOTAL=80.420228997890, fivephases inclsource;strict808=False, no otherfolds. best/stop=27/41, seconds=639.703, peakMiB=826.872; all144665replay/frozen exact.

最终整井条件化无益：PERM88.856289370908，相对177 -0.114601466664；best/stop27/41，actualTOTAL80.420228997890。初始函数/预算/loss/SAM/采样/Q/解码一致，额外条件编码器容量改变SAM梯度范围、不是等容量消融；144665恢复/冻结验证通过。清理负权重留SHA；关闭本128点0.5FiLM设置，不扫点数/界限，不泛化全部整井上下文。原177保留，不补折。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
