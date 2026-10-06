目标：独立PERM骨干突破。已完成：结构、训练侧初始化设计及记录。计划：启动第五折。未完成：训练/恢复/组合/时间/图谱；OOF>82未完成。后续：TOTAL严格>80.8且有收益后才补折。
训练快照（非最终）：v173 {'status': 'running', 'pid': 64464, 'stage': 'perm_conditional', 'epoch': 4, 'best': 87.79233514602183, 'best_epoch': 3, 'bad': 1, 'peak_MiB': 788.1005859375}；active685873。尚无最终磁盘恢复和三头组合，不改变当前配方。
CUDA插值及memory-efficient attention反向有warn_only非确定性警告，未报运行错误；不声称训练轨迹位级可复现，最终恢复预测一致性仍需验证。初始化68.383772436967，前2轮86.192730/87.420202；这是独立随机模型学习进展，非超过热启动对照的收益。

V173 final PERM=88.847573085146, compactcontrolgain=-0.530821125998; actualTOTAL=80.417178297873, fivephases inclsource;strict808=False, no otherfolds. best/stop=20/34, seconds=286.037, peakMiB=788.101; all144665replay/frozen exact.

最终独立骨干：PERM88.847573085146，相对v154 -0.557991771773，实际TOTAL80.417178297873；best/stop=20/34，active685873，144665磁盘恢复一致，训练286.037s，peak788.101MiB。虽然随机初始到学习后提升，但未超过来源/有效头、没有同容量消融正证据；不作为有效增益。清理负权重留SHA；本轮冷启动注意力骨干停止延伸，不泛化所有骨干。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
