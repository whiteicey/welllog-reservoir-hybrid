目标：纯DL、含源≤5训练阶段，原第五>80.8后再考虑完整OOF>82。
已完成：v108负对照结论；v109有限两项实现。
计划：显式串行geometry/content，最终best恢复、配对和实际三通道评分。
未完成：两项训练和增益、其他四折（不启动）。
达成后：新目录交付。

2026-10-01 本批已完成v108两项真实恢复与实际组合：posterior-control SW=-0.159649733443，关闭4统计输入；负模型/完整状态/特征缓存清理47.57MiB，源码/配置/预测/哈希/删除清单保留。两项串行从首次launch至最终report墙钟1180.004秒，单项642.404/472.643秒；缓存准备9.691秒单列，无并发加速结论。v109已显式启动geometry PID28976，已核实存活，epoch8最佳SW72.970691691062，峰值1807.24MiB；只是快照，content未启动。新增查询支路12672参数，全858585，输出零初始化、±0.25界，原窗口/pool/评分不变；从随机完整SW训练，全配方含原来源4阶段，无其他折或交付替换。配对记录器与图谱更新完成。目标第五>80.8/完整OOF>82仍未完成。

### geometry最终实际组合
SW=73.908896603003，TOTAL=80.400662301245，对旧最好-0.103535123232；strict808=False。本项best/stop=49/67，873.170秒，峰值1807.2MiB。仅原第五、4依赖训练阶段，来源算入，第5密度未执行；无其他折。

2026-10-01 本批完成：geometry SW73.908896603003，best49/stop67，873.170秒，峰值1807.24MiB；144665点实际重载一致，实际固定三通道TOTAL80.400662301245，4依赖阶段含源，低于当前local SW0.212856225155。配对content已显式启动PID23512并核实存活，epoch1/prior71.197115，峰值1795.52MiB；只是早期快照，尚无配对增益结论。两者同858585参数/其他初始化/预算，只改query输入。保留控制至最终配对；不再增加候选、不启动其他折、不改交付。实际组合、图谱和PROJECTS已更新；当前4阶段最好80.475161980049、完整OOF81.891019297225，目标仍未完成。

### content最终实际组合
SW=73.821136909607，TOTAL=80.369946408556，对旧最好-0.134251015921；strict808=False。本项best/stop=34/52，680.807秒，峰值1795.5MiB。仅原第五、4依赖训练阶段，来源算入，第5密度未执行；无其他折。

### SW逐点全局查询最终消融
content-geometry SW=-0.087759693396；same858585容量与其他参数初始化/原数据/Q50/预算，两项均实际恢复最佳。只有本逐点查询输入设置的单折证据，非完整OOF收益。
关闭本查询来源/残差边界设置，不自动扫头数或幅度，不泛化为所有记忆读取无效。

2026-10-01 v109已结束：content SW73.821136909607，geometry73.908896603003，匹配差值-0.087759693396，均低于local，关闭本查询来源/残差设置。负权重/完整状态清理36.91MiB并留清单；两项串行墙钟1658.677秒，单项873.170/680.807秒，含加载/预热/IO/启动间隔。v110转向同845913参数的分层动作奖励代理，复用v104/local真实控制；只训练loss改变，原验证动作argmax不改，含源4阶段，尚无成绩。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
