目标：含源≤5阶段，原第五折>80.8与完整OOF>82。
已完成：有限两项实现；730268行统计全覆盖、仅训练行拟合；准备9.691秒。
计划：显式串行control/posterior，选定best重载、同容量差值和固定三通道实际组合。
未完成：训练和增益、其余四折（不启动）。
达成后：新目录交付。

2026-10-01 v108 curve_control PID55952已实际启动，epoch4 SW最佳71.779416、峰值1726.49MiB；只是快照，posterior尚未启动，无自动队列。缓存准备9.691秒，阶段源/POR/PERM/新SW共4，其他折不启动。v107完成不替换当前4阶段TOTAL80.475162；dot负模型清理18.21MiB并留清单，cosine局部正对照及当前最佳保留。

2026-10-01 本批更新：v108曲线对照PID55952已核实存活，第37轮、best34 SW73.843376994495、峰值1726.49MiB；非最终收益，posterior仍未启动。完成记录器语义/阶段来源修正、清理复制的v107计划元数据（初始版本归档），新增老师训练井内预测的分布偏移限制说明，训练源码/配置未改；图谱已更新快照和真实最终组合入口。目标原第五>80.8及完整OOF>82仍未完成，当前精简4阶段TOTAL80.475161980049和旧交付不变。待本项实际结束后显式启动预定义posterior，无自动队列和其他折。

### curve_control最终实际组合
SW=73.843376994495，TOTAL=80.377730438267，对旧最好-0.126466986210；strict808=False。本项best/stop=34/52，642.404秒，峰值1726.5MiB。仅原第五、4依赖训练阶段，来源算入，第5密度未执行；无其他折。

2026-10-01 本批完成：curve_control最佳SW73.843376994495（best34/stop52，642.404秒，峰值1726.49MiB），实际三通道TOTAL80.377730438267，144665行恢复预测一致；比当前local低0.278375833663。预定义por_posterior已显式启动PID3552，当前epoch2/prior71.197115、峰值1726.49MiB；非最终成绩。两项严格串行，同846297参数、其他网络初始化/预算，仅4维输入来源改变，配对差值等待本候选最终恢复。控制权重暂留用于配对，不开启更多候选或其他折；原4阶段最好80.475161980049、完整OOF81.891019297225和旧交付保留。图谱、实际组合报告和PROJECTS已更新。

### por_posterior最终实际组合
SW=73.683727261052，TOTAL=80.321853031562，对旧最好-0.182344392915；strict808=False。本项best/stop=20/38，472.643秒，峰值1726.5MiB。仅原第五、4依赖训练阶段，来源算入，第5密度未执行；无其他折。

### POR语义输入最终消融
posterior-control SW=-0.159649733443；same846297容量与其他参数初始化/原数据/Q50/预算，两项均实际恢复最佳。只有本4维POR语义输入设置的单折证据，非完整OOF收益。
关闭本4维POR语义统计设置；不自动扫描更多老师或统计，不泛化为所有语义特征无效。

2026-10-01 v108结束：posterior SW73.683727261052，对control73.843376994495为-0.159649733443，均低于local，关闭4统计设置；负检查点/完整状态及特征缓存清理47.57MiB并留清单。两项串行墙钟1180.004秒，含启动间隔/加载/预热/评价/IO，缓存此前9.691秒另计，不与历史来源成本混同。v109换为逐点读取全局SW记忆，同容量geometry/content两项，新增12672参数零输出、残差±0.25，原pool/window/scoring保持，含源4依赖训练阶段；尚未训练，不补其他折。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
