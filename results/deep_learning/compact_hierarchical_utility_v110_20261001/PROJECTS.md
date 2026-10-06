目标：纯DL、含源≤5阶段、第五>80.8后再补原五折OOF>82。
已完成：v109负对照结论；v110有限损失实现和历史控制定义。
计划：显式启动一项，最终真实恢复、控制差值和实际组合。
未完成：训练/收益/其余四折（不启动）。
达成后：新目录交付。

2026-10-01 本批完成：v109内容查询SW73.821136909607、位置73.908896603003，差值-0.087759693396，均低于local，关闭本设置；负权重/完整状态清理36.91MiB，留源码/配置/预测/实际组合/删除清单。查询两项串行墙钟1658.677秒，单项873.170/680.807秒，含加载/预热/IO与启动间隔，无并发加速结论。v110改训练效用代理，原845913参数local结构/初始化/80-18/Q50/评分与验证argmax保持，复用v104/local历史平坦控制避免重训。hierarchical PID51128已核实存活，epoch8 SW最佳72.998922466155、峰值1721.17MiB；非最终收益，含源4依赖阶段。记录器、独立记录和图谱已更新；第五>80.8/完整OOF>82仍未完成，不启动其他折或修改旧交付。

最终：SW=74.002091084435，hierarchical-flat=-0.119661743722，实际TOTAL=80.433280369746，strict808=False；含源4训练阶段，仅原第五，完整OOF未验证。best/stop=34/52，677.367秒，峰值1721.2MiB。

2026-10-01 v110结束：SW74.002091084435，比flat -0.119661743722，实际TOTAL80.433280369746，best34/stop52，677.367秒、峰值1721.17MiB；关闭当前损失设置。负权重/last清理18.20MiB并留清单。v111改为PERM原均值更新范围，热启动同紧凑父checkpoint，gate_route/mean_coadapt两项，均lr3e-5，候选只多2064活跃均值变量；骨干/混合权重/尺度固定。源/POR/PERM初训练/PERM细化/SW共5阶段，尚无成绩，无其他折或交付覆盖。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
