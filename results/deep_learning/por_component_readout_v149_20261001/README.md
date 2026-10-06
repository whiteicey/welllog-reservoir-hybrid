# v149 分量共享与协同分布读出

探索重点：检验不同分量联合决定位置与概率质量是否优于独立分量读出。原256隐层特征投影为16个16维分量token，加入分量身份和冻结父模型的log权重、位置、log参考宽度、分布族标识；共享解码器输出每分量位置/权重修正。旧模型预测只作为同一模型内部的冻结分布参数，不使用外部模型、伪标签或旧18阶段预测。

对照条件：shared/communicate结构、参数、初始化、种子、优化器、损失完全相同；只改变分量注意力连通性为对角或全连接。shared中Q/K仍登记同容量，但对角注意力不依赖Q/K，实际有效梯度范围不同，这是连接消融的固有差异。两组均从冻结v113原函数开始，不加载v140训练头。新增末层为零，初始函数相同；单折实际恢复后再判定收益。

已经探索与验证：代码实现完成；尚未训练、无成绩。新增条件位置修正不超过原参考宽度，log权重修正±1；门控、尺度、族与原预测器冻结。相同原9曲线/206特征+63幅值/原64训练井统计/Q50/第五折/seed20261010/Student3+Cauchy自身POR似然；区间与动作解码不变。normal GPU/max2/80轮上限、patience18/min10/bs64/64步/3e-4/cos80*64；原子完整last，显式恢复，进程级临时防睡眠。

取得收益：尚未证实。机制对照比较communicate-shared；绝对比较最佳256读出77.912102888582。结构容量相对v140不同，不能将绝对差值全归因于注意力。实际固定PERM136/SW104合并验证，不用其他目标真值代替预测构造TOTAL。五阶段为source+POR113+POR149+PERM136+SW104，严格第五折TOTAL>80.8且明确收益后才能补其他折，完整OOF>82未完成。

不值得继续的点：宽度192/384/512均未超256；v145尺度修正-0.235107；v148逐区间密度修正-0.023091。关闭这些具体配置，不把负结果泛化为所有分布读出无效。当前仅预定义两个连接消融，不扩成自动候选队列，不恢复准备中的骨干路线，不覆盖交付。

结束时记录：实际恢复POR/三通道TOTAL/144665覆盖/冻结完整性；单项训练时间、并发总墙钟、峰值显存，加载预热是否计入；完成无亮点候选清理权重并留SHA清单，正控制与依赖保留。

Verified explicit dual launch: shared PID45244 and communicate PID22848, both command lines and actual training epochs confirmed live. Both initial POR77.540454326739 and290610 trainableparameters. Snapshot shared epoch6, communicate epoch4, no improvement yet; peak956.931MiB each, noOOM. Shared epochs ~6.4s and communication ~8.9s under overlap; no completed equal-work wall comparison yet. Final gain/actual assembly pending; do not promote.

V149 shared: restored POR=77.672952461232; delta vs v114=+0.124882966339, vs historical=+0.121015659347; actual fifth TOTAL=80.530276861749;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=17/35, seconds=320.591, peakMiB=956.931.

V149 communicate: restored POR=77.645461857202; delta vs v114=+0.097392362309, vs historical=+0.093525055317; actual fifth TOTAL=80.522029680540;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=17/35, seconds=322.662, peakMiB=956.931.

V149 closed: shared POR77.672952461232/TOTAL80.530276861749; communicate POR77.645461857202/TOTAL80.522029680540. Matched communication delta-0.027490604030; no gain. Both best17/stop35, exact144665 replay/frozen integrity, actual three-target assembly. Negative weights removed with SHAmanifest. Timing/concurrent total wall in timing.json; no serial equal-work benchmark. No further component-attention sweep; v150smoothCDF explicitly launched PID68944 after both terminal, one GPU. Best remains v140POR77.912103/TOTAL80.602022; strict808/fullOOF82 unmet, no otherfolds.
