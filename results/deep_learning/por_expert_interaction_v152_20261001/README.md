# v152 在有效分布读出上增补专家交互

探索重点：保留有效256位置/权重读出，在其隐层增补低秩专家交互。冻结两专家128维特征分别做无参数LayerNorm、投影为32维，相乘或相加，经同32维LayerNorm和零初始化32->256支路融入原256隐层。原206+63幅值与319维基础读出路径保留。

已经探索与验证：实现完成，尚未启动，无成绩。additive/product两候选结构、参数、初始化、数据、预算、优化器均相同，只差相加/相乘算子。恢复原ContextHead全部初始参数，再零初始化新增支路出口；初始新增功能为零，原预测函数保持。首步新增支路和原零出口造成的暂时梯度延迟必须如实理解，后续观察实际梯度及成绩。新增16768参数，总247712；相对v140容量变化，只有两候选间是匹配容量机制消融。

收益：尚未证实。机制比较product-additive，绝对比较有效v140POR77.912102888582。原预训练来自自身64训练井，v113冻结父模型，不加载v140/v149/v150/v151训练头；不是第六阶段串联。只自身真实POR似然训练，门控/尺度/族/骨干固定；未将负CDF/分量attention/RPS机制叠加进来。

不值得继续：宽度192/384/512负，尺度/自由区间/平滑CDF修正负，分量通信匹配消融负；保持已有效读出再增补输入条件交互，不泛化这些单次结果为所有机制失败。v151独立联合CDF监督仍在训练，不把未完成结果当负。

约束：原9曲线/206+63/训练64井统计/Q50/第五折/seed20261010/混合Student3+Cauchy/原动作和区间不变，80/18/min10/b64/64步/lr3e-4/cos80*64/normal/max2；完整原子last、显式恢复和临时防睡眠。source+POR113+POR152+PERM136+SW104共5阶段，实际三通道预测合并后判第五TOTAL>80.8与收益，再考虑其余折。完整OOF>82未完成，交付不覆盖。

计划：显式分批启动两项，v151终止前不得同时新增两GPU任务；记录真实初步/最终恢复、144665覆盖、冻结完整性、单项与同等工作量总墙钟、显存/预热；完成负权重留SHA清单清理，正控制保留。

Verified additive launch PID61136: commandline and actual epoch5 confirmed live. Initial POR77.540454326739,247712parameters; peak955.997MiB, noOOM/stderr. Product still prepared/untrained, no autoqueue. V151 PID29608 remains live alongside; max2 maintained. Partial results not gain.

Verified additive live snapshot epoch20, partialPORbest77.699544379378; product prepared/no score. Two totalGPUjobs (v151/additive), noOOM. Paired mechanism comparison incomplete.

V152 additive: restored POR=77.857372986694; delta vs v114=+0.309303491801, vs historical=+0.305436184809; actual fifth TOTAL=80.585603019388;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=30/48, seconds=469.133, peakMiB=955.997.

Product explicitly launched PID62976 after v151 terminal, initial POR77.540454326739/247712parameters; live actual epoch11, best77.540454326739, noOOM/stderr. Additive naturallycompleted48/best30, restored POR77.857372986694/TOTAL80.585603019388, delta vs direct256 -0.054729901887. Exact144665/frozen replay. Keep additive control weights until paired product comparison finalized; no restart. OneGPUjob now live; product final/paired gain pending. V151positive recipe80.606305 retained, no otherfolds.

Product verifiedlive epoch31, partialbestPOR77.848494843975; completed additive77.857373 remains matchingcontrol, finalpair gain pending. v153coeff2now overlaps; max2 maintained.

V152 product: restored POR=77.848494843975; delta vs v114=+0.300425349082, vs historical=+0.296558042090; actual fifth TOTAL=80.582939576572;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=30/48, seconds=412.136, peakMiB=955.997.

V152 final paired negative: additivePOR77.857372986694/productPOR77.848494843975; product minus additive-0.008878142719. Both below effective256/RPSpositive; exact144665/frozen replay and actual assemblies complete. Bothbest30/stop48. Four negativeweights deleted with SHAmanifest after pairfinal; all compact evidence/predictions kept. Timing and combined staggered wall in timing.json. Close rank32 add/product configuration, no negative stack into positive recipe; v153coeff2 continues alone. OriginalfifthTOTALbest80.606305, strict808/fullOOF82 unmet/nootherfolds.
