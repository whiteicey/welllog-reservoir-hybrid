# v151 进展

目标：在已有效256分布读出上增补联合分布监督，验证第五折收益并最终完整OOF>82。

已完成：同容量原结构，默认/连续联合有限边界RPS实现，PLAN和对照固定。

计划/未完成：显式启动一个GPU任务，与v150总数不超过2；最佳恢复及实际合并、收益/时间/显存证据、清理与图谱更新。

可选后续：若有正消融收益，可保留联合监督再优化；无收益则不继续同类扫参或叠加负模块，不自动其他折。

Verified launch PID29608, actual training observed, snapshot epoch8, partialbest77.540454326739, peakMiB924.547. Initial POR77.540454326739,230944parameters, exact effective256architecture/initialization retained; only addsjointRPS. No newgain established. V150 terminal now, onlyone training process remains.

V151 live snapshot epoch25, partialbestPOR77.820742086193, bestepoch17; not final gain vs25677.912103. Additivev152 now overlaps, no more than2jobs. No otherfolds or currentrecipe change.

Verified live snapshot epoch35, bestPOR77.926380760724 at30, partial delta vs samecapacity v140+0.014277872142. This is a training snapshot, not final replay/assembly or fullOOF gain. Preserve all active states; final restoration pending. Productv152 still not launched while twoGPUjobs occupied.

V151 joint_rps: restored POR=77.926380760724; delta vs v114=+0.378311265831, vs historical=+0.374443958839; actual fifth TOTAL=80.606305351597;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=30/48, seconds=624.994, peakMiB=924.547.

V151 final positive adopted in NEW CURRENT_RECIPE.json: restored POR77.926380760724, delta samecapacity v140+0.014277872142; actual TOTAL80.606305351597 (+0.004283361643). Default delta+0.008478438272, numeric+0.025155211367, positive10/16wells; descriptive frozen analysis only, no selection byrow/well. Best30/stop48, exact144665/frozen replay; training624.994seconds, peak924.547MiB, overlap/wall in timing.json. Preserve positivebest/last and previouscontrol/delivery. Five phases, strict808 false/nootherfolds, completeOOF82notachieved. Next bounded RPScoefficient expansion may be prepared only fromsamefrozenparent replacingheadphase, no negative module stacking.
