# v151 在有效读出上增补完整分布监督

探索重点：保留已有效的v140 256维位置/权重读出，不增容量，只增补joint ranked probability score（RPS）训练。原非默认条件似然不监督默认样本的条件分布；新增项让默认和连续真实标签共同监督整体CDF，检验是否降低默认得分损失。

已经探索与验证：实现完成，尚未训练、无成绩。原255个有限密度边界上计算联合CDF：g*I(默认标准化值<=边界)+(1-g)*F_mix(边界)，与真实标签阶跃CDF比较均方误差，边界均匀平均，Q50加权；NLL+1*RPS，均保持/3归一化。真实默认标签标准化log1p(0.1)，非默认标签使用原训练标签。系数1预先固定，不搜索边界、阈值、动作或分辨率。不称为无限支持CRPS：Cauchy混合无有限一阶矩，此处是原有限离散边界上的RPS。

对照与收益：尚未证实。结构和230944训练参数与有效v140完全相同，同初始化、父v113/种子/预算/lr；只新增训练目标，预定义对照v140 POR77.912102888582。不加载已训练v140头追加第六阶段；沿用它的有效结构，同阶段重训，区别需诚实记录。不能把初始原函数成绩称新增收益，也不能用真值填充其他目标报告TOTAL。

不值得继续：v149分量通信相对匹配独立组-0.027491，两个组均低于有效256读出；v148自由区间修正-0.023091；尺度修正/192/384/512负结果不叠加。历史v86/v92动作效用不是本新增的proper jointCDF监督，不重复其全骨干/旧18阶段/softargmax方案。

约束：原9曲线/206+63幅值、train-only64井统计/Q50/原第五折/seed20261010、原Student3+Cauchy概率族/冻结默认门控、尺度、骨干/原解码保持。1024行分块梯度checkpoint控制显存，normal最多2GPU、80/18/min10/b64/64步/3e-4/cos80*64。原子完整last，显式恢复、进程级防睡眠。source+POR113+POR151+PERM136+SW104=5阶段；第五严格TOTAL>80.8并有收益后补折，完整OOF>82未完成。固定PERM/SW实际预测合并，交付不改。

待完成：显式启动/观察真实训练；实际最佳重载、144665覆盖/冻结完整性及三通道合并；记录单项与并发墙钟/峰值/预热，结束或转向后更新结论，负权重清理留SHA证据。

Verified launch PID29608, actual training observed, snapshot epoch8, partialbest77.540454326739, peakMiB924.547. Initial POR77.540454326739,230944parameters, exact effective256architecture/initialization retained; only addsjointRPS. No newgain established. V150 terminal now, onlyone training process remains.

V151 live snapshot epoch25, partialbestPOR77.820742086193, bestepoch17; not final gain vs25677.912103. Additivev152 now overlaps, no more than2jobs. No otherfolds or currentrecipe change.

Verified live snapshot epoch35, bestPOR77.926380760724 at30, partial delta vs samecapacity v140+0.014277872142. This is a training snapshot, not final replay/assembly or fullOOF gain. Preserve all active states; final restoration pending. Productv152 still not launched while twoGPUjobs occupied.

V151 joint_rps: restored POR=77.926380760724; delta vs v114=+0.378311265831, vs historical=+0.374443958839; actual fifth TOTAL=80.606305351597;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=30/48, seconds=624.994, peakMiB=924.547.

V151 final positive adopted in NEW CURRENT_RECIPE.json: restored POR77.926380760724, delta samecapacity v140+0.014277872142; actual TOTAL80.606305351597 (+0.004283361643). Default delta+0.008478438272, numeric+0.025155211367, positive10/16wells; descriptive frozen analysis only, no selection byrow/well. Best30/stop48, exact144665/frozen replay; training624.994seconds, peak924.547MiB, overlap/wall in timing.json. Preserve positivebest/last and previouscontrol/delivery. Five phases, strict808 false/nootherfolds, completeOOF82notachieved. Next bounded RPScoefficient expansion may be prepared only fromsamefrozenparent replacingheadphase, no negative module stacking.
