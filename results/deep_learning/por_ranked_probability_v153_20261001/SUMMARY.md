# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v153 扩大已证实联合CDF监督

探索重点：沿已证实v151联合RPS正收益，预定义仅提高系数1->2，检验增益能否扩大。不改变有效256位置/权重读出结构或230944参数，不堆叠负模块。

已经探索与验证：实现完成，尚未启动、无成绩。同父v113/初始化/种子/窗口/预算/AdamW/lr，仅目标为NLL+2*jointRPS。原有限255边界、默认atom标准化值与连续真实标签共同CDF均方误差、Q50与/3归一化不变；不是无限支持CRPS，不用动作softargmax或解码调参。未加载v151/v140训练头，替换POR头阶段而非增加第六阶段。

收益：尚未证实，完成恢复后比较v151 POR77.926380760724，以及原NLL v14077.912102888582。v151已完整第五折+0.014277872142，默认+0.008478438272、非默认+0.025155211367，10/16井改善；单折单种子小收益，不夸大泛化。实际固定PERM136/SW104合并后才报告TOTAL。

不值得继续：v148自由区间、v150平滑CDF、v149分量通信、192/384/512/尺度修正均无对照增益，不叠加。v152相加低于基础，乘法仍在训练、结论未完成。此处只有一个预定义系数2，不自动扩成系数网格。

约束：原9曲线/206+63/64训练井统计/Q50/原第五折/seed20261010/Student3+Cauchy/门控尺度骨干冻结/原动作与区间不变；normal/max2/80-18/min10/b64/64步/3e-4/cos80*64；1024行loss梯度checkpoint，完整原子last与显式恢复、进程级防睡眠。source+POR113+POR153+PERM136+SW104共5阶段；第五严格TOTAL>80.8且收益后补折，完整OOF>82未完成。旧交付/当前正控制不改。

计划：利用乘法候选之外的空位显式启动；记录真实过程/单项与并发墙钟/预热/峰值，最佳磁盘恢复、144665覆盖/冻结完整性和三目标合并；负权重按SHA清理，正控制保留，更新PROJECTS/EXPLORATIONS/图谱。

Verified explicit launch PID43724, actual epoch4 and commandline live; initial POR77.540454326739/230944parameters, same asRPS1 control. Current partialbest77.540454326739, peak924.547MiB, noOOM/stderr. Only coefficient2 changed; original positivev151 checkpoint andrecipe untouched. Two GPUjobs with productv152; final gain/actualassembly pending, no otherfolds.

Verified live PID43724 snapshot epoch24, partialPORbest77.817462477236; no finalgain vs R

# v153 扩大已证实联合CDF监督

探索重点：沿已证实v151联合RPS正收益，预定义仅提高系数1->2，检验增益能否扩大。不改变有效256位置/权重读出结构或230944参数，不堆叠负模块。

已经探索与验证：实现完成，尚未启动、无成绩。同父v113/初始化/种子/窗口/预算/AdamW/lr，仅目标为NLL+2*jointRPS。原有限255边界、默认atom标准化值与连续真实标签共同CDF均方误差、Q50与/3归一化不变；不是无限支持CRPS，不用动作softargmax或解码调参。未加载v151/v140训练头，替换POR头阶段而非增加第六阶段。

收益：尚未证实，完成恢复后比较v151 POR77.926380760724，以及原NLL v14077.912102888582。v151已完整第五折+0.014277872142，默认+0.008478438272、非默认+0.025155211367，10/16井改善；单折单种子小收益，不夸大泛化。实际固定PERM136/SW104合并后才报告TOTAL。

不值得继续：v148自由区间、v150平滑CDF、v149分量通信、192/384/512/尺度修正均无对照增益，不叠加。v152相加低于基础，乘法仍在训练、结论未完成。此处只有一个预定义系数2，不自动扩成系数网格。

约束：原9曲线/206+63/64训练井统计/Q50/原第五折/seed20261010/Student3+Cauchy/门控尺度骨干冻结/原动作与区间不变；normal/max2/80-18/min10/b64/64步/3e-4/cos80*64；1024行loss梯度checkpoint，完整原子last与显式恢复、进程级防睡眠。source+POR113+POR153+PERM136+SW104共5阶段；第五严格TOTAL>80.8且收益后补折，完整OOF>82未完成。旧交付/当前正控制不改。

计划：利用乘法候选之外的空位显式启动；记录真实过程/单项与并发墙钟/预热/峰值，最佳磁盘恢复、144665覆盖/冻结完整性和三目标合并；负权重按SHA清理，正控制保留，更新PROJECTS/EXPLORATIONS/图谱。

Verified explicit launch PID43724, actual epoch4 and commandline live; initial POR77.540454326739/230944parameters, same asRPS1 control. Current partialbest77.540454326739, peak924.547MiB, noOOM/stderr. Only coefficient2 changed; original positivev151 checkpoint andrecipe untouched. Two GPUjobs with productv152; final gain/actualassembly pending, no otherfolds.

Verified live PID43724 snapshot epoch24, partialPORbest77.817462477236; no finalgain vs RPS1 control77.926381. V152paired negative finalized/cleaned, now oneGPUjob only. No more candidates launched this batch; current positive5phase recipe80.606305 retained, no otherfolds.

Verified live snapshot epoch42, bestPOR77.901655016833; below RPS1control77.926381, finalconclusion pending. Now overlaps PERM154; no more than2GPUjobs.

V153 joint_rps: restored POR=77.901655016833; delta vs v114=+0.353585521940, vs historical=+0.349718214948; actual fifth TOTAL=80.598887628429;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=30/48, seconds=536.930, peakMiB=924.547.

V153 final negative: coefficient2POR77.901655016833/TOTAL80.598887628429; deltaRPS1 -0.024725743891/NLL -0.010447871749. Best30/stop48, exact144665/frozen replay/actual assembly; training536.930s/peak924.547MiB, wall/overlap timing.json. Negativeweights deleted with SHAmanifest; keep positiveRPS1/newrecipe80.606305 and oldcontrols. Close coefficient2, no new coefficient grid. SW155explicitlaunched afterterminal alongsidePERM154(max2); no otherfolds/fullOOF82unmet.


