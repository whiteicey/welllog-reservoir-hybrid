# v153 进展

目标：扩大已有联合CDF监督收益，最终完整OOF>82。

已完成：同结构/容量/初始化，只有RPS系数1->2改变；PLAN与现有正控制固定。

计划/未完成：显式单项启动、真实训练观察；完成最佳恢复/实际合并/对照增益/耗时显存记录；更新图谱与必要负权重清理。

可选：正收益保留监督强度，再独立迁移其他目标。无收益不自动开系数网格或补折。

Verified explicit launch PID43724, actual epoch4 and commandline live; initial POR77.540454326739/230944parameters, same asRPS1 control. Current partialbest77.540454326739, peak924.547MiB, noOOM/stderr. Only coefficient2 changed; original positivev151 checkpoint andrecipe untouched. Two GPUjobs with productv152; final gain/actualassembly pending, no otherfolds.

Verified live PID43724 snapshot epoch24, partialPORbest77.817462477236; no finalgain vs RPS1 control77.926381. V152paired negative finalized/cleaned, now oneGPUjob only. No more candidates launched this batch; current positive5phase recipe80.606305 retained, no otherfolds.

Verified live snapshot epoch42, bestPOR77.901655016833; below RPS1control77.926381, finalconclusion pending. Now overlaps PERM154; no more than2GPUjobs.

V153 joint_rps: restored POR=77.901655016833; delta vs v114=+0.353585521940, vs historical=+0.349718214948; actual fifth TOTAL=80.598887628429;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=30/48, seconds=536.930, peakMiB=924.547.

V153 final negative: coefficient2POR77.901655016833/TOTAL80.598887628429; deltaRPS1 -0.024725743891/NLL -0.010447871749. Best30/stop48, exact144665/frozen replay/actual assembly; training536.930s/peak924.547MiB, wall/overlap timing.json. Negativeweights deleted with SHAmanifest; keep positiveRPS1/newrecipe80.606305 and oldcontrols. Close coefficient2, no new coefficient grid. SW155explicitlaunched afterterminal alongsidePERM154(max2); no otherfolds/fullOOF82unmet.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
