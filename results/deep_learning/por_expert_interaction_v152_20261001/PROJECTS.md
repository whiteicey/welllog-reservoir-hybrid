# v152 进展

目标：增补已有效读出，检验专家低秩交互收益，最终完整OOF>82。

已完成：additive/product同初始化、容量对照与零支路实现；PLAN和既有256绝对对照固定。

未完成/计划：显式分批训练、恢复最佳与实际三通道合并；记录机制/绝对增益、时间显存与负证据；更新图谱和清理。

可选：有收益则保留交互，独立迁移其他头；无收益不继续同类复杂模块堆叠，不自动补折。

Verified additive launch PID61136: commandline and actual epoch5 confirmed live. Initial POR77.540454326739,247712parameters; peak955.997MiB, noOOM/stderr. Product still prepared/untrained, no autoqueue. V151 PID29608 remains live alongside; max2 maintained. Partial results not gain.

Verified additive live snapshot epoch20, partialPORbest77.699544379378; product prepared/no score. Two totalGPUjobs (v151/additive), noOOM. Paired mechanism comparison incomplete.

V152 additive: restored POR=77.857372986694; delta vs v114=+0.309303491801, vs historical=+0.305436184809; actual fifth TOTAL=80.585603019388;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=30/48, seconds=469.133, peakMiB=955.997.

Product explicitly launched PID62976 after v151 terminal, initial POR77.540454326739/247712parameters; live actual epoch11, best77.540454326739, noOOM/stderr. Additive naturallycompleted48/best30, restored POR77.857372986694/TOTAL80.585603019388, delta vs direct256 -0.054729901887. Exact144665/frozen replay. Keep additive control weights until paired product comparison finalized; no restart. OneGPUjob now live; product final/paired gain pending. V151positive recipe80.606305 retained, no otherfolds.

Product verifiedlive epoch31, partialbestPOR77.848494843975; completed additive77.857373 remains matchingcontrol, finalpair gain pending. v153coeff2now overlaps; max2 maintained.

V152 product: restored POR=77.848494843975; delta vs v114=+0.300425349082, vs historical=+0.296558042090; actual fifth TOTAL=80.582939576572;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=30/48, seconds=412.136, peakMiB=955.997.

V152 final paired negative: additivePOR77.857372986694/productPOR77.848494843975; product minus additive-0.008878142719. Both below effective256/RPSpositive; exact144665/frozen replay and actual assemblies complete. Bothbest30/stop48. Four negativeweights deleted with SHAmanifest after pairfinal; all compact evidence/predictions kept. Timing and combined staggered wall in timing.json. Close rank32 add/product configuration, no negative stack into positive recipe; v153coeff2 continues alone. OriginalfifthTOTALbest80.606305, strict808/fullOOF82 unmet/nootherfolds.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
