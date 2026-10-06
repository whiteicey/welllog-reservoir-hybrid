# v155 进展

目标：检验联合CDF监督能否改善SW有效模型，最终完整OOF>82。

已完成：同SW104容量/初始化/架构/utility/SAM，仅RPS1增补实现与固定POR151/PERM136真实合并入口。

计划/未完成：显式启动后记录真实训练、时间显存；恢复最佳/原第五144665覆盖/实际合并、增益结论、负权重清理和图谱。

可选：有收益保留目标独立设置；无收益不机械提高系数或引入负模块，不自动其他折。

Explicitlaunch PID53596 verified actual epoch2 and commandline. InitialSW71.197115081472/845913parameters same randomlocal104 start, partialbest71.197115081472, peak1728.449MiB, noOOM. First epoch32.045s with PERM154 overlap, not standalone equal-work speed evidence. Existing CUDAwarn_only attention/interpolation warnings observed; no bitwise training reproducibility claim. Initiallearning is not gain vs SW10474.121753; final replay/assembly pending. Onlytwo jobs, no otherfolds.

SWverifiedlive PID53596 snapshot epoch13, partialbest73.396265624043; initiallearning not gainvsSW10474.121753. PERM154positive finished, now oneGPUprocess; original fixedPERM136 assembly plan unchanged, no source/config mutation. Current combined positivePOR151/PERM154/SW10480.612475 recipe recorded separately.

V155 restored SW=73.670351522101, delta v104=-0.451401306057; actual TOTAL=80.448314894477, five phases/source included, strict808=False; no otherfolds. best/stop=20/21; exact144665/frozen replay.

V155 priority-stop final (INCOMPLETE): saved completeepoch21/best20 restoredSW73.670351522101, below retainedSW104by0.451401306057. User latest focusesPORaugmentation/MAINPERM, not full-budget convergence negative. Exact144665/frozen replay/actual original-planned assembly verified. Best/last cleanedwith SHAmanifest, no auto-resume. Timing records stop/replay/overlap; frozen copiedv104 trainingentry lacked TrainingAwake call, finalizer used it; no retrospective source mutation. POR157 explicitnewlaunch and PERM156 now2jobs; latest positive5phase recipe80.612475 untouched.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
