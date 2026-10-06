# v116 SW fixed-capacity dilated depth filter
Focus: successful v104 learned65-tap local filter gains may depend on depth span. Compare dilation2/span129 and dilation4/span257 against retained dilation1/span65, with the same845913 total/6432 extra parameters and zero filter initialization.
Controls: same random SW initialization, raw206+relative input, original per-well conditioning/global/local backbone/labels/Q50/fifth split/score, own Gaussian NLL+utility1, SAM.1, DropKey.4, lr3e-4/cos80,80/18/min10/b64/64steps. Only local residual filter dilation/padding changes; no window/grid/voting changes. Zero filters give identical initial predictive function.
Explored/verified: two scoped implementations prepared, no scores. Serial explicit SW arms chosen because prior SW pair~28seconds/epoch each versus13-14solo did not show aggregate benefit. Max2GPU remains an upper limit; no automatic queue.
Gain: not established. Retained SW74.121752828158; current actual fifthTOTAL80.489472312223. Fixed PORv114/PERMv100 plus new random SW totals5 dependency stages including source and both POR phases. No old trained SW loaded.
Not worth continuing: prior spectral/query/hierarchical-utility/teacher changes have no net benefit and are not reopened. This receptive-span study has no result yet.
Pending: explicit dilation2 launch first, then decide second launch from evidence; restore best and actual3-target assembly, record same-capacity delta and real wall/peak. Strict fifth>80.8 plus gain before other folds. FullOOF>82 unmet. Keep prior delivery/positive controls.
Optional after completion: evidence-based retention/refinement only, within5-stage limit.

Live snapshot: dilation2 PID=71100, epoch=5, bestSW=72.175100151643; not final. InitialSW71.197115081472 matches original control initial function.845913 active parameters, measured~12s/epoch, peak=1722.197MiB, loading/warmup and validation counted by trainer. Dilation4 remains prepared/untrained, no queue. No fullCV or other folds started.

V116 dilation2: restored SW=74.036906311599; local dilation1 delta=-0.084846516558; actual fifth TOTAL=80.459776031428;5 stages including source and2 POR phases; strict808=False; no other folds. best/stop=32/50, seconds=617.022, peakMiB=1722.197.

Dilation2 final: restoredSW74.036906311599, against dilation1=-0.084846516558; actual3-target fifthTOTAL80.459776031428, no promotion. best32/stop50,617.022s including load/warmup/train/eval/replay/save, peak1722.197MiB. Checkpoint replay144665 exact and frozen parameters exact. Negative best/last removed with manifest (18.202MiB), compact result/source retained. Dilation4 explicitly launched after dilation2 terminal, PID28492; actual serial execution, not concurrent, no automatic queue. Current best5-stageTOTAL80.489472312223 unchanged; no other folds. Pending dilation4 final and full ablation decision.

V116 dilation4: restored SW=74.039683407705; local dilation1 delta=-0.082069420453; actual fifth TOTAL=80.460748015065;5 stages including source and2 POR phases; strict808=False; no other folds. best/stop=32/50, seconds=685.994, peakMiB=1722.197.

Final v116: dilation4SW=74.039683407705, original dilation1 delta=-0.082069420453, actual fifthTOTAL=80.460748015065, best32/stop50. Dilation4-minus-dilation2=+0.002777096106, a tiny secondary contrast; both worse than predeclared retained original dilation1. Close this fixed-capacity dilation mechanism, no more dilation/kernel-width sweeps. Both144665-point best replays/frozen checks passed. Current5-stageTOTAL80.489472312223 unchanged, strict808false; no other folds. Actual SW pair serial, measured pair wall=1368.167s including gap, singles617.022/685.994s, peak1722.197MiB each; second overlaps small PERM task, no standalone speedup claim. Both unpromising best/last outputs cleaned with separate manifests, prior local positive/source/current/delivery retained. FullOOF>82 remains unmet. Next direction must change mechanism rather than prolong these failed density/dilation scopes.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
