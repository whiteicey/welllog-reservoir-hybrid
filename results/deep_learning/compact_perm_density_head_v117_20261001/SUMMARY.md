# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v117 compact PERM nonlinear joint-density readout
Focus/hypothesis: compact PERM previously adjusts only default gates and expert routes; fixed source density may constrain its gain. Add a bounded nonlinear191-32-24 residual per expert jointly adjusting mixture log-weights, locations and raw widths while keeping source density rows/backbone frozen.
Controls: original source initialization only, no v100 trained PERM loaded. Existing gate/side/routes training remains; new output zero initialized so initial full function equals v100 Gaussian control. New capacity means an architecture bundle, not equal-capacity ablation. All own PERM Gaussian likelihood/CDF and original nine-curve/206+63/input/data/Q50/fifth/scoring unchanged. Bounds fixed .25 logit/raw-width and .25 detached source component width for location.
Explored/verified: one predefined implementation prepared, no score yet.80/18/min10/bs64/64steps, existing LRgroups3e-5/3e-4, atomic complete state; max2GPU. Intended overlap with one active SW task, not two SW tasks; record observed wall/contention without speedup claims.
Gain: not established; historical PERM89.378394211144, current actual fifthTOTAL80.489472312223. Frozen PORv114 and SWv104 mean5 total training stages including source and2 POR phases; added joint head trained together with existing PERM components, no extra phase.
Not worth continuing: failed warm lin

viously adjusts only default gates and expert routes; fixed source density may constrain its gain. Add a bounded nonlinear191-32-24 residual per expert jointly adjusting mixture log-weights, locations and raw widths while keeping source density rows/backbone frozen.
Controls: original source initialization only, no v100 trained PERM loaded. Existing gate/side/routes training remains; new output zero initialized so initial full function equals v100 Gaussian control. New capacity means an architecture bundle, not equal-capacity ablation. All own PERM Gaussian likelihood/CDF and original nine-curve/206+63/input/data/Q50/fifth/scoring unchanged. Bounds fixed .25 logit/raw-width and .25 detached source component width for location.
Explored/verified: one predefined implementation prepared, no score yet.80/18/min10/bs64/64steps, existing LRgroups3e-5/3e-4, atomic complete state; max2GPU. Intended overlap with one active SW task, not two SW tasks; record observed wall/contention without speedup claims.
Gain: not established; historical PERM89.378394211144, current actual fifthTOTAL80.489472312223. Frozen PORv114 and SWv104 mean5 total training stages including source and2 POR phases; added joint head trained together with existing PERM components, no extra phase.
Not worth continuing: failed warm linear-mean and fixed heavy-tail PERM directions closed; this new nonlinear joint density is not yet tested.
Pending: direct train, restore best/144665 prediction integrity, actual3-target assembly, delta/peak/time; strict fifth>80.8 plus gain before other folds. FullOOF>82 remains unmet. Old delivery preserved.
Optional: follow positive structural evidence within stage budget; no automatic queue.

V117 joint_density: restored PERM=89.347354882560; Gaussian delta=-0.031039328584; actual fifth TOTAL=80.478608547219;5 stages including source and2 POR phases; strict808=False; no other folds. best/stop=2/20, seconds=185.930, peakMiB=786.465.

Final v117: restoredPERM=89.347354882560, against Gaussian gate-only=-0.031039328584; actual3-target fifthTOTAL=80.478608547219. best/stop=2/20, seconds=185.930, peak=786.465MiB;144665 exact replay/frozen params checks passed. Close this bounded joint-density architecture scope, not all density learning. Prior compact5-stageTOTAL80.489472312223 retained. Both old linear-mean and this nonlinear joint-density failed; do not continue small PERM density-row sweeps. Source/current/positive controls retained; negative best/last removed with manifest; SW dilation4 stays active, no training stopped for cleanup. No other folds/fullgoal claim.


