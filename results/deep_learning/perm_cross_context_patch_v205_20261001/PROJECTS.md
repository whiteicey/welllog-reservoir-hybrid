Goal: preserve gains and test whether nonlocal input context helps distribution readout.
Done: newmodule/trainer/plan prepared, reference checked, no training yet.
Next: explicit launch when one verified slot frees; no automaticqueue. Actual restored score/assembly, cleanup negative if necessary.
Unfinished: efficacy/fullOOF>82; no remainingfolds.
Optional after goal: connectivity ablation if positive.


V202 restored PERM89.406833212247, +0.001268355328 vs154 and +0.001002030895 vs198; actualTOTAL80.612919342359, optional6phase recipe retained, not default or fullOOF. V205 context patch verified initial89.405564856919/6992parameters; admitted alongside203, at most2GPU. PyTorch warns adaptive pooling backward nondeterministic under existing warn_only setting: inference/replay checks remain required, no exact training-trajectory reproducibility claim. V204 construction failed no epochs/score, source retained. No remainingfolds.


V205 attention final PERM=89.408616231012, matched198gain=+0.002785049659; actualTOTAL=80.613543398926, sixphases inclsource;strict808=False, no otherfolds. best/stop=1/15, seconds=151.016, peakMiB=786.465; all144665replay/frozen exact.

V205 actual assembly completed TOTAL80.613543398926/PERM89.408616231012, +0.003051374093 vs154. Retain positive best/last and optional6phase recipe; default5 because gain tiny, strict808false. V206 local feature adapter running PID81288; V207 same205 architecture lowerLR3e-5 running PID84032, exactly twoGPU admitted. No source changes to old trained files; oneextra perhead only, candidates not stacked. Goal fullOOF>82 unmet.



## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
