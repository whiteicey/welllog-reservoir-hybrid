Goal: reduce optimization overshoot in small positive scale readout, preserve learned154.
Done: prepared immutable separate module/trainer/plan, same initial function.
Next: explicit scale launch; replay/assembly, compare154 and198, update atlas.
Unfinished: training and evidence; full OOF>82 not achieved. No remainingfolds.
Optional after goal: robustness/seeds only if justified.

V201 direct skip-middle completed best1/stop15 POR77.415345964852; lower than11377.540454 and15177.926381. All prefix recipes verified, keep three POR phases. V202 lower LR3e-5 scale-only4656patch explicitly launched PID53300; same154 parent, no198 weights; one optional PERM stage replaces198, not stacks. No GPU job stopped; peak concurrent two, one now live. 197/199/201 negative checkpoints removed with SHA manifests after full144665 replay/assembly. Goal fullOOF>82 still unmet; no remainingfolds.


V202 scale final PERM=89.406833212247, matched198gain=+0.001002030895; actualTOTAL=80.612919342359, sixphases inclsource;strict808=False, no otherfolds. best/stop=2/16, seconds=168.443, peakMiB=786.465; all144665replay/frozen exact.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
