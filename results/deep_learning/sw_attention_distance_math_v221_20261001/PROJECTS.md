Goal: structural gain from small attention mechanism. Done: source/trainer/record/plan. Next: explicit training and assembled score. Not done: gain/fullOOF>82. Optional after goal: new delivery.

221 uses explicit float32 QK/softmax/V in the modified last block after220 failed CUDA LSE backward alignment before epoch1. Original weights and zero bias kept; initial score separately measured for rounding.220 immutable failure, not negative evidence. No auto-resume.

V221 final SW74.122157243259, gain+0.000404415101, actualTOTAL80.612616963279; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop3/13, trainingseconds225.275674/peakMiB1211.433594. No otherfolds.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
