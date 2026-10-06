# v122 training utility matrix timing probe
Focus: large8192x257 by257x801 FP32 utility matrix multiplies may limit training speed. Compare IEEE versus TF32 for only this synthetic training-objective graph, without changing any active training source or final prediction/argmax precision.
Explored/verified: prepared short fixed-shape CUDA timing, actual utility matrix from retained training-side SW grid; synthetic probabilities/rewards, no labels/model updates. One benchmark plus one activeSW at most2GPU processes.8 warmup and32 measured forward/backward repetitions per precision, both own GPU stream synchronization/event timing and observed contention. No claim of real epoch/fullCV speedup from this isolated operation.
Gain: no timing result yet, no trained model or score; this is a runtime hypothesis, not accuracy improvement.
Not worth continuing: if isolated timing benefit small, no full-training precision experiment. If clear, a separately documented training-only approximation can be tested while validation/inference keeps original IEEE settings.
Pending: record elapsed/peak/precision/hardware/scope; updatePROJECTS/atlas. No training stage added by benchmark, no checkpoint, original best/fold splits/scoring/delivery remain unchanged. Original fullOOF>82 not achieved.

Final timing-only result: IEEE=2.651776ms versus TF32=2.308480ms per8192-point utility forward/backward, isolated ratio=1.148711;8 warmup/32 measured each while splitSW active. Whole benchmark runtime including source-grid load/warmup=0.397472s (Python startup outside this), peak210.849MiB. Estimated utility-only saving at2 evaluations*64steps is only~0.044seconds per epoch, compared with~13seconds actual epoch; this is an operation-count estimate, not measured epoch speedup. Do NOT launch full training precision experiment; keep original IEEE precision/settings in future trainers. No model/score or checkpoint produced, no claim of accuracy gain. Timing bottleneck hypothesis not worth further scope here; original goal remains unmet.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
