# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW first QKV fine-tune

Focus: preserve SW104 body/readout and fine-tune one original QKV,27936 existing parameters,zero added. No214/mass weights loaded.

Explored: prepared,untrained; LR3e-5,continuous NLL/BCE+utility1,SAM0.1,first block training DropKey0.4 only; original fold5/Q50/80-14.

Gain: unproved,reference SW74.12175282815758. Scope,LR,training DropKey change together; no pure scope attribution.

Not worth continuing:214 rank8 only+0.002429,no rank/layer grid;213/216 views below repeated-hidden control,no expansion.

Timing includes loading/eval/warmup, record overlap and peak memory. Restored-state replay is not disk reload. Six phases/one SW addition; default five-phase unchanged.

V218 final SW74.121752828158, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds283.698458/peakMiB1255.786621. No otherfolds.

218 final SW74.121752828158, best0/14,no gain; actual assembly recorded.219 active;220 small distance-bias alternative prepared. Max2 GPU,no remainingfolds.217 gate variant below211, no continuation.




