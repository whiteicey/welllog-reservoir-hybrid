# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW mass correction: increased finite bound
Focus: test whether bounded probability reweighting caps the emerging209 gain. Same12576parameter architecture/initial tensors/function (both zerooutput and derivative1), only logmass bound0.5 to2. Both warm original104, no209 weights.
Explored: prepared/untrained/noqueue. Original256bins/800actions/scoring/own binNLL+BCE+utility1/temp0.05/SAM0.1/LR1e-4/fold5/Q50/80-14 unchanged. Gate/source/oldGaussian parameters frozen; floor1e-12 float initial delta recorded separately.
Gain: unproved;209 stilltraining, its restored finalSW is matchedcontrol. Primary baseline10474.12175282815758. One additionalSWstage only, sixoverall, alternatives209/211/212 neverstack. Defaultfive until meaningful added benefit, no remainingfolds/fullOOF claim.
Not worth continuing: no bound grid predefined; this single expansion follows209 positive trajectory. Increasing bound does not create flexible support from nothing; Gaussian underflow/floor limitations must remain explicit. If no matched benefit, keep original bound.
Timing: fullsingle run including loading/eval and real batchwall/overlap/peak, no isolatedpatch equalswhole recipe claim. Atomicstates/frozen/replay/actualassembly retained.

V212 final SW74.144913000682, gain+0.023160172525, actualTOTAL80.620581478378; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best

# SW mass correction: increased finite bound
Focus: test whether bounded probability reweighting caps the emerging209 gain. Same12576parameter architecture/initial tensors/function (both zerooutput and derivative1), only logmass bound0.5 to2. Both warm original104, no209 weights.
Explored: prepared/untrained/noqueue. Original256bins/800actions/scoring/own binNLL+BCE+utility1/temp0.05/SAM0.1/LR1e-4/fold5/Q50/80-14 unchanged. Gate/source/oldGaussian parameters frozen; floor1e-12 float initial delta recorded separately.
Gain: unproved;209 stilltraining, its restored finalSW is matchedcontrol. Primary baseline10474.12175282815758. One additionalSWstage only, sixoverall, alternatives209/211/212 neverstack. Defaultfive until meaningful added benefit, no remainingfolds/fullOOF claim.
Not worth continuing: no bound grid predefined; this single expansion follows209 positive trajectory. Increasing bound does not create flexible support from nothing; Gaussian underflow/floor limitations must remain explicit. If no matched benefit, keep original bound.
Timing: fullsingle run including loading/eval and real batchwall/overlap/peak, no isolatedpatch equalswhole recipe claim. Atomicstates/frozen/replay/actualassembly retained.

V212 final SW74.144913000682, gain+0.023160172525, actualTOTAL80.620581478378; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop9/23, trainingseconds542.029938/peakMiB908.493652. No otherfolds.


