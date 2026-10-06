# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW coupled state / probability readout
Focus: extend positive likelihood-only mass correction with a single extra defaultgate output. One128->32->257 patch, 12609newparameters; first256 boundedlogmass0.5, last1 boundedgate-logit0.25. Allold104 weights frozen and samezero initialfunction exceptexisting floor/floatnormalization. No211 weights orsecondtrainingstage.
Explored: prepared/untrained. OwnSW categoricalNLL+BCE/utility0/SAM0.1/LR1e-4/80-14, originalfold5/Q50/9curves/bins/actions/score unchanged. Gate anddistribution can coadapt throughsharednew32width nonlinearity; conditionaloldmeans/scales/weights retained.
Gain: unproved; reference211SW74.14762262262707 and10474.12175282815758. Not capacitymatched pure-gate claim (extra33parameters). OptionaloneSWstage/sixoverall, allmassvariantsalternativesnotstacked, default5; no remainingfolds/fullOOFclaim.
Not worth continuing:213posteriorview belowrepeathidden by0.003417;216Sourcehidden belowsamefunctionalcontrol by0.001143. Both viewsclosed; do notcompound them.217 uses only original128SW features.
Timing: record fullrun/batchwall inclloading/eval/warmup/overlap/memory. Atomicstates/frozen/144665 restore/replay/actualassembly. Validatorclasserrordiagnostics ifused neverfittrainingweightsorper-row selection.

CPU diagnostic104->211: default100786 rows SW98.715632->98.637659 (-0.077974); continuous43879 rows17.631889->17.896277 (

# SW coupled state / probability readout
Focus: extend positive likelihood-only mass correction with a single extra defaultgate output. One128->32->257 patch, 12609newparameters; first256 boundedlogmass0.5, last1 boundedgate-logit0.25. Allold104 weights frozen and samezero initialfunction exceptexisting floor/floatnormalization. No211 weights orsecondtrainingstage.
Explored: prepared/untrained. OwnSW categoricalNLL+BCE/utility0/SAM0.1/LR1e-4/80-14, originalfold5/Q50/9curves/bins/actions/score unchanged. Gate anddistribution can coadapt throughsharednew32width nonlinearity; conditionaloldmeans/scales/weights retained.
Gain: unproved; reference211SW74.14762262262707 and10474.12175282815758. Not capacitymatched pure-gate claim (extra33parameters). OptionaloneSWstage/sixoverall, allmassvariantsalternativesnotstacked, default5; no remainingfolds/fullOOFclaim.
Not worth continuing:213posteriorview belowrepeathidden by0.003417;216Sourcehidden belowsamefunctionalcontrol by0.001143. Both viewsclosed; do notcompound them.217 uses only original128SW features.
Timing: record fullrun/batchwall inclloading/eval/warmup/overlap/memory. Atomicstates/frozen/144665 restore/replay/actualassembly. Validatorclasserrordiagnostics ifused neverfittrainingweightsorper-row selection.

CPU diagnostic104->211: default100786 rows SW98.715632->98.637659 (-0.077974); continuous43879 rows17.631889->17.896277 (+0.264388), weighted contributions -0.054323/+0.080193. Diagnostic only, no fitted weights/oracle.219 tolerance-target loss prepared only;217/218 remain sole two GPU jobs.

V217 final SW74.123007594015, gain+0.001254765857, actualTOTAL80.612914586044; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop1/12, trainingseconds256.543644/peakMiB966.999023. No otherfolds.


