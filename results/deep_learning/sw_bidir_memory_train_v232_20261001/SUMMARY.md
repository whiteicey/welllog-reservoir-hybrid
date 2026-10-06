# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW additive bidirectional token memory

Focus: preserve positive104 final attentionblock; add small normalized BiGRU16perdirection over its64pooled token features and zero-output32->96 projection,bound0.1pointRMS,14304params. Final FiLM/detail/readout remain. Input-only forward/backward sequence memory,not predictions/labels.

Explored: prepared/untrained,ownoriginalcontinuousNLL/BCE+utility1,SAM0.1/LR1e-4,originalfold5/Q50/80-14. Originalblocks eval/noDropKey; new GRU trainingmode only for backward and explicitfloat32. Invalid token inputs zeroed/output masked; padding still traversed by GRU,not packed/label context.

Gain: unproved,reference104SW74.121752828158. Zero last projection preserves initialfunction; oneSWextra stage/six total,no225/229/mass repair weights loaded.

Not worth continuing: isolated oldQKV/localfilter/wholewellcondition refinements failed; localamplitude229 declining. This tests recurrent sequence dynamics instead of more pointwise calibration. No GRU-size/window/layer grid planned.

Timing includes loading/warmup/eval,actualoverlap/peak. Restored RAM replay/frozen/actual144665 assembly required. No otherfolds until TOTAL>80.8+gain; default5 retained.

232 fresh version: explicitly set only new recurrent module training flag after parent.eval();231 intended switch was absent,firstbackwardfailed/no epochs. Keep231source/config/error immutable,232 from10

# SW additive bidirectional token memory

Focus: preserve positive104 final attentionblock; add small normalized BiGRU16perdirection over its64pooled token features and zero-output32->96 projection,bound0.1pointRMS,14304params. Final FiLM/detail/readout remain. Input-only forward/backward sequence memory,not predictions/labels.

Explored: prepared/untrained,ownoriginalcontinuousNLL/BCE+utility1,SAM0.1/LR1e-4,originalfold5/Q50/80-14. Originalblocks eval/noDropKey; new GRU trainingmode only for backward and explicitfloat32. Invalid token inputs zeroed/output masked; padding still traversed by GRU,not packed/label context.

Gain: unproved,reference104SW74.121752828158. Zero last projection preserves initialfunction; oneSWextra stage/six total,no225/229/mass repair weights loaded.

Not worth continuing: isolated oldQKV/localfilter/wholewellcondition refinements failed; localamplitude229 declining. This tests recurrent sequence dynamics instead of more pointwise calibration. No GRU-size/window/layer grid planned.

Timing includes loading/warmup/eval,actualoverlap/peak. Restored RAM replay/frozen/actual144665 assembly required. No otherfolds until TOTAL>80.8+gain; default5 retained.

232 fresh version: explicitly set only new recurrent module training flag after parent.eval();231 intended switch was absent,firstbackwardfailed/no epochs. Keep231source/config/error immutable,232 from104not231weights. Evaluation recurrentflagFalse; originalblocks alwayseval.

V232 final SW74.133475000603, gain+0.011722172445, actualTOTAL80.616578178350; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop2/16, trainingseconds252.972477/peakMiB1205.791992. No otherfolds.

232 finalSW74.133475000603,best2/16,252.972477s/1205.791992MiB, gain104+0.011722172445 but below219 by0.014704431009.233 finalPERM89.409673958189,best6/20,213.481596s/791.4375MiB, width64 below32 by0.003447093213; close widthgrid,positive controls retained. Both actual144665 assemblies/restoredreplay/frozen exact.234PERM tolerance-loss and235SW joint memory+mass(one phase,26880newparams) explicitly launched/max2GPU. Bestoptional7TOTAL80.624369397272 unchanged,default5 kept/no otherfolds.


