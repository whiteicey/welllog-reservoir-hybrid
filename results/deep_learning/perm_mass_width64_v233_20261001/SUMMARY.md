# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# PERM positive probability-readout width64

Focus: expand positive230 mass correction width32->64,24896parameters,all154 weights frozen. Same prior/input128/bound0.5/ownbinNLL+BCE+jointRPS1/seed/LR1e-4/noSAM/80-14/fold5/Q50. Zero last projection,initialfunction same except inheritedfloor/floatnormalization.

Explored: prepared/untrained; control230 PERM89.413121051402 (+0.007556 vs154). No230trainedweights,each sibling is one extraPERMstage/six total. Width changes initialization shape/capacity,not samecapacity ablation.

Gain: unproved; report matchedwidth delta and absolute154 delta after actual best restoration/three-channel assembly. Default5 remains,optional7 withSW219; no otherfolds until TOTAL>80.8.

Not worth continuing: PORwidehead fine-tunes226/228 bothbest0; no further nativehead-wide loss/LR grid. SWwidth128 was below32,doesnot prove PERMwidth64 ineffective. Single positive-mechanism capacity followup,not broad grid.

Timing includes loading/warmup/eval,actualoverlap/peak,not sum of concurrent times as serialbaseline. Positive230 and154 preserved.

V233 final PERM89.409673958189, gain+0.004109101270, actualTOTAL80.613913603438; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop6/20, trainingseconds213.481596/peakMiB791.437500. No otherfolds.

232 finalSW74.133475000603,best2/16,252.972477s/1205.791992MiB, gain104+0.0117

# PERM positive probability-readout width64

Focus: expand positive230 mass correction width32->64,24896parameters,all154 weights frozen. Same prior/input128/bound0.5/ownbinNLL+BCE+jointRPS1/seed/LR1e-4/noSAM/80-14/fold5/Q50. Zero last projection,initialfunction same except inheritedfloor/floatnormalization.

Explored: prepared/untrained; control230 PERM89.413121051402 (+0.007556 vs154). No230trainedweights,each sibling is one extraPERMstage/six total. Width changes initialization shape/capacity,not samecapacity ablation.

Gain: unproved; report matchedwidth delta and absolute154 delta after actual best restoration/three-channel assembly. Default5 remains,optional7 withSW219; no otherfolds until TOTAL>80.8.

Not worth continuing: PORwidehead fine-tunes226/228 bothbest0; no further nativehead-wide loss/LR grid. SWwidth128 was below32,doesnot prove PERMwidth64 ineffective. Single positive-mechanism capacity followup,not broad grid.

Timing includes loading/warmup/eval,actualoverlap/peak,not sum of concurrent times as serialbaseline. Positive230 and154 preserved.

V233 final PERM89.409673958189, gain+0.004109101270, actualTOTAL80.613913603438; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop6/20, trainingseconds213.481596/peakMiB791.437500. No otherfolds.

232 finalSW74.133475000603,best2/16,252.972477s/1205.791992MiB, gain104+0.011722172445 but below219 by0.014704431009.233 finalPERM89.409673958189,best6/20,213.481596s/791.4375MiB, width64 below32 by0.003447093213; close widthgrid,positive controls retained. Both actual144665 assemblies/restoredreplay/frozen exact.234PERM tolerance-loss and235SW joint memory+mass(one phase,26880newparams) explicitly launched/max2GPU. Bestoptional7TOTAL80.624369397272 unchanged,default5 kept/no otherfolds.


