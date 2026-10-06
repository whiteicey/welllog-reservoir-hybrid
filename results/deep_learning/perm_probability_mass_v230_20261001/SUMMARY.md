# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# PERM retained conditional mass readout

Focus: retain positive154Gaussian16 prior and originalgate; add zero-output128->32->256 bounded0.5 conditionalmass correction,12576params. Capture learned154 hidden features,not old prediction voting.

Explored: prepared/untrained,ownbinNLL/BCE+originaljoint255-edgeRPS1,LR1e-4/noSAM,originalfold5/Q50/80-14/bins/actions. Likelihood changes with representation; no pure architecture claim. No207/scale/otherrepair weights.

Gain: unproved,controlPERM89.405564856919. Initialfloor/renormalization separately measured. OnePERMextra phase/six total or7 withSW219,at most8 withonePORrepair. Default5 kept.

Not worth continuing: nativePOR refinement226/228 bestinitial,no further repeat of head-wide fine-tuning. PORmass224 negative does not prove PERM Gaussian density readout ineffective; this is one scoped target-specific transfer,not blind feature/width search.

Timing includes loading/warmup/eval/actualoverlap/peak; RAM restored replay/frozen/actual144665 assembly required. No testlabels/pseudo labels/teacher supervision or decoder changes.

V230 final PERM89.413121051402, gain+0.007556194483, actualTOTAL80.615120086063; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop10/24, trainingseconds178.528288/peakMiB791.250000. No otherfolds.

230 finalPERM89.413121051402, gain154+0.007556194483/gain207+0.00

# PERM retained conditional mass readout

Focus: retain positive154Gaussian16 prior and originalgate; add zero-output128->32->256 bounded0.5 conditionalmass correction,12576params. Capture learned154 hidden features,not old prediction voting.

Explored: prepared/untrained,ownbinNLL/BCE+originaljoint255-edgeRPS1,LR1e-4/noSAM,originalfold5/Q50/80-14/bins/actions. Likelihood changes with representation; no pure architecture claim. No207/scale/otherrepair weights.

Gain: unproved,controlPERM89.405564856919. Initialfloor/renormalization separately measured. OnePERMextra phase/six total or7 withSW219,at most8 withonePORrepair. Default5 kept.

Not worth continuing: nativePOR refinement226/228 bestinitial,no further repeat of head-wide fine-tuning. PORmass224 negative does not prove PERM Gaussian density readout ineffective; this is one scoped target-specific transfer,not blind feature/width search.

Timing includes loading/warmup/eval/actualoverlap/peak; RAM restored replay/frozen/actual144665 assembly required. No testlabels/pseudo labels/teacher supervision or decoder changes.

V230 final PERM89.413121051402, gain+0.007556194483, actualTOTAL80.615120086063; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop10/24, trainingseconds178.528288/peakMiB791.250000. No otherfolds.

230 finalPERM89.413121051402, gain154+0.007556194483/gain207+0.004472011161, best10/stop24,178.528288s/791.25MiB. Initial floor+float delta+0.000008777740, learnedgain+0.007547416743 separated. Actual optional7(POR151+PERM230+SW219)TOTAL80.624369397272, gain previous7+0.001565203906, all144665 fixedchannel assembly; default5 unchanged/below80.8/no otherfolds.231 failed firstbackward(no epochs) because GRU trainflag was absent;232freshfixexplicitGRUtraining.233 width64 sibling from154 prepared; no230 repair stacking.

Gain diagnosis,not trainingfit: PERM default66.037% scores97.278705->97.281652,continuous33.963%74.096912->74.113431; aggregate+0.007556. SW default69.669%98.715632->98.644703,continuous30.331%17.631889->17.881934; weighted gains-0.049416/+0.075842,net+0.026427. ContinuousSW zero-official-score66.389%,pred>=99.9 30.281%; latter is action range,not directgateclassification. Frozen-small-repair scope limits correction; representation/generalization hypothesis remains unproven, no noise/data-ceiling claim. Diagnostics saved CHANNEL_GAIN_STRATA.json and CONTINUOUS_ERROR_DIAGNOSTIC.json.


