# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# POR proper-CDF-loss refinement

Focus: isolate density likelihood versus CDF training in the retained151 readout. Same226 parent151/contexthead/230944parameters/LR3e-5/seed/optimizer, but RPS-only instead of NLL+BCE+RPS. Gate fixed so its BCE would have no trainable gradient. True train-label empiricalCDF on original255 finite edges, includes default atom. No teacher/validation-derived targets.

Explored: prepared/untrained,originalmixedT3/Cauchy parameterization retained,113/SOURCE/gate/scale/routes frozen,originalfold5/Q50/80-14/noSAM. No226/224 trainedweights.

Gain: unproved,control151POR77.926380760724 and matched226 outcome pending. OnePORextra stage/six total or8 with207/219,zero newparameters.

Not worth continuing: categoricalmass224 had bestinitial/no gain; keep continuous parameterization.226 early NLL+RPS fine-tune sharply lowers validation despite native loss; this single loss ablation checks whether likelihood gradient opposes CDF utility before dropping this refinement direction. No loss-weight grid planned.

Timing includes loading/warmup/eval,actualoverlap/peak. Restored RAM replay/frozen/actual144665 assembly required; no diskreload claim. No otherfolds until strictTOTAL80.8 and gain.

V228 final POR77.926380760724, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop

# POR proper-CDF-loss refinement

Focus: isolate density likelihood versus CDF training in the retained151 readout. Same226 parent151/contexthead/230944parameters/LR3e-5/seed/optimizer, but RPS-only instead of NLL+BCE+RPS. Gate fixed so its BCE would have no trainable gradient. True train-label empiricalCDF on original255 finite edges, includes default atom. No teacher/validation-derived targets.

Explored: prepared/untrained,originalmixedT3/Cauchy parameterization retained,113/SOURCE/gate/scale/routes frozen,originalfold5/Q50/80-14/noSAM. No226/224 trainedweights.

Gain: unproved,control151POR77.926380760724 and matched226 outcome pending. OnePORextra stage/six total or8 with207/219,zero newparameters.

Not worth continuing: categoricalmass224 had bestinitial/no gain; keep continuous parameterization.226 early NLL+RPS fine-tune sharply lowers validation despite native loss; this single loss ablation checks whether likelihood gradient opposes CDF utility before dropping this refinement direction. No loss-weight grid planned.

Timing includes loading/warmup/eval,actualoverlap/peak. Restored RAM replay/frozen/actual144665 assembly required; no diskreload claim. No otherfolds until strictTOTAL80.8 and gain.

V228 final POR77.926380760724, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds183.171263/peakMiB924.547363. No otherfolds.

227 finalSW74.121752828158,best0/14,no gain,242.930387s/1612.75MiB;228 finalPOR77.926380760724,best0/14,no gain,183.171263s/924.547363MiB.229 amplitude finalSW74.121752828158,best0/14,no gain,165.671455s/1397.282227MiB; repeat untrained/held as negativeamplitude does not warrant control cost, no featuregain claim. All actual144665 assemblies/replays/frozen exact; negative best/last deleted withSHA manifests, positive104/151/154 preserved.230PERM conditionalmass and231SW additiveBiGRU16 are current twoexplicitGPUjobs, default5/bestoptional7TOTAL80.622804193366 unchanged. No otherfolds.


