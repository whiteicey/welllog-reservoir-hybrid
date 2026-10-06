# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW input-conditioned backbone repair

Focus: keep positive104 whole-well encoder and add one zero-output96->16->1216 residual before its learned modulation tanh.22160newparameters,rawbound0.1; inherited0.5tanh limits modulation change to0.05. Context uses deterministic input-only bank already present in104; no labels/IDs.

Explored: prepared/untrained,oldcondition/body/readout frozen. Ownoriginal continuousNLL/BCE+utility1,SAM0.1,LR1e-4,noDropKey,originalfold5/Q50/80-14. Zeroinitialfunction unchanged. No225/mass/otherrepair weights loaded.

Gain: unproved,control104SW74.121752828158. One extraSWphase/six total,siblings notstacked.

Not worth continuing: isolatedQKV/localfilter oldweight retuning failed; instead this tests input-conditioned modulation with oldweights frozen. No condition-width/bound sweep planned. Does not claim adaptation with heldout labels.

Timing includes loading/warmup/eval,actualoverlap/peak; restored replay and actual144665 assembly required. No otherfolds until strictTOTAL80.8 plus gain.

Runtime clarification: actual condition patch parameters22224 (96*16+16 +16*1216+1216); planned22160 was arithmetic typo. Runtime initializer andconfig correctly calculate22224; active source/plan immutable. No architecture/phase change.

V227 final SW74.121752828158, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedSWstage, strict808=Fa

# SW input-conditioned backbone repair

Focus: keep positive104 whole-well encoder and add one zero-output96->16->1216 residual before its learned modulation tanh.22160newparameters,rawbound0.1; inherited0.5tanh limits modulation change to0.05. Context uses deterministic input-only bank already present in104; no labels/IDs.

Explored: prepared/untrained,oldcondition/body/readout frozen. Ownoriginal continuousNLL/BCE+utility1,SAM0.1,LR1e-4,noDropKey,originalfold5/Q50/80-14. Zeroinitialfunction unchanged. No225/mass/otherrepair weights loaded.

Gain: unproved,control104SW74.121752828158. One extraSWphase/six total,siblings notstacked.

Not worth continuing: isolatedQKV/localfilter oldweight retuning failed; instead this tests input-conditioned modulation with oldweights frozen. No condition-width/bound sweep planned. Does not claim adaptation with heldout labels.

Timing includes loading/warmup/eval,actualoverlap/peak; restored replay and actual144665 assembly required. No otherfolds until strictTOTAL80.8 plus gain.

Runtime clarification: actual condition patch parameters22224 (96*16+16 +16*1216+1216); planned22160 was arithmetic typo. Runtime initializer andconfig correctly calculate22224; active source/plan immutable. No architecture/phase change.

V227 final SW74.121752828158, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds242.930387/peakMiB1612.750000. No otherfolds.

227 finalSW74.121752828158,best0/14,no gain,242.930387s/1612.75MiB;228 finalPOR77.926380760724,best0/14,no gain,183.171263s/924.547363MiB.229 amplitude finalSW74.121752828158,best0/14,no gain,165.671455s/1397.282227MiB; repeat untrained/held as negativeamplitude does not warrant control cost, no featuregain claim. All actual144665 assemblies/replays/frozen exact; negative best/last deleted withSHA manifests, positive104/151/154 preserved.230PERM conditionalmass and231SW additiveBiGRU16 are current twoexplicitGPUjobs, default5/bestoptional7TOTAL80.622804193366 unchanged. No otherfolds.


