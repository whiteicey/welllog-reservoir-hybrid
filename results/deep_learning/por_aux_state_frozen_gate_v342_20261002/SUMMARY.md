# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# POR auxiliary state with frozen main gate v342

## Focus
Preserve useful continuous-representation learning from340 while mainpredictiongate staysoriginal151. Fromoriginal151only +fresh335raw48/curve64density encoder+49zerogateclassifier,128385active. Auxiliaryga=g0+.5tanh(u/.5) usedONLYtrainingBCE; everymainforward/train/evalreturnsoriginalg0, originaljointRPS usesg0. No340/341trainedweights, noevalaftertraininggatechoice, nofixedroworacle. Allfresh encoder/density/aux jointlyonePORphase/max8.

## Controls and verification
TotalL=L335-B0+Ba; bothBCEwhole-rowQmean/3, no doubleBCE/balance/coefficient. Vs340same128385params/init, aux-vs-main roles andRPSgradientchange notpurecausal. Vs335extra49/notcapacitymatched; ordinary335BCEwasconstant, effectivebody gradienttasknowadded. MainmixedCE/directCONTCE/RPS/alpha.25/decoder/LR/split/seed unchanged. Predefined42 vs335actual42, terminal80/14 ownwholebest, noCONTepochselection. Atomicstates/ordinaryinitialscore/frozenprior/RAM144665replay atterminal, noextra preflight/review.

## Evidence and gains
340natural43/best29 whole77.992995342042/CONT41.916925164121/DEFAULT97.227550685164; vs335whole-.049217293/CONT+.192414334/DEFAULT-.178047204, source151CONTstill-.037035891. Earlierbest20CONT42.05959notfinalselectedepoch andnot nativebreakthrough. Tradeoffmotivatesexactoneauxrole candidate, notpromisedinheritance. CandidatePREPARED/UNTRAI

tant, effectivebody gradienttasknowadded. MainmixedCE/directCONTCE/RPS/alpha.25/decoder/LR/split/seed unchanged. Predefined42 vs335actual42, terminal80/14 ownwholebest, noCONTepochselection. Atomicstates/ordinaryinitialscore/frozenprior/RAM144665replay atterminal, noextra preflight/review.

## Evidence and gains
340natural43/best29 whole77.992995342042/CONT41.916925164121/DEFAULT97.227550685164; vs335whole-.049217293/CONT+.192414334/DEFAULT-.178047204, source151CONTstill-.037035891. Earlierbest20CONT42.05959notfinalselectedepoch andnot nativebreakthrough. Tradeoffmotivatesexactoneauxrole candidate, notpromisedinheritance. CandidatePREPARED/UNTRAINED/no score. Retained3388TOTAL80.704522967002 unchanged. Requirewhole andCONT>335, whole>340; actualfixedtargetcombination and stronger338reference beforecurrentrecipeadoption. Alwaysreport335/340/151strata gaps; wholepositive/CONTnegative notclaimCONTpreserved.

## Limits and closure
301density-ownedbodydetach/304PCGrad/SW331nearnegatives,309PERMdefaultdensityCECONTnegative, oldPORgatemicrotuning negative; limitedsearchno exactfreshintersection. FreezegateprobabilitynotDEFAULTactionguarantee: densitystillchangesBayesutility. No teacher/priorKL/extradata/pseudolabels/validationweights. IfwholeorCONTfails335closeauxrole/singleboundsetting, no BCE/bound/detach/LRgrid ortrainedrepairchain. Strict80.8/fullOOF82unmet.

## Completed and not done
Research/source/PLAN/localsyntax prepared,341activeunchanged. NoGPU/queuefor342; laterexplicit solo boundedattempt. Recordactualwall/warmup/data/overlap/peak andterminal. Nootherfolds/deliverychanges.

2026-10-02 explicitsolo PID50104 after343terminal, globalGPU0. INITIAL77.926380760724/128385params, epoch1 normal/2074.652MiB. Running/unproven. Reference now341 optional8 TOTAL80.704705566183; no autorepairqueue.

V342 final POR78.041027224873, gain+0.114646464150, actualTOTAL80.646869357239; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop28/42, trainingseconds497.890525/peakMiB2074.651855. No otherfolds.

2026-10-02 terminal342 natural42/best28 POR78.041027224873, CONT41.721102080908, DEFAULT97.405597888869. Vs335 whole-.001185410129/CONT-.003408749230; vs340 whole+.048031882831 butCONT-.195823083212, native151CONT-.232858973996. Neither absolute whole+CONT condition holds, no adoption/auxrole setting closed/no grid.497.890525s individual/502.480048s process wall solo includes prep/warmup/restoration,2074.652MiB; all144665RAMbest replay/frozenexact notmodeldiskreload/fullOOF. Current341 eight-stage TOTAL80.704705566183 retained.


