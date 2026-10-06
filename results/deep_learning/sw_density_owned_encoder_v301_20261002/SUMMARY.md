# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW density-owned raw representation v301

## Focus / evidence
Recent newraw joint state/density encoders mainly increase default score while continuous deteriorates: 289continuous17.72013, 297well17.50275, 296pooled17.46917, 299hierarchical17.40412, versus frozen-feature25217.88487. This does not prove gradient conflict, but motivates isolating gradient ownership rather than more width/normalizer sweeps.

EXACT original289 code/body/readout/init/loss/share.25ramp5/statebound4/balancedBCE/directcontinuousCE1/conditionalRPS1/80-14, ORIGINAL104 checkpoint only. Fresh207->48 rawencoder, three pointMLP branches/fuse, direct256density and49state residual train in ONE addedSWstage. Only change: gate_patch receives h.detach(). Stateweights still learn BCE on continually updated rawfeatures; encoder and density still train jointly via continuous mixtureCE/RPS/directcomponentCE. No losses/gates/actions removed or frozen, no inference function/value change from detach alone, no newparams. This can reduce representation competition, but does not guarantee actual gradients conflicted or defaultprediction remains unchanged after training.

## Explored and validation
Prepared/untrained. Historical completed289 is same-capacity/configuration control, not concurrentpaired rerun; original286/284 failures not controls. Only old104 weights loaded, no289/297/299repair weights. Ordinary initialsou

arantee actual gradients conflicted or defaultprediction remains unchanged after training.

## Explored and validation
Prepared/untrained. Historical completed289 is same-capacity/configuration control, not concurrentpaired rerun; original286/284 failures not controls. Only old104 weights loaded, no289/297/299repair weights. Ordinary initialsourcefunction check, allnew optimizergroup gradients/frozenproof/atomic laststate; ownWHOLESW officialbest, not continuous-selected. Existing frozen helper hashes included. Actual144665 composition and strata after terminal; RAMbest replay not independentdiskreload. Prior v11 splitTCN differs from this sharedfeature gradient-routing intervention.

## Gain
Unproved. Report wholeSW-minus289 and continuous-minus289/252, and absolute297overall74.16003022301975. Positive local gain may be carried into a fresh isolated jointstage; only complete useful actualcombination strictly>80.8 qualifies otherfolds. Current optional8/default5/oldfullOOFdelivery retained. Max8wholemethod, oneaddedSWphase, no teacher/testlabels/extra inputs/oracle.

## Unworthy followup
If no meaningful continuous benefit and overall no gain, close this detached-state representation setting, no detachratio/LR/BCEweight grid. Detached changing features can harm state fit, and encoder may still lack discriminative continuous inputs. A failed trial cannot prove all multitask methods fail. Frozen-feature252 is an absolute reference, not samearchitecture fairness control.

## Timing
Individual data/initial/train/finalrestore wall+peak; historical289 wall measuredunder different contention, no speedranking. Codeprep/CPUassemblyexcluded; maximum2 not assumedoptimalparallelism.

V301 density_owned final SW74.148904173757, actualTOTAL80.621978388954; learned gain+0.027151483988, vs211+0.001281551130. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.


## Final gradient-routing outcome
SW74.148904173757/best17/stop31, versus289-0.004822255242; continuous17.494506745789, versus289-0.225620057348/252-0.390361394121. Not adopted; no detachratio/BCE/LR grid. Actualencoder/readout/state nonzerograd/frozen/RAMbest144665assembly; no independentdiskreload. Individual294.684s/peak1241.643MiB, overlaps300/no speedranking. Decoded atom actionfraction is a posterior decision diagnostic, not gateprobability/classifier accuracy; no roworacle or stratum-specific model selection. Currentrecipes unchanged; finishedcleanup remains blocked by automaticpolicy, no deletionperformed.


