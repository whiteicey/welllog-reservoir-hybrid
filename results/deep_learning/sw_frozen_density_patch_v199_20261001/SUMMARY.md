# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v199 One small SW density repair on frozen learned104

## Focus
Extend positive SW104, preserve all learnedbody/attention/FiLM/localfrequency/gate/scale weights. Only a4656parameter128to32/GELU/32to16 zero-output branch repairs8component locations and logweights. Same mechanism family as proven POR/PERM density readout, applied conservatively to the already-trained SW rather than rebuilding it.

## Explored and validation
Prepared/untrained. Exact104SHA07e0b8df208fe93c8ac07dcf18924459c47afb12ec84b1b7306a0e6017f58670 loaded/frozen. Capture normalized128features before originallinearoutput. Locations add +/-0.25currentcomponentSD using bound*tanh(u/bound); logweights add+/-0.25. Original gate andscale functions unchanged. Zero branch gives algebraic initial104 function; actual74.121752828158 score must be checked. Existing NLL/BCE+jointSWutility1/temp0.05 unchanged; original GaussianCDF/binmass/actions/argmax evaluation, no decoder or label changes. Train only newbranch lr1e-4/SAM0.1/AdamW, originalbaseforcedeval/no keydrop; SAM two passes same input/RNG, base frozen. This differs from retraining full104 or replacing Gaussian families; previous155 added jointRPS to randomwhole104 and is not included.
Original206ninecurve/relative/wholewellinputbank/train64/fold5/Q50/seed/b64/64steps/max80/pat14/min10 retained. Newoptimizer, not oldtrajectorycontinuation. Atomicfullstate/current

# v199 One small SW density repair on frozen learned104

## Focus
Extend positive SW104, preserve all learnedbody/attention/FiLM/localfrequency/gate/scale weights. Only a4656parameter128to32/GELU/32to16 zero-output branch repairs8component locations and logweights. Same mechanism family as proven POR/PERM density readout, applied conservatively to the already-trained SW rather than rebuilding it.

## Explored and validation
Prepared/untrained. Exact104SHA07e0b8df208fe93c8ac07dcf18924459c47afb12ec84b1b7306a0e6017f58670 loaded/frozen. Capture normalized128features before originallinearoutput. Locations add +/-0.25currentcomponentSD using bound*tanh(u/bound); logweights add+/-0.25. Original gate andscale functions unchanged. Zero branch gives algebraic initial104 function; actual74.121752828158 score must be checked. Existing NLL/BCE+jointSWutility1/temp0.05 unchanged; original GaussianCDF/binmass/actions/argmax evaluation, no decoder or label changes. Train only newbranch lr1e-4/SAM0.1/AdamW, originalbaseforcedeval/no keydrop; SAM two passes same input/RNG, base frozen. This differs from retraining full104 or replacing Gaussian families; previous155 added jointRPS to randomwhole104 and is not included.
Original206ninecurve/relative/wholewellinputbank/train64/fold5/Q50/seed/b64/64steps/max80/pat14/min10 retained. Newoptimizer, not oldtrajectorycontinuation. Atomicfullstate/currentbest/frozen checks/all144665 ownscore and exactreload; actual fixedPOR151/PERM154 assembly. No teacher/otherlabels/testdata. One addedSWstage only, whole6(SOURCE/POR113/POR151/PERM154/SW104/SW199); if independentPERM repair adopted too, combination totals7, no stacking samehead. Parent weights/delivery neveredited.

## Gains
No training/no gain. Exact functionalcontrolSW74.121752828158/currentTOTAL80.612475417994. Actualinitial equality/learnedSWdelta/TOTAL separately; no predictionaccuracy on unknowntest. Strict80.8+gain before otherfolds, fullOOF>82 unmet. Only one newstage perhead and max8.

## Not worth continuing
No evidence yet. If no gain, close this jointlocation/logweight bound0.25/32width repair, no bound/width/SAMgrid or wholebodyreset. Freeze protects oldfunction; selecting initial if no better. Report realtraining/peak/pairedwall/concurrency/data/initial/finaleval/warmup, not full6stage or equalcompute costs.

V199 final SW74.121752828158, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds158.374369/peakMiB957.755859. No otherfolds.


