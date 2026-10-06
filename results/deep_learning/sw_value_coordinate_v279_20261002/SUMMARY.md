# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW v279 continuous coordinate energy

## Focus
New rawinput branch: original207 plus ninecurves17orderedneighbors and17validmask=377 input, MLP48/96/48,32coefficients. New valuecoordinate trunk1/32/32/32 at original256 transformedgridmidpoints; dot/sqrt32 produces categoricallogits. Both encoders/readout train3e-4 in onephase. Ordered coordinates vs seededshuffledcoordinates, sameparams/init/uniformcomponent. Neither oldbin/action order nor decoding changes. Fixed104 prior/atom protected by alpha0?.1 in5epochs; no trained275/278 repair weights.

## Explored and validation
Prepared/untrained. DirectconditionalCE1 halfhard/halfphysical5pcttarget independentofalpha, mixedconditionalCE/RPS1 as275. Aux CE already excludesdefaults; this tests target-value geometry and input-neighbor representation instead. Coefficients zero at initialization: firststep trunk/inputencoder gradients zero, second optimizerstep expected nonzero; same normaltrainer recordedgradient atstep1. Initial final prediction same prior. Relative/raworiginalfeatures retainunboundedinfo; addedneighborhood zero-pad invalid positions with explicitflags, no crosswell input. DeepONet-inspired factorization, not full operator reproduction or proven measurement physics. Capacity differs from275; pair isolates coordinate order only.

## Gains
Unproved. Predefine ordered-minus-shuffled and absolute SW25274.14922820781207. 

# SW v279 continuous coordinate energy

## Focus
New rawinput branch: original207 plus ninecurves17orderedneighbors and17validmask=377 input, MLP48/96/48,32coefficients. New valuecoordinate trunk1/32/32/32 at original256 transformedgridmidpoints; dot/sqrt32 produces categoricallogits. Both encoders/readout train3e-4 in onephase. Ordered coordinates vs seededshuffledcoordinates, sameparams/init/uniformcomponent. Neither oldbin/action order nor decoding changes. Fixed104 prior/atom protected by alpha0?.1 in5epochs; no trained275/278 repair weights.

## Explored and validation
Prepared/untrained. DirectconditionalCE1 halfhard/halfphysical5pcttarget independentofalpha, mixedconditionalCE/RPS1 as275. Aux CE already excludesdefaults; this tests target-value geometry and input-neighbor representation instead. Coefficients zero at initialization: firststep trunk/inputencoder gradients zero, second optimizerstep expected nonzero; same normaltrainer recordedgradient atstep1. Initial final prediction same prior. Relative/raworiginalfeatures retainunboundedinfo; addedneighborhood zero-pad invalid positions with explicitflags, no crosswell input. DeepONet-inspired factorization, not full operator reproduction or proven measurement physics. Capacity differs from275; pair isolates coordinate order only.

## Gains
Unproved. Predefine ordered-minus-shuffled and absolute SW25274.14922820781207. Restore best officialSW overall144665, actualcombinedTOTAL required. RAMreplay notdiskreload. OneSWaddedphase/sixalone/sevenwithPERM274, <=8. Originalfivefold82 unmet, strictfifth80.8+gain beforeotherfolds.

## Not worth continuing
Lowrank32 basis may smooth too strongly, shuffledinput might learn highfrequency functions, realgrid unequal spacing preserved. Positivepair does not prove neighborhood itself helps. If ordered failscontrol and retained252, close thisbasis/coordinate/ramp configuration; no basis-width/neighborhood/alpha grid. No oracleperrow selection/teacher/training-to-test labels.

## Timing
Individual and actual2concurrent wall separately. Include startup/data/initial/final, exclude codeprep/CPUassembly; peakmeasured notestimated. Prepared holds untilGPU slots free. Research source to be attached from independentagent.

V279 ordered failed startup: infinite terminal-edge midpoint createsNaN coordinate; initial0 invalid/no validscore. Shuffled neverlaunched. New280 finiteoriginalphysicalcenter correction, notrainedweights resume.


