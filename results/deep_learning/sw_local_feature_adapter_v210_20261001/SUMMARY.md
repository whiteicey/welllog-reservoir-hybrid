# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW local feature adapter before frozen readout
Focus: small architectural extension preserving104 body and learned Gaussian readout. New128->32->depthwise9->128 input residual bound0.1masked local RMS, zerooutput, no other repairweights.
Explored: prepared/untrained. All original weights frozen/eval, only8672newparameters train; gradients through unchanged frozen output to adapter (no outer no_grad). Original ownSW continuous NLL+BCE+jointutility1/temp0.05/SAM0.1,LR1e-4 preserved. Original ninecurves/206features/relative/wholewell/fold5/Q50/256bins/800actions/scoring unchanged. Gate/means/scales/mixweights may all respond to modified features, with old coefficients fixed.
Gain: not proved; baseline SW74.12175282815758/TOTAL80.61247541799393. Optional oneaddedSWstage/six overall; candidate209 massrepair and210 featureadapter are alternatives, never stacked. Default5; at most8if three head repairs each justify one.
Not worth continuing: POR206 equivalent9point featureadapter no gain underPOR151, so notclaimgeneral benefit; SW has different body/readout/objective. SW199Gaussian parameter patch no gain;210 couples old output factors through shared feature correction. Close if no gain, no parametergrid by default.
Budget/timing:80/14/min10/b64/64steps length1024 max2normalGPU. Atomicstates/frozen/all144665restoredreplay/actualassembly. Record loading/training/eval wall, actual ove

# SW local feature adapter before frozen readout
Focus: small architectural extension preserving104 body and learned Gaussian readout. New128->32->depthwise9->128 input residual bound0.1masked local RMS, zerooutput, no other repairweights.
Explored: prepared/untrained. All original weights frozen/eval, only8672newparameters train; gradients through unchanged frozen output to adapter (no outer no_grad). Original ownSW continuous NLL+BCE+jointutility1/temp0.05/SAM0.1,LR1e-4 preserved. Original ninecurves/206features/relative/wholewell/fold5/Q50/256bins/800actions/scoring unchanged. Gate/means/scales/mixweights may all respond to modified features, with old coefficients fixed.
Gain: not proved; baseline SW74.12175282815758/TOTAL80.61247541799393. Optional oneaddedSWstage/six overall; candidate209 massrepair and210 featureadapter are alternatives, never stacked. Default5; at most8if three head repairs each justify one.
Not worth continuing: POR206 equivalent9point featureadapter no gain underPOR151, so notclaimgeneral benefit; SW has different body/readout/objective. SW199Gaussian parameter patch no gain;210 couples old output factors through shared feature correction. Close if no gain, no parametergrid by default.
Budget/timing:80/14/min10/b64/64steps length1024 max2normalGPU. Atomicstates/frozen/all144665restoredreplay/actualassembly. Record loading/training/eval wall, actual overlap with209, batchwall and memory; not whole-recipe speed claim.

V210 final SW74.121752828158, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds360.206421/peakMiB1018.911621. No otherfolds.

V210 finished SW74.121752828158 best0/14 no learnedgain,360.206421s/1018.912MiB; all144665 replay/frozen/actualassembly, negativebest/lastdeleted withSHAmanifest. No further samefeatureadapter sweep. V211 matching utilityalpha0ablation explicitlylaunch PID88568,209live unchanged; max2GPU, oneSWstage alternatives not stacking. V212 bound2 prepared/no automaticqueue. Default5 maintained, fullOOF>82 not achieved.




