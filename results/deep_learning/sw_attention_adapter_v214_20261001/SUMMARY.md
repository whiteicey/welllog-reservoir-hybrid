# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW first-attention low-rank adaptation
Focus: limited backbone-side change, preserving all trained104 body/readout. Replace firstglobal QKV linear96->288 by old(x)+up8->288(down96->8(x)). Upzero gives initial identity; only3072parameters train, oldweights unchanged. No other layers replaced.
Explored: prepared/untrained/noqueue. Original ownSW continuousNLL/BCE+jointutility1/temp0.05/SAM0.1/LR1e-4, original1024window/9curves/206features/relative/wholewell/Q50/bins/actions/scoring preserved. Baseeval disables keydrop; gradients through remaining frozen layers to adapter must flow (no outer no_grad). Effective attention can change, oldweight tensors remainexact.
Gain: unproved; primary104SW74.12175282815758/TOTAL80.61247541799393. OptionaloneSWstage/sixoverall, alternative to209/211/212 neverstack. Default5 remains. This is baseline-versusadapter, no capacitymatched mechanism claim.
Not worth continuing: finalfeature9point adapter210 best0/14 negative, not general rejection of internal attention changes. No rank/layer grid predefined; close this placement if no gain. Preserve all positives/failure sources.
Budget/timing:80/14/min10/b64/64steps normalGPUmax2; atomicstate/frozen/all144665 restoredreplay and actualassembly. Record model/run/batchwall and memory, including loading/validation/overlap. No pretrained external weights or labels beyond owntrainingfold.

V211 restored SW

 train, oldweights unchanged. No other layers replaced.
Explored: prepared/untrained/noqueue. Original ownSW continuousNLL/BCE+jointutility1/temp0.05/SAM0.1/LR1e-4, original1024window/9curves/206features/relative/wholewell/Q50/bins/actions/scoring preserved. Baseeval disables keydrop; gradients through remaining frozen layers to adapter must flow (no outer no_grad). Effective attention can change, oldweight tensors remainexact.
Gain: unproved; primary104SW74.12175282815758/TOTAL80.61247541799393. OptionaloneSWstage/sixoverall, alternative to209/211/212 neverstack. Default5 remains. This is baseline-versusadapter, no capacitymatched mechanism claim.
Not worth continuing: finalfeature9point adapter210 best0/14 negative, not general rejection of internal attention changes. No rank/layer grid predefined; close this placement if no gain. Preserve all positives/failure sources.
Budget/timing:80/14/min10/b64/64steps normalGPUmax2; atomicstate/frozen/all144665 restoredreplay and actualassembly. Record model/run/batchwall and memory, including loading/validation/overlap. No pretrained external weights or labels beyond owntrainingfold.

V211 restored SW74.147622622627 +0.025869794469 vs104, +0.005438135132 vs same209 utility1; best10/stop24,551.40299s/908.494MiB. V212 bound2 SW74.144913000682, +0.002728513187 vs209; best9/stop23,542.029938s/908.494MiB. Both actual144665 frozen/restored/assembly done, positives kept, no broaderutility/boundgrid. V214 first-attention rank8 backbone repair explicitPID60644; V213 posterior view explicitPID86544, repeatmatchedcontrol prepared/noqueue; at most2GPU and oneextraSWstage alternatives. Default5 retained, no remainingfolds/fullgoal>82 unmet. V214 interpolation/attention backward warnings mean no exact trajectory reproducibility claim; inference/replay checks remain required.


V214 final SW74.124181854091, gain+0.002429025933, actualTOTAL80.613325577071; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop1/15, trainingseconds341.289236/peakMiB1255.336914. No otherfolds.

V214 completed SW74.124181854091 +0.002429025933 vs104 best1/15,341.289236s/1255.337MiB; actual144665assembly/frozen/restore done; localpositive retained but no layer/rankgrid. V213 posterior completed SW74.134546961565 +0.012794133407 vs104, -0.013075661062 vs211, best5/19,371.503420s/1069.142MiB; parameter-matchedrepeat pending. Explicit213repeat PID84596 and215source PID54776 nowlaunched, twoGPUmax, noextraSOURCEtraining/no predictionvotes. AllcandidatesoneSWrepair alternative, default5 remains/fullOOF>82 unmet.



