# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW existing local filter fine-tune

Focus: tune only learned104 depthwise65tap96channel coefficients (6240 existing parameters), retaining norm,TCN,global attention,well conditioning and output weights. Same initial104 function,no new parameters.

Explored: prepared/untrained,LR3e-5,SAM0.1,ownoriginal continuousNLL/BCE+utility1,allblocks eval/noDropKey,originalfold5/Q50/80-14. Noother repair weights.

Gain: unproved,controlSW74.121752828158. OneSW extra stage/six total,not a new decoder/window.

Not worth continuing:218 firstQKV+DropKey no gain; this adjusts local backbone feature filter before both global/detail paths, different from final-hidden9tap feature adapter210 (no gain). Single trial, no filter-window grid.

Record timing including load/warmup/eval,overlap/peak and actual restored144665 three-channel score. FullOOF82 unachieved; no otherfolds.

V223 final SW74.121752828158, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds271.268866/peakMiB1269.003906. No otherfolds.

222 final SW74.141314487722, gainvs104+0.019561659564 but width128 below32 by0.006308134905; best7/stop13,245.395531s/933.126465MiB, close width expansion.223 finalSW74.121752828158 best0/14 no gain,271.268866s/1269.003906MiB; negative best/last deleted withSHA manifest after actual assemb

# SW existing local filter fine-tune

Focus: tune only learned104 depthwise65tap96channel coefficients (6240 existing parameters), retaining norm,TCN,global attention,well conditioning and output weights. Same initial104 function,no new parameters.

Explored: prepared/untrained,LR3e-5,SAM0.1,ownoriginal continuousNLL/BCE+utility1,allblocks eval/noDropKey,originalfold5/Q50/80-14. Noother repair weights.

Gain: unproved,controlSW74.121752828158. OneSW extra stage/six total,not a new decoder/window.

Not worth continuing:218 firstQKV+DropKey no gain; this adjusts local backbone feature filter before both global/detail paths, different from final-hidden9tap feature adapter210 (no gain). Single trial, no filter-window grid.

Record timing including load/warmup/eval,overlap/peak and actual restored144665 three-channel score. FullOOF82 unachieved; no otherfolds.

V223 final SW74.121752828158, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds271.268866/peakMiB1269.003906. No otherfolds.

222 final SW74.141314487722, gainvs104+0.019561659564 but width128 below32 by0.006308134905; best7/stop13,245.395531s/933.126465MiB, close width expansion.223 finalSW74.121752828158 best0/14 no gain,271.268866s/1269.003906MiB; negative best/last deleted withSHA manifest after actual assembly, rest evidence retained. Current live224POR mass (initial77.926320949376 vs15177.926380760724, float renormalization -0.000059811348 not learned gain) and225SW lastFF residual. Max2GPU/default5/best optional7TOTAL80.622804193366, no otherfolds.


