# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW mass readout: likelihood-only ablation
Focus: isolate whether209 mass correction needs joint-action utility. Exactly the same12576parameter module, parent104 SHA, initial tensors/function, data/seed/Q50/bins/actions, categorical NLL/BCE, SAM0.1 andLR1e-4. Only utility alpha1 to0 changes; reward graph still calculated/traversed.
Explored: prepared/untrained, nolaunch/noqueue. Preserve209source/weights untouched; independentlyload104, never209warmstack. Originalgate frozen, bound0.5logmass, floor1e-12 normalization initialization measured separately.
Gain: unproved;209stilltraining, use its final restored ownSW score as matchedcontrol. Baseline10474.12175282815758. At mostone optionalSWstage/sixoverall, defaultfive;211 replaces209 ifadopted, never addsanotherstage. FullOOF>82 unachieved.
Not worth continuing: no utility grid predefined. This one ablation attributes candidate gain rather than claimrepresentation alone; negative199/206closed, no automaticrestart.
Timing:80/14/min10/64steps normalGPUmax2; atomicstates/frozen/replay/actualassembly. Record single andsamebatchwall incldata/eval/overlap. Identical adaptivepayload/decoder preserved.
`nThe initializer copied from209 contains a stale utility1 descriptive flag. Actual trainer STAGES/config/criterion coefficient is0, recorded in CONFIG_ANNOTATION.json; active sources remain immutable. Utility graph is computed but its l

# SW mass readout: likelihood-only ablation
Focus: isolate whether209 mass correction needs joint-action utility. Exactly the same12576parameter module, parent104 SHA, initial tensors/function, data/seed/Q50/bins/actions, categorical NLL/BCE, SAM0.1 andLR1e-4. Only utility alpha1 to0 changes; reward graph still calculated/traversed.
Explored: prepared/untrained, nolaunch/noqueue. Preserve209source/weights untouched; independentlyload104, never209warmstack. Originalgate frozen, bound0.5logmass, floor1e-12 normalization initialization measured separately.
Gain: unproved;209stilltraining, use its final restored ownSW score as matchedcontrol. Baseline10474.12175282815758. At mostone optionalSWstage/sixoverall, defaultfive;211 replaces209 ifadopted, never addsanotherstage. FullOOF>82 unachieved.
Not worth continuing: no utility grid predefined. This one ablation attributes candidate gain rather than claimrepresentation alone; negative199/206closed, no automaticrestart.
Timing:80/14/min10/64steps normalGPUmax2; atomicstates/frozen/replay/actualassembly. Record single andsamebatchwall incldata/eval/overlap. Identical adaptivepayload/decoder preserved.
`nThe initializer copied from209 contains a stale utility1 descriptive flag. Actual trainer STAGES/config/criterion coefficient is0, recorded in CONFIG_ANNOTATION.json; active sources remain immutable. Utility graph is computed but its loss coefficient is0.

V211 final SW74.147622622627, gain+0.025869794469, actualTOTAL80.621529846058; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop10/24, trainingseconds551.402990/peakMiB908.493652. No otherfolds.

V211 actual optional6 TOTAL80.621529846058; actual optional7 (PERM207+SW209) TOTAL80.622609310221, at mostone extra perhead. Both fixed144665assembly, no rowwise model choice. Default5 kept because gains small; below80.8, no remainingfolds/fullOOF claim.


