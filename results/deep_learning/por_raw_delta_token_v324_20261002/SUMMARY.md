# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# POR raw + normalization displacement token v324

## Focus and controls
Final v322 adaptive/reference gain is +0.001824678338 whole POR and +0.009222531043 continuous, but both below retained302. Test one small change: each numeric token reads raw u and delta=normalized(u)-u through Linear2->32. Both weight columns copy the original scalar weight, bias unchanged, zero normalizer and bridge preserve the initial predictive function. All new raw encoder, token attention, normalizer and readout train jointly from original151, no trained302/322 repair weights.

Two explicit matched arms adaptive/reference differ only statistics source. Active76786 each, +32 versus historical322; comparing historical322 cannot isolate tokenizer gain. Same seed20261010, Q50, original9/206+documented amplitude63, 80/14/min10/b64/64steps/alpha.25ramp5/CE-RPS loss and decoder. One addedPOR phase, six standalone/maxeight full method; no checkpoint chaining or other folds.

## Explored and validation
Implemented module and cloned atomic trainer/recorder. Local syntax check only; initial evaluation belongs to ordinary training. No numerical preflight/review. Initial tokenizer is algebraically w*normalized; independent raw amplitude weights emerge only after learning. Zero original bridge initially delays gradients, not a dead new normalizer/output double-zero initialization.

## Gains
Not trained yet, no c

 TOTAL>80.8 is required before any remaining-fold consideration; full OOF>82 unmet.

## Close / not worth extending
One adaptive/reference pair only. If whole and continuous paired gains do not both persist, close this token position; no tokenizer width, normalizer bound, LR or window grid. Historical raw bypass already exists; do not call this first amplitude/shape modeling. Single-seed micro gain is not robustness proof.

## Timing
Record actual process starts/stagger, individual wall time and total same-work pair wall time, GPU peak, includes data/initial evaluation/training/restoration; exclude implementation/finalization. Concurrency cap is not a speed claim.

V324 adaptive launched PID41036 alongside SW323 after fresh actual GPU process check; reference remains unlaunched, no automatic queue. Final322 recorded, absolute replacement closed; existing best recipes/deliveries unchanged.

V324 reference explicitly launched PID65712 after authoritativeSW323terminal. AdaptivePID41036 and referencePID65712 are the only2 GPU tasks. No automatic queue/remainingfolds.

V324 live adaptive refreshed best20 POR78.026020857869 after best6 plateau; reference best11 POR78.017000201492. Earliernearstop inference superseded by observed20 improvement, not terminalpairedgain. Both PIDs41036/65712 confirmedlive, max2GPU. V325 preparednext, no slotlaunch.

V324 final POR78.027986511084, gain+0.101605750360, actualTOTAL80.642957143102; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop29/43, trainingseconds1046.970454/peakMiB1815.477051. No otherfolds.

V324 final POR78.027189922468, gain+0.100809161745, actualTOTAL80.642718166517; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop29/43, trainingseconds917.975614/peakMiB1815.164551. No otherfolds.

## Final closure
Adaptive/reference bothnaturalstop43/best29: whole78.027986511084/78.027189922468, CONT41.665712583009/41.669385189908. Pairedwhole+0.000796588616 but CONT-0.003672606900; adaptive belowretained302whole-0.002566283060/CONT-0.009367324062. No adoption, close raw+delta token and normalizationextension; no width/bound/LR/windowgrid. Source/bestrecipes/deliveries unchanged. All144665ownbestRAM replay/frozenexact, learnednormalizerupdatesnonzero, no independentdiskreload/fullOOF. Individual1046.970454/917.975614s, pair1179.685143s with256.129325s stagger andinitialSW323overlap, peaks1815.477051/1815.164551MiB; samework wall includesprep/initial/train/bestrestore, excludesimplementation/CPUfinalization, notisolatedspeedranking.


