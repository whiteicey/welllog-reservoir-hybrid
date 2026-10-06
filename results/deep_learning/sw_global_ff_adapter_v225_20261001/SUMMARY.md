# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW last global feedforward residual

Focus: preserve original104 and add one zero-output96->32->96 nonlinear residual at last global attention FF, bound0.1 per feature,6272parameters. OriginalFF remains.

Explored: prepared/untrained; original owncontinuousNLL/BCE+utility1,SAM0.1,LR3e-4,allblocks eval/noDropKey,originalfold5/Q50/80-14. No mass/QKV/filter repair weights.

Gain: unproved,reference104SW74.121752828158. Same initialfunction; oneSWextra stage/six total,not stackedwith211/219.

Not worth continuing:218 originalQKV and223 localfilter updates best0, close these fine-tune scopes.221 four distance biases tiny+0.000404,no broader sweep.225 changes late global nonlinear capacity with old function frozen instead of altering oldweights. No layer/width grid planned.

Timing includes loading/warmup/eval,actual overlap/peak. Restore/replay and actual144665 assembly needed; no exactGPU backward trajectory claim.

V225 final SW74.128789410226, gain+0.007036582068, actualTOTAL80.614938221718; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop1/14, trainingseconds234.072998/peakMiB1199.011230. No otherfolds.

225 finalSW74.128789410226, gain104+0.007036582068 but below retained219 by0.019390021386; best1/stop14,234.072998s/1199.011230MiB. Restored-state replay/frozen exact/actualassembly; positive kept, no FF width/layer expansion.226 l

# SW last global feedforward residual

Focus: preserve original104 and add one zero-output96->32->96 nonlinear residual at last global attention FF, bound0.1 per feature,6272parameters. OriginalFF remains.

Explored: prepared/untrained; original owncontinuousNLL/BCE+utility1,SAM0.1,LR3e-4,allblocks eval/noDropKey,originalfold5/Q50/80-14. No mass/QKV/filter repair weights.

Gain: unproved,reference104SW74.121752828158. Same initialfunction; oneSWextra stage/six total,not stackedwith211/219.

Not worth continuing:218 originalQKV and223 localfilter updates best0, close these fine-tune scopes.221 four distance biases tiny+0.000404,no broader sweep.225 changes late global nonlinear capacity with old function frozen instead of altering oldweights. No layer/width grid planned.

Timing includes loading/warmup/eval,actual overlap/peak. Restore/replay and actual144665 assembly needed; no exactGPU backward trajectory claim.

V225 final SW74.128789410226, gain+0.007036582068, actualTOTAL80.614938221718; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop1/14, trainingseconds234.072998/peakMiB1199.011230. No otherfolds.

225 finalSW74.128789410226, gain104+0.007036582068 but below retained219 by0.019390021386; best1/stop14,234.072998s/1199.011230MiB. Restored-state replay/frozen exact/actualassembly; positive kept, no FF width/layer expansion.226 live native context fine-tune,227 live input-conditioned residual (one smallmodule);228 prepared true-label RPS-only matchedloss ablation after early226 decline. Max2GPU/no stacks/no otherfolds.


