# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# PERM input-context lower learning rate
Focus: test whether the positive205 cross-context patch can retain more gain with LR3e-5 instead of1e-4.
Explored: new independent experiment from the same frozen154 SHA, same6992parameter LayerNorm/QKV16/output32 patch, same initial function/seed/data/masks/bounds0.1/NLL+BCE+jointRPS/80-14. No205/202 weights loaded. Patch queries16 pooled input-hidden tokens, no history labels, no releasedweights. CATS-inspired but not a reproduction: https://arxiv.org/abs/2405.16877.
Gain: training, no final result. Matched control205 PERM89.40861623101172; original15489.40556485691894. Optional6phase recipe only, default5; no seventh sequentialPERMstage. Joint with at most onePOR repair max7overall.
Not worth continuing: no wider automaticLRgrid; one follow-up because205 verified positive but best1/15. NegativePOR203 closed, stage-removal201 closed. Prior204 startup hook-name failure is historical and unrelated to207 efficacy.
Timing: record full per-run including data/eval and actual overlap with206 plus same-work batch wall; isolated patch is not whole recipe training time. Existing adaptive-pool nondeterministic backward warning disclosed, exact inference/reload checks required; no exact training-trajectory claim.

V207 attention final PERM=89.408649040241, matched205gain=+0.000032809230; actualTOTAL=80.613554882157, sixphases inclsource;strict808=

# PERM input-context lower learning rate
Focus: test whether the positive205 cross-context patch can retain more gain with LR3e-5 instead of1e-4.
Explored: new independent experiment from the same frozen154 SHA, same6992parameter LayerNorm/QKV16/output32 patch, same initial function/seed/data/masks/bounds0.1/NLL+BCE+jointRPS/80-14. No205/202 weights loaded. Patch queries16 pooled input-hidden tokens, no history labels, no releasedweights. CATS-inspired but not a reproduction: https://arxiv.org/abs/2405.16877.
Gain: training, no final result. Matched control205 PERM89.40861623101172; original15489.40556485691894. Optional6phase recipe only, default5; no seventh sequentialPERMstage. Joint with at most onePOR repair max7overall.
Not worth continuing: no wider automaticLRgrid; one follow-up because205 verified positive but best1/15. NegativePOR203 closed, stage-removal201 closed. Prior204 startup hook-name failure is historical and unrelated to207 efficacy.
Timing: record full per-run including data/eval and actual overlap with206 plus same-work batch wall; isolated patch is not whole recipe training time. Existing adaptive-pool nondeterministic backward warning disclosed, exact inference/reload checks required; no exact training-trajectory claim.

V207 attention final PERM=89.408649040241, matched205gain=+0.000032809230; actualTOTAL=80.613554882157, sixphases inclsource;strict808=False, no otherfolds. best/stop=1/15, seconds=185.537, peakMiB=786.465; all144665replay/frozen exact.


