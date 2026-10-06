# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v192 Multi-order interaction in private PERM

## Focus and hypothesis
Borrow multi-order spatial aggregation and multiplicative gating from MogaNet, adapted to masked 1D sequences. Isolate multiplication from addition using identical modules and initialization. This is a local variant, not an exact MogaNet reproduction.
Primary source: https://arxiv.org/abs/2211.03295 and https://github.com/Westlake-AI/openmixup/blob/main/openmixup/models/backbones/moganet.py

## Explored and validation
Prepared two arms add/multiply. Retain original185 first detail block; add LayerNorm128, parallel depthwise5 at dilation1/2/3 split16/48/64, pointwise gate/value/project. SiLU(g)*SiLU(v) versus (SiLU(g)+SiLU(v))/sqrt2. Final projection zero initialized, residual scale0.1, so identical initial hidden and predictive function; no assertion of matched learned activation variance or bitwise CUDA trajectory. Mask before depthwise and after each branch; no well crossing. Full private model trains randomly with same seed20261010, SAM0.05, mixed-row/well sampling, Gaussian16, jointRPS1, NLL/BCE,80/14/min10/b64/64steps. Load no185 or downstream pretrained weights. SOURCE preprocessing only; fixed actual POR151/SW104. One replacement PERM phase, whole method5 including SOURCE. Initial/full144665 evaluation and final exact restoration are in trainer; no redundant preflight.

## Gains
Not trained yet; no v

replacement PERM phase, whole method5 including SOURCE. Initial/full144665 evaluation and final exact restoration are in trainer; no redundant preflight.

## Gains
Not trained yet; no verified gain. Primary fair delta multiply minus add after both final. Secondary comparison18588.988533638321 has smaller capacity. Absolute current15489.405564856919/TOTAL80.612475417994 remains unchanged. No remaining folds before strict80.8 plus gain.

## Not worth continuing
No evidence yet. Do not grid dilations, gates or scale if this matched mechanism fails; do not combine with failed wavelet/MixStyle/IQF. Record actual timing/peak and staggered total pair wall, not full5phase or equal-compute cost.

v192 add/multiply both live and GPU guard max2 verified; normal/temporarySYSTEM_REQUIREDTrue. Identical1162673 active params (+50560 versus185), original same predictive initial68.38377243696718; common core and zero-residual initialization, not bitwise CUDA trajectory. Same seed/data/loss/SAM0.05/sampling/80-14, one replacement PERM phase/whole recipe5. Launch snapshots only, no mechanism gain yet.

V192 add final PERM=88.885009729025, compactcontrolgain=-0.493384482119; actualTOTAL=80.430281123231, fivephases inclsource;strict808=False, no otherfolds. best/stop=16/30, seconds=554.660, peakMiB=867.146; all144665replay/frozen exact.

v192 add final PERM88.885009729025, actual TOTAL80.430281123231, best16/stop30, training554.660232s/peak867.146484MiB,144665 exact replay/frozen. Multiply still live; preserve both pair checkpoints until fair final comparison. No pair gain/other folds;185 and current154 unchanged.

V192 multiply final PERM=88.892166252243, compactcontrolgain=-0.486227958901; actualTOTAL=80.432785906357, fivephases inclsource;strict808=False, no otherfolds. best/stop=16/30, seconds=574.888, peakMiB=874.612; all144665replay/frozen exact.

FINAL v192 paired: multiply PERM88.892166252243 minus add88.885009729025=+0.007156523218, actual multiplyTOTAL80.432785906357. Same1162673 params/initialhidden/priors/loss/Q/sampling/SAM/LR/80-14; both best16/stop30 and144665 exact reload/frozen actual3head assembly. Local mechanism-positive singlefold/singleseed; not significance claim, both below185 and154. Keep multiply best/last as localpositive, hold furtherprivate replacements under latest user small/additive direction; keep original154 best. Add negative checkpoints eligible cleanup after fullpair proof, source/config/preds/delta preserved. Pair totalwall649.418650s actualstaggered2GPU/overlap193, trainingadd554.660232/multiply574.887780s, not equalcompute/full5phase.


