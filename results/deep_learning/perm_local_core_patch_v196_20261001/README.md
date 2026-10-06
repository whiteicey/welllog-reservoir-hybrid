# v196 Small additive local curve-core patch on the positive readout

## Focus
Retain proven154 Gaussian location/logweight head,128point trunk and NLL/BCE/jointRPS1. Add one small residual to its stem, using original9curves in local16point patches. SOURCE backbone/gates/scales and fullresolution point readout remain. This follows latest user instruction and repurposes SOFTS inspiration as a small head patch; held194 private-backbone trial is not launched.
Source: https://arxiv.org/abs/2404.14197 and https://github.com/Secilia-Cxy/SOFTS/blob/main/models/SOFTS.py

## Explored and validation
Prepared/untrained. Exact raw curve feature indices from original selected_feature_names, no first9 assumption/new statistics. SharedLinear16to16 and9curve identity; fourSTAR-style MLPs16to16/16to8/24to16/16to16. Mean or featurewise softmax-weighted core, same train/eval, no stochasticpooling. Flatten9x16, zero project128, fixed0.1 residual added to original head stem. Common old parameter tensors/zero patch/algebraically identical initial hidden/output, same capacity for paired arms, actual count/initial score at launch. WholeSOURCE parameters freeze, no private-model random replacement. Preserve main319hidden+amp input/point processing/originalGaussian output/location and logweight bounds. Small local side interpolation can miss narrow beds but originalpoint branch intact; validmask inputs/empty patch/output, no well crossing/fullwellbank prediction.
Same Source initialization/seed/Q50/train64/fixed5/NLL/BCE/jointRPS1/lr3e-4/AdamW/noSAM/80-14/min10/b64/64steps/decoder. AdditiveCRPS195 is not included: one change only. No trained154head warmstart, which would require honest extra stage; keep original154 weights untouched. One replacementPERM phase, whole5 inclSOURCE/POR113/POR151/SW104. Atomic states, all144665 own official selection/reload/frozen and actual3head assembly at final, no fabricatedTOTAL.

## Gains
Not trained/no gain. Primary weighted-minus-mean after both final. Secondary samebudget165adamw originalhead89.344760174753 has smallercapacity; historical15489.405564856919/TOTAL80.612475417994 remains absolute best with patience18, not samebudget control. Initial score/learned gain/local comparison/absolute gap reported separately; singlefold not completeOOF. Strict80.8+gain before other folds.

## Not worth continuing
No evidence yet. Do not grid patch/core/width if pair fails. Do not infer SOFTS universally useless. No extra privatebackbone/family replacement; no failedCRPS/IQF/wavelet stacking. Actualtime/peak/concurrency/interarmgap/data/initial/final eval disclosure, not full5phase/equalcompute cost.

Latest user explicitly raises total training stage cap to8, superseding5. Prioritize freezing learned positive functions with small additive repairs. New197 warm154+one patch honestly6stages, ownPERM loss/fixedscore/max2/80-14. Existing sources/deliveries unchanged. Prepared196 source-retrained variant HELD in favor of197 preserving learned154 weights; no automatic queue/remainingfolds.
