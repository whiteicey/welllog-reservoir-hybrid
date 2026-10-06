# v194 Local nine-curve core aggregation

## Focus
Borrow STAR aggregation/redistribution from SOFTS, NeurIPS2024. Only local16-point patches of nine original curves feed a small side branch; keep full206-feature main path and point-level positions. Hypothesis: communicating curves through a small content-selected core improves PERM without large pairwise attention.
Sources: https://arxiv.org/abs/2404.14197 and https://github.com/Secilia-Cxy/SOFTS/blob/main/models/SOFTS.py

## Explored and validation
Prepared/untrained, no GPU launch. Resolve GR/PE/SP/CAL/AC/DEN/CNL/RXO/RT by exact selected_feature_names; do not assume first9 columns, fit no new stats. Each16point patch ->9x64 variable tokens via sharedLinear16to64 plus learned curve identity. Four STAR-style MLPs64to64/64to32/96to64/64to64, mean or featurewise softmax-weighted core across nine curves, same train/inference pooling (not original stochastic pooling). Redistribute core, flatten9x64 and zero-initialized project128; fixed0.1 residual enters stem before local blocks. Fixed same16patch row pairing, masked input and emptypatch/output; no cross-well communication, no full-well statistics. Piecewise interpolated side path may miss thin layers; original fullresolution main path stays intact. This is a local variant, not exactSOFTS reproduction, and differs from full-well FiLM and posterior-head interactions.
Two arms have same parameter tensors/core modules, zero branch output and algebraically same hidden/predictive initial function. Full private128 model random initialization, no185 weights. Retain positive SAM0.05 and 50percent row/well sampling, originalGaussian16/NLL/BCE/jointRPS1/80-14/min10/b64/64steps/lr6e-4. Original seed/fold/Q50/data/decoder unchanged; CUDA backward not guaranteed bitwise trajectory. Full144665 own checkpoint reload/frozen proof and actual fixedPOR151/SW104 assembly required at final. One replacement PERM stage, whole recipe5 inclSOURCE; no extra supervised warmstart.

## Gains
Not trained; no gain. Primary weighted-minus-mean only after both final. Secondary18588.988533638321 has smaller capacity; absolute15489.405564856919/TOTAL80.612475417994 preserved. StrictTOTAL80.8+gain before other folds. Measured parameter count and peak at launch, not speculative speed claim.

## Not worth continuing
No evidence yet. If fair pair fails, close current patch16/core32/weighted setting without patch/core/randompool grid; no universalSOFTS conclusion. Training and pair totalwall must disclose actual concurrency/gaps/data/initial/final eval/warmup, not full5phase cost.

## Latest user: retain proven mechanisms, small-module or additive changes
Keep verified best architecture/recipe/weights. Subsequent trials retain positive mechanisms and change only one small module or add a residual/auxiliary term; broad readout/backbone replacement is held. Current192multiply/193adaptive may complete already-authorized fixed comparisons unchanged, max2. Prepared194 random-private curve-core pair is HELD, no launch under this direction. New additive experiments must count all training phases honestly; do not load trained154 and call another stage free. Source-only initialization with same effective architecture plus one change can replace the existing head phase and remain5 stages. Original splits/Q/scoring/80-14/promotion strict80.8/OOF>82 preserved.
