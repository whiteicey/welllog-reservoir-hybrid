# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# Fixed independent head mixture weights

## Focus and control
User requests different optimal weights for each head. Reuse V364 original model plus complete new seed20261010 native probability/official utility grid0:0.01:1. For each POR/PERM/SW independently choose the maximum official whole-target score, use one fixed ratio for every row in that channel, then actualassemble3columns and score original144665fifth rows. Whole-target score selects, not CONT-specific score; in exact ties prefer greater original-model weight. Compare samepair common81:19 TOTAL80.724892908977 and originalalone80.706129723238.

## Explored and verification
Native endpoints and0.5, immutable weights/modelSHA/population/coverage and independent diskreload were verified in V363/V364. CPU-only reuse of their compressed action-index grids, no repeated GPU inference or training. Pending actual chosen ratios, assembledprediction TOTAL and DEFAULT/CONT diagnostics.

## Gain
Measured: POR original0.93/new0.07 score78.06314784005319; PERM0.49/0.51 score89.56546548645483; SW0.71/0.29 score74.1951684727142. ActualTOTAL80.73516623772511, originaldelta+0.02903651448704636/common81:19delta+0.010273328748354515. Same selectedratios CONT POR41.733030179719194 (-0.003610112276), PERM74.01018085296752 (+0.000543769770), SW17.72871820182738 (-0.250680948219). Wholegain does not preserveSWcontinuous subgroup. Validation-

lSHA/population/coverage and independent diskreload were verified in V363/V364. CPU-only reuse of their compressed action-index grids, no repeated GPU inference or training. Pending actual chosen ratios, assembledprediction TOTAL and DEFAULT/CONT diagnostics.

## Gain
Measured: POR original0.93/new0.07 score78.06314784005319; PERM0.49/0.51 score89.56546548645483; SW0.71/0.29 score74.1951684727142. ActualTOTAL80.73516623772511, originaldelta+0.02903651448704636/common81:19delta+0.010273328748354515. Same selectedratios CONT POR41.733030179719194 (-0.003610112276), PERM74.01018085296752 (+0.000543769770), SW17.72871820182738 (-0.250680948219). Wholegain does not preserveSWcontinuous subgroup. Validation-selected per-head maximum within0.01grid only, not independent generalization/fullOOF or robust model gain. Two8stage models/16prior successful phases total; no additional learnedstage/no adoption as single8stage model. Default and originalrecipes unchanged, no otherfolds.

## Not worth extending
No per-row weight selection, learnedrouter, CONT-head oracle, finer grid, newmodels or training in this question. User authorization covers fixed per-head ratios only. Report all3 scores and actualTOTAL, including any continuous subgroup decline despite totalgain.

## Timing and pending
CPU only0.742s scoring/selection/strata and9.529s process/import; existing training uninterrupted. Read compact cachedindices, selectratios/assemble/officialscore/strata; GPU0 extra. Done FINAL.json/SELECTED_WEIGHTS.csv/actualpred artifact. Continue existing five-seed batch, no additional V365 training/grid.

Terminal userauthorized per-head fixed ratios {"POR": {"original_weight": 0.93, "new_seed_weight": 0.06999999999999995, "score": 78.06314784005319, "selected_grid_index": 93}, "PERM": {"original_weight": 0.49, "new_seed_weight": 0.51, "score": 89.56546548645483, "selected_grid_index": 49}, "SW": {"original_weight": 0.71, "new_seed_weight": 0.29000000000000004, "score": 74.1951684727142, "selected_grid_index": 71}}. ActualTOTAL80.735166237725/originaldelta+0.029036514487/common81:19delta+0.010273328748. CONTsame selectedratios {"POR": 41.733030179719194, "PERM": 74.01018085296752, "SW": 17.72871820182738}; CONTdeltaoriginal {"POR": -0.003610112275950428, "PERM": 0.0005437697704593347, "SW": -0.2506809482192516}. CPU0.742s/nativeverifiedcachedgrids/noextraGPU/training. Same-fifth tuned maximum0.01grid/fixedperchannel ratios, notroworacle/CONToracle/generalization/OOF. Two8stage models/16prior phases/noadoption as one8stage model/nootherfolds. Existingfive-seedtraining continues.


