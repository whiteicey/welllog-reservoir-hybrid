# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW additive bidirectional token memory

Focus: preserve positive104 final attentionblock; add small normalized BiGRU16perdirection over its64pooled token features and zero-output32->96 projection,bound0.1pointRMS,14304params. Final FiLM/detail/readout remain. Input-only forward/backward sequence memory,not predictions/labels.

Explored: prepared/untrained,ownoriginalcontinuousNLL/BCE+utility1,SAM0.1/LR1e-4,originalfold5/Q50/80-14. Originalblocks eval/noDropKey; new GRU trainingmode only for backward and explicitfloat32. Invalid token inputs zeroed/output masked; padding still traversed by GRU,not packed/label context.

Gain: unproved,reference104SW74.121752828158. Zero last projection preserves initialfunction; oneSWextra stage/six total,no225/229/mass repair weights loaded.

Not worth continuing: isolated oldQKV/localfilter/wholewellcondition refinements failed; localamplitude229 declining. This tests recurrent sequence dynamics instead of more pointwise calibration. No GRU-size/window/layer grid planned.

Timing includes loading/warmup/eval,actualoverlap/peak. Restored RAM replay/frozen/actual144665 assembly required. No otherfolds until TOTAL>80.8+gain; default5 retained.

231 failedfirstbackward because intended GRU trainingmode switch was missing,no epoch efficacy score;232freshsource explicitlysets recurrent.train(mode) and has completed realepochs.233PERM width64 and23

# SW additive bidirectional token memory

Focus: preserve positive104 final attentionblock; add small normalized BiGRU16perdirection over its64pooled token features and zero-output32->96 projection,bound0.1pointRMS,14304params. Final FiLM/detail/readout remain. Input-only forward/backward sequence memory,not predictions/labels.

Explored: prepared/untrained,ownoriginalcontinuousNLL/BCE+utility1,SAM0.1/LR1e-4,originalfold5/Q50/80-14. Originalblocks eval/noDropKey; new GRU trainingmode only for backward and explicitfloat32. Invalid token inputs zeroed/output masked; padding still traversed by GRU,not packed/label context.

Gain: unproved,reference104SW74.121752828158. Zero last projection preserves initialfunction; oneSWextra stage/six total,no225/229/mass repair weights loaded.

Not worth continuing: isolated oldQKV/localfilter/wholewellcondition refinements failed; localamplitude229 declining. This tests recurrent sequence dynamics instead of more pointwise calibration. No GRU-size/window/layer grid planned.

Timing includes loading/warmup/eval,actualoverlap/peak. Restored RAM replay/frozen/actual144665 assembly required. No otherfolds until TOTAL>80.8+gain; default5 retained.

231 failedfirstbackward because intended GRU trainingmode switch was missing,no epoch efficacy score;232freshsource explicitlysets recurrent.train(mode) and has completed realepochs.233PERM width64 and232SW memory are sole GPU jobs. Current bestactualoptional7TOTAL80.624369397272,fullOOF82 stillunmet,no otherfolds.


