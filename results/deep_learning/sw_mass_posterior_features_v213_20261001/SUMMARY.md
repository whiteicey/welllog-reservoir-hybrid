# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW explicit predictive-distribution view
Focus: read the frozen predictive distribution itself, following the positive probabilitymass readout. Add256dimension sidefeatures to128oldhidden; same LayerNorm256/Linear384->32->256 patch in both arms. Posterior uses logprior massclipped-20; repeatcontrol duplicates128hidden. Exact21280activeparameters/same initial tensors and zero-output function; comparison matches parameters, not effectivefunctionrank.
Explored: prepared/untrained. Both old104 fullyfrozen; original256bins/800actions/scoring, bound0.5, owncategoricalNLL/BCE utility0 after211positivefinal, SAM0.1/LR1e-4/80-14. No209/211weights loaded. View is input-model predictive distribution, not labels/teacher supervision; sourcepriormass computed withoutgrad.
Gain: not proved; original10474.12175282815758, positive21174.14762262262707. Primary attribution within pairposterior-repeat; architecturecapacity versus211 differs, so do not assign anytotalgain solelyto distributionview until paircomplete. OptionaloneSWstage/sixoverall, modes and211 are alternatives neverstack, default5. No remainingfolds/fullOOFclaim.
Not worth continuing:209utility1 below211utility0 by0.005438135132;212boundexpansion adds only0.002728513187 vs209, no broadboundgrid. Sourcebone214 independentrank8 prepared. If posterior failsfaircomparison, no view sweep.
Timing: record single/samebatchwall incldata/e

# SW explicit predictive-distribution view
Focus: read the frozen predictive distribution itself, following the positive probabilitymass readout. Add256dimension sidefeatures to128oldhidden; same LayerNorm256/Linear384->32->256 patch in both arms. Posterior uses logprior massclipped-20; repeatcontrol duplicates128hidden. Exact21280activeparameters/same initial tensors and zero-output function; comparison matches parameters, not effectivefunctionrank.
Explored: prepared/untrained. Both old104 fullyfrozen; original256bins/800actions/scoring, bound0.5, owncategoricalNLL/BCE utility0 after211positivefinal, SAM0.1/LR1e-4/80-14. No209/211weights loaded. View is input-model predictive distribution, not labels/teacher supervision; sourcepriormass computed withoutgrad.
Gain: not proved; original10474.12175282815758, positive21174.14762262262707. Primary attribution within pairposterior-repeat; architecturecapacity versus211 differs, so do not assign anytotalgain solelyto distributionview until paircomplete. OptionaloneSWstage/sixoverall, modes and211 are alternatives neverstack, default5. No remainingfolds/fullOOFclaim.
Not worth continuing:209utility1 below211utility0 by0.005438135132;212boundexpansion adds only0.002728513187 vs209, no broadboundgrid. Sourcebone214 independentrank8 prepared. If posterior failsfaircomparison, no view sweep.
Timing: record single/samebatchwall incldata/eval/startup/overlap andmemory. Atomicstates/frozen/144665restoredreplay andactualthreeheadassembly. No pretrained externalmodels/testlabels or rowwise selection.

V213 posterior final SW74.134546961565, actualTOTAL80.616953364687; learned gain+0.012794271796, vs211-0.013075661062. Six optionalphases, default5; no fair viewgainclaim until repeat/posterior paircomplete.

V213 repeat final SW74.137964012421, actualTOTAL80.618149332486; learned gain+0.016211322652, vs211-0.009658610206. Six optionalphases, default5; no fair viewgainclaim until repeat/posterior paircomplete.

V213 matchedposterior-minusrepeat SW-0.00341705085586, posterior74.134546961565 versusrepeat74.137964012421; explicit posteriorfeatures didnothelp thispair, bothbelow21174.147622622627. Close this view/no broadergrid. Repeat retainedas213control and reusedfunctionally for216, no redundant216repeat GPUrun; Source inferencetime not matched, see216 CONTROL_REUSE.md. Current216source continues asonlylivejob (cap2), default5/fullgoal>82 unchanged.



