# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW shared frozen source-feature view
Focus: preserve104 prediction while supplying another already-trained inputrepresentation to one smallmasspatch. Existing sharedSOURCE used byPOR/PERM has two128hidden experts; concatenate with128SW hidden through sameLayerNorm256/MLP384->32->256 patch. Repeatcontrol substitutes256repeatedSW hidden; SOURCEforward still computed in botharms to matchcost.
Explored: prepared/untrained/nolaunch/noqueue. Load originalSOURCE master state and freeze it; all104 frozen. Sourceoutputs are discarded, no predictions/votes/teacher/pseudo-targets. Only input-hiddenfeatures enternewpatch; ownSW binlikelihood/BCE, utility0, SAM0.1/LR1e-4/80-14, originalfold5/Q50/bins/actions/score. Same21280activeparameters/sameinitialtensors/function, compare parametercount noteffectivefunctionrank. Extraencoder inference/time/memory must be measured.
Gain: unproved; baseline104SW74.12175282815758; currentpositive21174.14762262262707. No viewbenefit claim untilsamecapacityrepeat/source comparisoncomplete. Sourcewasalready trainedfororiginal5stage recipe and countedonce, no additionalSOURCEtrainingstage; optionaloneSWstage/sixoverall. Neverstackother massrepair, default5; no remainingfolds/fullOOFclaim.
Not worth continuing: finalfeatureadapter210 hadnogain, not a reason to reject complementaryfrozeninputfeatures. Original SOURCE uses9curves/206 trainfittedinputs; no exte

# SW shared frozen source-feature view
Focus: preserve104 prediction while supplying another already-trained inputrepresentation to one smallmasspatch. Existing sharedSOURCE used byPOR/PERM has two128hidden experts; concatenate with128SW hidden through sameLayerNorm256/MLP384->32->256 patch. Repeatcontrol substitutes256repeatedSW hidden; SOURCEforward still computed in botharms to matchcost.
Explored: prepared/untrained/nolaunch/noqueue. Load originalSOURCE master state and freeze it; all104 frozen. Sourceoutputs are discarded, no predictions/votes/teacher/pseudo-targets. Only input-hiddenfeatures enternewpatch; ownSW binlikelihood/BCE, utility0, SAM0.1/LR1e-4/80-14, originalfold5/Q50/bins/actions/score. Same21280activeparameters/sameinitialtensors/function, compare parametercount noteffectivefunctionrank. Extraencoder inference/time/memory must be measured.
Gain: unproved; baseline104SW74.12175282815758; currentpositive21174.14762262262707. No viewbenefit claim untilsamecapacityrepeat/source comparisoncomplete. Sourcewasalready trainedfororiginal5stage recipe and countedonce, no additionalSOURCEtrainingstage; optionaloneSWstage/sixoverall. Neverstackother massrepair, default5; no remainingfolds/fullOOFclaim.
Not worth continuing: finalfeatureadapter210 hadnogain, not a reason to reject complementaryfrozeninputfeatures. Original SOURCE uses9curves/206 trainfittedinputs; no externalB/D/ABD branches/releasedweights. If no matched benefit, closethisview, no encodergrid.
Timing: expectgreaterinferencecost fromtwo SOURCEexperts; no speedclaim before realtimings. Record single/sameworkbatchwall including sourceinference/loading/validation/warmup/overlap andmemory. Atomicstates/frozen/all144665restore/replay/actualassembly. No testlabels or historicaltargetvalues ininput.

V215 failed during initial evaluator before initialscore/epochs: SOURCE cosine patch requires original256window, full1024 unsupported. Failure/source preserved, not negativeefficacy. V216 fresh source uses only central256 original SOURCE window around sameanchor; fills256hidden side into current1024tensor but loss/evaluation select central128 only. SW original1024context/bins/actions unchanged. Sourcecomputed inbotharms. No additionalsource training/queue.



