# v38 Direct official action-value readouts

Fixed original fifth TOTAL 80.336513436816 -> 80.336513436816, delta +0.000000000000.

|Target|Context|Best after update|Selected with parent fallback|Best/stop epoch|
|---|---|---:|---:|---|
|POR|point|77.507769906|77.551936802|0/18|
|POR|local5|77.508897729|77.551936802|0/18|
|SW|point|73.614883759|73.663082797|0/18|
|SW|local5|73.605314096|73.663082797|0/18|

## Scope, execution correction and verification
Same initialized parameters, original input/hidden features/actions/score; identity vs five-point masked average at the hidden readout. All parent predictors frozen. New heads regress official action utility with uniform squared error, not a likelihood or soft-action objective.
The first POR run incorrectly inverted log1p with exp. It was stopped after epoch4; its scores are excluded and no weights were reused. Corrected POR restarted from the original parent, uses expm1, and checked 2048 training labels against the official kernel (maximum difference below1e-4). SW log/exp was correct and finished unchanged.
Selected-head full disk reconstruction: []. Fixed three-column predictions scored on144665 original fifth rows.

## Mechanism contrasts and scope of conclusions
POR local-point selected channel difference +0.000000000000; post-update-only difference +0.001127823503.
SW local-point selected channel difference +0.000000000000; post-update-only difference -0.009569662511.
Adopted: []
Epoch0 fallback is not training improvement. Positive comparisons are development-fold evidence, not independent validation. Both valid targets failed to improve the parent; close this direct-value readout setting without an additional width/context/loss sweep.

## Time and outstanding objective
Valid training interval wall time 354.385796 seconds; per-target times, peak memory and invalid-attempt overhead are in timing.json.
Full original5foldOOF>82 is unachieved. No other folds, old tasks or delivery replacements were started.
