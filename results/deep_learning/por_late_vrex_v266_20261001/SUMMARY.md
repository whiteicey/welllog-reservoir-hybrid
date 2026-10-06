# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# V266 POR late representation and readout with risk variance

## Focus
Test cross-training-well continuous riskvariance regularization while jointly adjusting copied actualdec0+combine blocks and retained151 readout. Source [V-REx paper](https://proceedings.mlr.press/v139/krueger21a.html); local adaptation, not full reproduction or causal guarantee. Crosswellshortcut hypothesis unproved.

## Explored and validation
Prepared/nottrained. From151 fixedparent,113 prior/gate/oldheads/routes fullyfrozen. Copy each expert terminaldec0/combine from151 and recompute hidden from detached actualinputs; jointtrain copiedexistingblocks1e-5 and retained151contexthead3e-5. Latecombine dropout eval sameinitialfunction, originalsourceeval/frozen. Newcopies create inferenceparams; record actualcount atlaunch. Train originalnativeT3/CauchyNLL/BCE+jointRPS1 unchanged. Extracontinuous finite-bin CE computes pertrainingwell Q-normalized means, >=8continuousrows/atleast2groups. Penalty populationvariance /stopgrad(mean squared), fixed.1 ramp5epochs, controlbeta0 computes samediagnostics. Original64rowanchors/64steps/80-14 unchanged botharms; differsresearch suggested8x8balanced sampling/fixedpilot scale. No pilot/preflight or newphase, no crossstepstats needed. TrainwellIDs only group loss, no inferencefeature/heldoutfit. Data9curves/206+existing63trainfittedamplitude/Q50/originalfold/scoring unchan

1normalizedvariance if officialpairednegative, no beta/wellweightgrid. Two copiedblocks+readout differs oldallbody SWupdates, no universal backbone claim. Need actualfrozen checks and preserveoriginalparent/delivery.

## Timing
Individual and actualconcurrentwall/peak inclstartup/data/initialeval/finalrestore, excludesprep/CPUassembly. Max2GPU, helduntil264terminal. No serialspeedup claim.

V265 both failed beforefirst optimizer step: cdf.diff(-1) interpreted as negativeorder, requires dim=-1. Initialfunction equals151POR77.926380760724 both, trainable265632/newcopies34688. No trained result. NewimmutableV266 fixesdiff(dim=-1) only, fresh151/copyinit/resetoptimizers/samehypothesis. Failure source/log/checkpoint evidence tracked, no mechanismconclusion.

V266 liveverified PIDs64340/13884 epoch3. ExactinitialPOR77.926380760724 both; currentcontrol77.539310605749/vrex77.531868980095, beststillinitial/no learnedgain yet. Finite nativeNLL (negative is density logscore, not failure), V-RExfinite/28traininggroups,265632updates/34688newcopiedparams, peak925.151MiB. Dropoutofallcopiedlateblocks eval/gradients enabled, old113frozen; max2/80-14. V265failed initialweights deleted withSHAmanifest; original151/264positivepreserved. Currentoptional7microTOTAL80.625533220240, notmaterial/no808/nootherfolds. Atlas1032experiments/197deadends/219unresolved, unresolvedancestryexplicit.

V266 control finalPOR77.926380760724, actualTOTAL80.612475417994; gainvs151+0.000000000000. Sixphases alone, onePORaddition/no otherfolds; restoredRAM replay, not independent diskreload.

V266 vrex finalPOR77.926380760724, actualTOTAL80.612475417994; gainvs151+0.000000000000. Sixphases alone, onePORaddition/no otherfolds; restoredRAM replay, not independent diskreload.

V266 FINAL control/vrex POR77.926380760724 both/best0/natural14stop, learnedgain0; actual144665assembly/RAMbest/frozenexact/no diskreload. SoloTOTAL80.612475417994 is olddefault5 function replay, not newgain. Individual246.521914/248.109755s/925.151MiB, concurrentwall253.429s inclstartup/data/initialeval/finalrestore/excludesprep.265632trainable/34688newcopiedlateparams; close thiscoadapt/variance design, no beta/encoderwidthscan. V267 launched ownPERM154context+mass coadapt andsame264wellmean mechanism, p.25/strength.1 training-onlystyle versus matchedcontrol, onePERMaddition/max2/80-14. Gainunproved, currentoptional7 80.625533220240/default5/delivery unchanged/nootherfolds.

No-gain checkpoints deleted afteractualcomposition; CHECKPOINT_DELETION preservesSHA/path/reason. Original151 and currentpositives/active267 retained.


