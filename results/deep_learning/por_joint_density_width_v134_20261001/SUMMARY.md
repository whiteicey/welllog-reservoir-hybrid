# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v134 POR positive density-head capacity extension
Focus: widen positive v131 point-conditional location/logweight head from64 to128. Conditional nonlinear density readout gave real historicalPOR+0.177077 and point-control+0.167638; prior failed wholebody/FFN expansion is a different scope. Default gate remains frozen.
Explored/verified: implemented one wide128 arm:319input/128stem/norm/summed33coefficient pointdepth/256GLU/32zerooutputs. Trainablehead82720 versus33184 in v131, +49536params; identicalfullinitialv113 prediction, internalrandomfeatures differ (not same initialization matrices/capacity). Same LR3e-4/ownpropermixedStudent3-CauchyNLL/Q/labels, newlocationbound1fixedreferencewidth/logweightbound1, originalCDF/actions/argmax unchanged. Frozen v113 predictor, no trainedv131/v133 head warmstart, no new supervision or othertargetupdates.
Gain: untrained. Primary compare retainedv131POR77.729014112744, TOTAL80.543755697579. Incremental widthgain must beat this control, not just initiallearning. HistoricalPOR>77.801936801885 target still pending, strictTOTAL80.8 before remainingfolds.
Not worth continuing: defaultgateopening v133 currently weak, close onlyafterfinalevidence; oldnegative tail/linearcoeff/widthscope/utility/thawing remainclosed. No additionalwidth/LR candidates in thisbatch unless thisextension yields measuredgain.
Pending: explicitly launch onlyonce curren

to128. Conditional nonlinear density readout gave real historicalPOR+0.177077 and point-control+0.167638; prior failed wholebody/FFN expansion is a different scope. Default gate remains frozen.
Explored/verified: implemented one wide128 arm:319input/128stem/norm/summed33coefficient pointdepth/256GLU/32zerooutputs. Trainablehead82720 versus33184 in v131, +49536params; identicalfullinitialv113 prediction, internalrandomfeatures differ (not same initialization matrices/capacity). Same LR3e-4/ownpropermixedStudent3-CauchyNLL/Q/labels, newlocationbound1fixedreferencewidth/logweightbound1, originalCDF/actions/argmax unchanged. Frozen v113 predictor, no trainedv131/v133 head warmstart, no new supervision or othertargetupdates.
Gain: untrained. Primary compare retainedv131POR77.729014112744, TOTAL80.543755697579. Incremental widthgain must beat this control, not just initiallearning. HistoricalPOR>77.801936801885 target still pending, strictTOTAL80.8 before remainingfolds.
Not worth continuing: defaultgateopening v133 currently weak, close onlyafterfinalevidence; oldnegative tail/linearcoeff/widthscope/utility/thawing remainclosed. No additionalwidth/LR candidates in thisbatch unless thisextension yields measuredgain.
Pending: explicitly launch onlyonce currentPORslotfree, beside backbonev132r1(max2). Original9curves206+allowed63side/relative/trainonlystats/Q50/fixedfifth/seed/normalGPU,80-18/min10/b64/64steps/cos80*64. Atomiclast/explicitresume/epochstop/best144665replay/frozencheck; actual fixedPERMv100/SWv104 assembly. Five dependencyphases source/v113/v134/PERM/SW. Source/config ofpasttrials immutable, deliveriesunchanged. Record realwall/overlap/memory, no automaticqueue.
Optional: if nonlinear densityhead expansion helps, retain evidencefor next boundedPORstep towardhistorical+0.25 while alternatebackbone progresses.

V134 wide128: restored POR=77.869090371913; delta vs v114=+0.321020877020, vs historical=+0.317153570028; actual fifth TOTAL=80.585778575329;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=34/52, seconds=363.200, peakMiB=922.284.

V134 VERIFIED POR milestone:77.869090371913 best34/stop52, historicalgain+0.317153570028 (>0.25) andwidth64gain+0.140076259170. Actualthree-channelTOTAL80.585778575329, five totaldependencyphases; strictTOTAL80.8 stillfalse. All144665 replay/frozen exact, keeppositivebest/last andcurrentrecipe. This meets userone-channel+0.25 milestone, NOT fullgoal; firstroute now independently targets PERM readout, fixedPORv134/SWv104. OriginalfullOOF>82 unmet, no otherfolds/delivery edits.


