# v258 PERM input-only whole-well conditioned probability readout

## Focus
Frozen154 feature128 plus small wholewell branch. Original input-only well bank128samples x206 train-fitted features; mean/std acrosssamples gives412 summary ->32GELU. Concat frozen128hidden with repeated32condition then160->32->257mass/background readout. Reference arm uses existing train-only reference_inputs mean/std, well arm uses currentwell input curve summary. Same26849storedparams/init/loss; referenceconstant changes effectiveinput variability, not same functional expressivecapacity claim.

## Explored and validation
Prepared/held while257live. Existing64trainingwell-fitted normalization and reference reused; heldoutwell INPUT summaries permitted per original inference context, no heldoutlabel stats/IDs/onehot/neighborwell labels. Feature bank excludes forbiddencurves through original206contract. Small branch and new readout train together ONE PERMphase, full154/gatefixed. Retain248 effective normalizedowncontinuousNLL+jointRPS1/background. No trained248/257weights, no source prediction blending. Samezero finalprojection/.001background; initialscore separatelyreported. Originalfold5/Q50/256bins/800actions/80-14/noSAM preserved. Final RAMrestore/frozen/actual144665 composition required, not diskreload claim.

## Gain
Unproved. Predefined well-minus-reference plus absolute retained248PERM89.413465595388. New26849=wellbranch13216+readout13633. Sixalone/sevenwith retainedSW252,total<=8. Wholewell input branch is explicit derived feature variant, not restored originalexcluded wellmean feature or forbiddencurve restoration.

## Not worth continuing
257 localcrosscurve branch remains below retained so far; this tests larger context scope rather than depth/dilation scanning. Previous SW wholewell conditioning/frozen repairs not automatically positive forPERM. If wellbelowreference and bothbelowretained, close this mean/std128sample summary configuration; no moment/order/summarywidth grid.

## Timing
Record individual/concurrentprocesscreation-to-lastreportwallclock/peakMiB; startup/model/data/initialevalincluded, codeprep/CPUassemblyexcluded; no serialspeedup claim.

258 actual launch referencePID66400/wellPID41832, bothsameinitial/26849params. Epoch2reference89.408134584882/well89.409475154060, earlyrelative+0.001340569178 butbelowretained248; snapshot not proved finalgain. Bothgradienttrain/noerrors/OOM, peaks790.477/790.296MiB. Existinginput-only128x206bank/trainreference, no labelstats/wellIDs or heldout fitting. Original154frozen/onePERMaddition/max2/nootherfolds.

V258 final PERM89.411293594790, gain+0.005728737871, actualTOTAL80.614480476249; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds186.966599/peakMiB790.476562. No otherfolds.

V258 final PERM89.411012124353, gain+0.005447267434, actualTOTAL80.614381961596; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds187.184188/peakMiB790.295898. No otherfolds.

V258 closed: reference PERM 89.411293594790, well 89.411012124353, delta -0.000281470438; natural patience14 stop19 both. Actual144665 composition/RAMbest replay/frozen checks complete, no independent diskreload. Pair wall 193.287s, includes startup/data/initialeval/finalrestore, excludes codeprep. Wholewell mean/std no gain; retain248/default5/optional7, no otherfolds.
V259 launched CE/RNC joint continuous SW backbone/readout, original104 atom fixed, one addedSWphase, max2GPU/80-14. Gain unproved; study results/deep_learning/sw_rank_representation_v259_20261001.
