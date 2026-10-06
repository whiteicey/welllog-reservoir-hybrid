# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# PERM v272 full feature encoders + direct distribution branch

## Focus
Updated user goal mandates backbone changes together with readout. New full copies of both original SOURCE expert encoders, including input/calibration/point/stem/all downsample/encoder/bottleneck/decoder/combine weights, update jointly1e-5 with new319->128->256 readout3e-4. Original154 complete prior/atomgate untouched. Copies initializedfromoriginal SOURCE parts inside154, no271 trainedweights/no existingrepairphasechain. Copied oldmain/auxheads frozen and not used in finalprediction; prehookcaptures hidden WITHOUTdetach. evaldropout retained but gradient enabled. Directcomponent CE1 plus effective271 halfrow/halfequalwell mixtureCE andRPS1; allownPERMlabels. No teacher/objectivealignment/oldvote.

## Explored and validation
Prepared held271GPUslots. Original256window/206+63side/relative/64train16val/Q50/seed/decoder/80-14/b64/64steps. Nonzero newreadout gives backbone first-stepgradient independently ofalpha, same fixed5epoch alpha0->.1. Original154probabilitydiscretization measured separately. Original and copied parameter signatures tracked, schedulealpha intentionally mutable/saved; own best state includes it. Allvalid144665 actualcombine afterRAMrestoredreplay; no diskreload claim. No numericalpreflight/independentreview. Actualactiveparameters/memory fromruntime, not assumption of noOOM.

## Gains


a0->.1. Original154probabilitydiscretization measured separately. Original and copied parameter signatures tracked, schedulealpha intentionally mutable/saved; own best state includes it. Allvalid144665 actualcombine afterRAMrestoredreplay; no diskreload claim. No numericalpreflight/independentreview. Actualactiveparameters/memory fromruntime, not assumption of noOOM.

## Gains
Unproved. Absolutecompare retained268PERM89.415840060079 and completed271score onceavailable; not samecapacity/initialization ablation with271 rawMLP. Any architecturepluswarmstart effect cannot be separated.6alone/7withretainedSW252; onejointPERMaddedstage replaces optionalrepair, neverappendto271weights. StrictTOTAL>80.8+meaningfulgain beforeotherfolds, fullOOF82 unmet/default5/deliveriespreserved.

## Not worth continuing
270 fullnative head/latecopies producedzero gain, so do not repeat same update/loss. This broadensrepresentation update and uses independentdirectdensitysupervision observedpromisingin271 snapshots, not proofcause. Highriskofunlearning, trainonlydropoutoff overfit; priorkept90percent reducesdamagebutdoesnotguarantee. Ifbestremainspriororloseagainstretainedcontrols, closethiswholeencoder/directbranch configuration withoutLR/alpha/width grid.

## Timing
Record individual/actualoverlap pairwall/peak, includestartup/data/initial/finalrestore, exclude preparation/CPUassembly. Max2GPU upperbound; admit onlyafteractualotherprocess slotsfree, no launchqueue. No unmeasuredserialspeedup.

V272 liveprocesscommand verified PID70928/epoch11/best89.4207751145966; trainable1346678, bothfeaturebackbone anddirectreadout updated. Max2GPU/80-14/onePERMphase; oldSOURCE/154priorfrozen/no271trainedweights/nootherfolds. This supersedes prepared/heldtext.

V272 final PERM89.424282481650, gain+0.018717624731, actualTOTAL80.619026586650; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop17/31, trainingseconds496.498249/peakMiB971.438965. No otherfolds.

V272 FINAL PERM89.424282481650, actualTOTAL80.619026586650, vsretained271-0.015108877840; best/stop17/31, actual144665/RAM/frozenverified/no independentdiskreload. Positivelearningfromprior but backbonevariant doesnotbeat cheaperretainedrawencoder; no fair isolatedcapacity claim, no adoption/no newwidth-orLRgrid. Individual496.498s/peak971.439MiB, launchwall502.894s inclstartup/data/initialeval/finalrestore/excludesprep, overlapexplicit/no measuredserialspeedup. Source/config/score/predictions kept; unpromisingredundantweights removablewithSHAmanifest, original154/271/active274/275 retained.


