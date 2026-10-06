# v273 fresh depth-local PERM encoder + direct distribution

## Focus
Channelroute extends271directlysupervisedrepresentation gain by training a fresh96-wide temporal backbone with stem207->96 and3 gateddepthwise9 blocks/dilations1/4/16 pluscategorical96->256readout. All jointly3e-4; no frozennewbackbone/head-only update. Original154prior/atomgate frozen90percentatfinalalpha.1; fullownCE independentofalpha, native271mixedCE/equalwell/RPS1 preserved. Newrandomweights/not271checkpoint. Initialalpha0 then5epochramp.1. No additionalpretrainingphase.

## Explored and validation
Prepared / independent secondroute beside272fullwarmexpertencoding. Original206+relative/no newcurves, maskedper-tokenLayerNorm/no crosswellpool; no dropout. OriginalQ50/data/seed/64steps/80-14/decoder256bins800actions. Allrawfeaturebackboneparams trainable; activecount/gradients/memory observedatlaunch. Initialprior discretizationdifference recorded, notgain. Atomicbest/last includesalpha/RNG; RAMreplay notdiskreload. Capacity/architecture/initialization differfrom271, no clean depth-only causalclaim.

## Gains
Unproved. Compare completed271repeat PERM89.43939135949043 and currentdefault15489.40556485691894. OnePERMaddedstage/6alone/7withSW252, nochainon271. Keepoldbest/default5/delivery. Strict5TOTAL80.8+meaningfulgain beforeotherfolds/fullOOF82unmet.

## Not worth continuing
Previous257shallowrawTCN fed zero massrepair and gave no depthgain;273 usesindependent directcategoricalsupervision/newdepthencoder after271positive. SW263coldspecialists failed, notproofPERM success. Largerwidth/contextcanoverfit/corruptthinlayers. Ifbothtraining andofficialevidence do not beat retained271, close thiskernel/dilation design; no automaticwidth/kernel/LR scan.

## Timing
Record individual/overlapwall/peak, includesstartup/data/initialeval/finalrestore, excludesprep/CPUassembly; no unmeasuredserialspeedup. Max2GPU guard andcheckactive272beforelaunch.

V273 liveprocesscommand verified PID88956/epoch3/best89.42320118267263; trainable132064, bothfeaturebackbone anddirectreadout updated. Max2GPU/80-14/onePERMphase; oldSOURCE/154priorfrozen/no271trainedweights/nootherfolds. This supersedes prepared/heldtext.

V273 final PERM89.434129958417, gain+0.028565101498, actualTOTAL80.622473203518; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop8/22, trainingseconds232.049750/peakMiB789.281250. No otherfolds.

V273 FINAL PERM89.434129958417, actualTOTAL80.622473203518, vsretained271-0.005261401073; best/stop8/22, actual144665/RAM/frozenverified/no independentdiskreload. Positivelearningfromprior but backbonevariant doesnotbeat cheaperretainedrawencoder; no fair isolatedcapacity claim, no adoption/no newwidth-orLRgrid. Individual232.050s/peak789.281MiB, launchwall238.655s inclstartup/data/initialeval/finalrestore/excludesprep, overlapexplicit/no measuredserialspeedup. Source/config/score/predictions kept; unpromisingredundantweights removablewithSHAmanifest, original154/271/active274/275 retained.
