# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# PERM v276 numeric PLE / scalar-repeat joint representation comparison

## Focus
NineoriginalGR/PE/SP/CAL/AC/DEN/CNL/RXO/RT curves gain boundedpiecewise-linear coordinates fromtrain-input quantiles. Maximum8segments, merge duplicate/nearby knots tau=max(1e-6,range*1e-6), nearconstantcurve no addedcoordinates. Variableconcatenateddimension avoids paddedzero/dummychannels. Sameknots/dimension/stemparameters for scalar_repeat control andPLE; controlrepeatnormalizedscalarclippedsameendpoints, candidateordinalcontinuousintervalcoordinates. Same registeredcapacity, effective functioncapacity/rank differs. Original206+relative/63allowedside remainsparent154prior and206maininput; missingflags andoutside-range rawmagnitudes kept. Knotsfit existingalreadyimputedtrainingcurvevalues withno label/Q/targetstate selection; notguessing normalizedmissingflag semantics. Bothnew48widthencoder+256head jointly3e-4, effective274 fixedalpha.25/directcomponentCE/equalwellNLL/RPS1 preserved.

## Explored and validation
Prepared. Fitonly578891existingtrainanchors/from64trainingwells, notvalidation/test inputs orlabels. Quantilescomputedinside data-loader/preparation, no trainingphase. Storedcurveindex/knots/bounds/fitinfo inconfig/state, fullyfrozenonceinitialized; tau predeclared. No numericalpreflight/review, normalfold5directtrain. Newstemseed20261276 commonpair, otherbranch/readoutinitsame271; orig

ls. Quantilescomputedinside data-loader/preparation, no trainingphase. Storedcurveindex/knots/bounds/fitinfo inconfig/state, fullyfrozenonceinitialized; tau predeclared. No numericalpreflight/review, normalfold5directtrain. Newstemseed20261276 commonpair, otherbranch/readoutinitsame271; original154priorfrozen/no274trainedweights. OriginalQ/data/splits/80-14/b64/64steps/256bins800actions. Initialpriorclampdifference separatelyrecorded. Persistentalpha/knots/best/optimizer/RNG inatomiclast; RAMreplay notindependentdiskreload. Researchpad-zero recommendation adaptedto variableactive dimension toavoid inactiveweights; estimate<=61024params, actualruntimecount authoritative.

## Gains
Unproved. Predefine PLE-minus-scalar_repeat, absolutebest vs274PERM89.482758477027. Samecapacity counts do not implysamefunctionalexpressivity, repeatedscalarredundancy intentionalcontrol. OnePERMaddedstage/6alone/7withSW252, nochainon274weights. StrictTOTAL80.8+meaningfulgain beforeotherfolds, fullOOF82unmet/default5/deliveryunchanged.

## Not worth continuing
Rawscalar outsidefittedrangemaintainedin206mainroute; PLEside saturates. Rare/extreme quantiles/noise/ties can weaken coding; mainmissingflags retainedbutfitting imputedvalues mayreserveintervalsatcommonfillvalues. IfPLEnotbeatsame-structurecontrol afterrestore, close bins8/tau/scope, no bins/width/alpha grid. Do notascribecontrolgainsolelytoPLE; no treebins/teacher/labelretrieval/pretrainedweights.

## Timing
Individual/actualconcurrentwall/peak includesstartup/knotsfit/model/data/initialeval/finalrestore, excludescodeprep/CPUassembly. Max2GPU. No full-data PLEcache, no measuredserialspeedup claim.

Research: ../cross_domain_research_20261001/NUMERIC_PLE_RESEARCH.md (primaryGorishniy etal NeurIPS2022/officialRTDLimplementation; partialboundedencoding adaptation, notfullreproduction).

V276 final PERM89.472074255939, gain+0.066509399020, actualTOTAL80.635753707651; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop8/22, trainingseconds239.697639/peakMiB787.984375. No otherfolds.

V276 final PERM89.482745940151, gain+0.077181083232, actualTOTAL80.639488797125; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop8/22, trainingseconds243.965727/peakMiB787.984375. No otherfolds.

V276 pair finalized: PLE89.482745940151 vs scalar89.472074255939, fair delta+0.010671684212; versus retained274 -0.000012536876, no adoption. Both best8/stop22, new encoder/readout jointly trained. Retain useful positive and matched control. No remaining folds.


