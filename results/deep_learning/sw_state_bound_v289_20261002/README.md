# SW v289 stronger state correction under joint raw representation

## Focus
Same57617rawencoder/state/density as285, joint3e-4/alpha.25 ramp5/calibratedBCE/directcontinuousCE1/RPS1. Statebound2?4 only, old104priorweights fixed. Freshoriginal104/init, no285trainedweights/no extraSWphase. Sameinitialfunction andfirststep correctionderivative1, differentlaternonlinear bound.

## Explored and validation
Preparedheldslots. 285official74.152376 vs25274.149228 smallgain comesfromdefaultstratum, continuous17.71256below25217.88487. Test whether boundedstatecorrection restricts erroneousdefaultdecisions, hypothesisunproved/no claimedtrain-saturation observation. Samecapacity/seed/data/Q/full80-14/min10/b64/64steps asnatural285; no18prefix-budget mismatch. Allnewencoder/density/state retrainedjointly, nothead-only finetune.

## Gains
Unproved. Predefine pairedabsoluteSWdelta vs285 andactualdefault/continuous decomposition; current7TOTAL80.655864 remains. OneSWadditionalstage/7withPERM281/8withPOR287; <=8. No strict808pass/fullOOF82claim/otherfolds/default5deliveryoverwrite.

## Not worth continuing
Largerstatebound can falselydefault continuouspoints anderase densitygains. If4 fails2 control, close bound4, no6/8/logitgrid. Do notcountdefault-onlyimprovementascontinuousrepresentation progress. Nooracle or changingofficialscore/population.

## Timing
Atmost2normalGPU, individualelapsed/peak andactualoverlapwall separately; startup/data/initial/finalincluded, codeprep/CPUassemblyexcluded. PreparedheldPOR287/PERM288slots, explicitlaunchonlyafterterminal.

V289 balanced final SW74.153726428999, actualTOTAL80.623666178289; learned gain+0.031973739231, vs211+0.006103806372. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V289 finalSW74.153726428999/best32/stop46, same80-14 budget vs285delta+.001349683148. Actualoptional8TOTAL80.685480205378, eachheadone addedphase. Strata{"SW252": {"default": 98.64493054166336, "continuous": 17.88486813991017, "overall": 74.14922820781207}, "SW285": {"default": 98.72446679129287, "continuous": 17.71256117758909, "overall": 74.15237674585131}, "SW289": {"default": 98.72311025198368, "continuous": 17.72012680313723, "overall": 74.1537264289993}}. Smallgain/notmaterial breakthrough, no bound6/8grid. Original5/oldrecipes/deliveries unchanged; no otherfolds, strict808false/fullOOF82unmet.
