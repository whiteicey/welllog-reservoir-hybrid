# V269 calibrated class-balanced SW state training

## Focus
Target default-versus-continuous state learning, not another backbone replacement. Historical continuous points often receive finaldefault action, but that statistic is NOT rawclassifier accuracy. Test whether balanced calibrated gate gradients help with useful densityreadout.

## Explored and validation
Prepared/untrained. Same16770params (12609mass/background+4161independentgate), frozen104only/no252/253repairweights. Both gate residualbound2 (vs253.5), pairedzero gate/.001uniformbackgroundinitial. Carry252conditionalRPS1 +halfhard/halftolerance binCE. Two-factor SAMequalrho.1/sqrt2 pergroup botharms, independent1e-4 groups, original64x64/max80/pat14/min10/Q50/splits/scoring. OnlyBCE differs: controlordinary; balanced Q-weighted TRAININGdefaultprior p, wdefault=.5/p andwcontinuous=.5/(1-p). TRAININGlogit g+log(wdefault/wcontinuous); weightedBCE. Evaluation uses g withoutoffset, so populationoptimum g retains probability logodds, rather than blindclassprior thresholdshift. No heldoutlabels/statistics fit, sourcegate frozen. Traininglabelprior not Q/anomaly deletion; alltrainingpopulation remains. Fixedstaticcriterionprior reportedconfig/derivedsame trainingdata for resume, no hiddenmutableEMA. MassCDFterm remainscontinuousonly/non-BCE identical. Sixalone/sevenwithcurrentPERM/eightiffuturePORpositive; eachhead<=1addition.

## Gain
Unproved. Predefined balanced-control sameparams/init/radius/bound/budget. Absolute retainedSW25274.149228207812/currentoptionalTOTAL80.625533220240. Initialfloor difference104 explicitly scored, no claimearlyprediction is learnedgain. Stagealoneactual144665/RAM/frozen and independentdiskreload ifqualification. Default/continuousstrata diagnostics only, not rowselection or oracle.

## Not worth continuing
240 normalizedconditionalNLL did not improve enough; 253 balancedSAM yielded tinygain but gatebound.5. This tests stateBCE with known prior-correction, not claimsnew bound alone useful. State/densityambiguity may make default action legitimate; fewerdefault predictions alone not improvement. If balancedloss harmsaverageofficialscore or pairedgainabsent, close fixedclassweights/bound2, no gateweight/boundgrid.

## Timing
Individual/concurrentwall/peak inclstartup/data/initialeval/finalrestore, prep/CPUassembly excluded. Max2GPU; helduntil268terminal. No serialspeedup claim.

V269 control final SW74.133230187542, actualTOTAL80.616492493778; learned gain+0.011528606867, vs211-0.014392435085. Six optionalphases, default5; samecap16770/statebound2/equalradiusSAM/conditionalRPS1, calibratedclass-balanced BCE versus ordinaryBCE.

V269 balanced final SW74.143370538826, actualTOTAL80.620041616728; learned gain+0.021668958151, vs211-0.004252083802. Six optionalphases, default5; samecap16770/statebound2/equalradiusSAM/conditionalRPS1, calibratedclass-balanced BCE versus ordinaryBCE.

V269 FINAL ordinarySW74.133230187542/calibratedbalanced74.143370538826, pairedgain+0.010140351284, retained252delta-0.005857668987; both best5/naturalstop19, actual144665/RAMreplay/frozenproof/no independentdiskreload. Keep useful positive mechanism/control checkpoints; do not replace252. Individual449.070910/445.477191s/1119.828MiB, concurrentwall455.858s inclstartup/data/initialeval/finalrestore/excludesprep. TrainingQ-defaultprior.7228821516/logitshift-.9588033485 onlytrain, not inference. No classweight/bound sweep. V270 launched PERMhead-only82720 vs copiedlateencoder joint117408trainable, exactinitial15489.405564856919 verified during training. Max2/80-14/onePERMphase/nootherfolds. Currentbestactualoptional7TOTAL80.625688121979 unchanged/default5/deliveries preserved/fullOOF82unmet.
