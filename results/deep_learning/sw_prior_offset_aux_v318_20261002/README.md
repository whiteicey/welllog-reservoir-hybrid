# SW residual-density auxiliary prior offset v318

## Focus
Test whether auxiliary supervision should account for the prior in an actual prior-relative residual distribution. Fresh297well raw body/state/FiLM/readout jointly train; only original104 weights loaded. No317 numeric module or trainedrepair chaining.

## Predefined control and verification
Actual output BOTH: softmax(logprior+delta(fused)), delta=gamma*3*tanh(raw/3), gamma5epoch ramp0-to1. Statepatch4/balancedBCE/wholewellFiLM/threepointMLP/readout/lossweights/Q/masks/decoder unchanged. Same65265 activeparams and initialization.
Fourbranch+fused auxiliary halfhard/physical5percentsoftCE1 BOTH uses the same delta/tanh/ramp and label/CONTQ normalization. neutral uses softmax(delta), offset uses softmax(logprior+delta). Only the logprior offset differs; neither optimizes imitation/KL to a teacher. Prior is frozen train-fold predictor input to existing residual architecture. MainCE/RPS reads actualposterior in both. This neutralcontrol is NOT old275raw.logsoftmax aux, because tanh/ramp also change that historical operator. No claim of full275 replication or exclusively diagnosing historical cause.
Atgamma0 original104 recovered; gamma rampsbeforetraining asusual. Nonzero oldfreshreadout init, encoder/readout/state/FiLM allrealgradientgroups, ownwholeSWbest, atomicresume sourceimmutability. No pertrialpreflight/review. Tanhbound3 limits attainable density-ratio correction; do not claim removing all prior limitations.

## Gains
Prepared/untrained, gainunproved. Offset-minusneutral whole/DEFAULT/CONT at ownwholebest plus absolute297whole74.16003022301975/CONT17.502752767347182 and252CONT17.88486813991017. Actual144665fixedtargetcomposition before any adoption. Currentoptional8TOTAL80.69570795800789/default5/delivery unchanged. StrictTOTAL>80.8+usefulgain beforeotherfolds, originalfullOOF82unmet.

## History and closure
263/275tilt mainposteriorCE alreadyexists but theiraux independentrawsoftmax;269/219/242 frozen-feature posterior CE differ from this freshrawfourbranch sameoperator pair. Learnedmix262/prioruniformmix269/PERM274 differ. PCGrad304/detach301/splitv11 do not prove representation-conflict bottleneck; do not revive them. 317initialpairedCONTpositive disappeared afterLbest26; finalLR-Lwhole+.004900/CONT-.007868, no317positivecarry. This pair changesauxoffset only, not a newarchitecturecapacity win. If offset fails whole/CONT versusneutral, close without offsettemperature/bound/lossweight grid; negative applies to this setup, not all residual densitylearning.

## Timing
Prepared, no launch/queue. Max2realGPUjobs; eachdata/warmup/train/restore time/peak andpairwall inclgap, excludesbeforefirstlaunch andCPUfinalization. Noisolatedspeedclaimfromvariableoverlap.

V317 FINAL: LR74.1669496554495/best17/natural31, L74.16204929073999/best26/natural40; LR-Lwhole+.0049003647095/CONT-.0078684427712. BothCONTbelow297/252, notadopted/no widththresholdgrid. Realembeddingweight/bias/outdeltas nonzero (LR.700/.717/2.999,L.260/.324/1.599), notdeadmodule. Pairwall638.151032s, each504.753633/542.533978s,90.282s stagger/peaks1315.602/1297.477MiB/variableoverlap, notisolatedspeedranking. Sourcefrozen/RAM144665exact, noindependentdiskreconstruction. V318neutralPID91028/offsetPID79268explicitpairlaunched, fresh297rawbody/state/FiLM/readoutjoint65265, sameactualdelta/formula/ramp/params; onlyauxprioroffset0-vslogprior. No317module/weights or priorKL. Retainedoptional8TOTAL80.695707958008/default5/delivery unchanged, nootherfolds.

[Historically screened auxiliary prior-offset research](../cross_domain_research_20261001/SW_PRIOR_OFFSET_AUX_RESEARCH_20261002.md). This is localinput-dependent prioroffset, not globalclassfrequency logit-adjustment theory.

V318 live verified both PID91028neutral/PID79268offset epoch11, sameinitialSW74.12175268976877, ownbeststill0,65265activeparams/realnormaltraining/max2CIMconfirmed, nostderrerrors. Source104frozen, newrawencoder/state/FiLM/densityjoint; nominalmixalpha buffer meansgamma=alpha/.25, not probabilitymixture. Auxinitiallossdropin offset is mathematical baseline difference, not officialgain. CPU terminal evaluator prepared for wholebeststrata/readoutdelta andactualcomposition; await14patience/min10. Atlas refreshed317closure/318nodes; retainedoptional8TOTAL80.69570795800789 unchanged, strict808/fullOOF82unmet.

V318 neutral: SW74.121752689769, actualTOTAL80.612475369558; fresh65313jointencoder/readout/state, original104prioronly; derivedproxy background, oneaddedSWphase. No fullOOF/otherfolds.

V318 offset: SW74.121752689769, actualTOTAL80.612475369558; fresh65313jointencoder/readout/state, original104prioronly; derivedproxy background, oneaddedSWphase. No fullOOF/otherfolds.

Final318 whole scores {'neutral': 74.12175268976877, 'offset': 74.12175268976877}; offset-neutral +0.000000000000; CONT paired +0.000000000000; strata {'neutral': {'default': 98.71563205981698, 'continuous': 17.63188906048012}, 'offset': {'default': 98.71563205981698, 'continuous': 17.63188906048012}}. Actual six-stage assemblies recorded; optional eight-stage adoption requires whole/continuous assessment, not automatic. Pair wall 244.686s; no fullOOF or remaining folds.

FINAL CLOSED: both natural14/best0; SW74.12175268976877, learned gain0. Pair wall244.6860683s, individual239.7933385/239.1074064s, peak1203.3418/1203.0918MiB. Actual6 TOTAL80.61247536955786; no adoption, no independent disk reload, no sweep.
