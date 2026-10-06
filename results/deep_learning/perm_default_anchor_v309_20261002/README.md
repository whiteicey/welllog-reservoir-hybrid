# PERM default-label direct component regularizer v309

## Focus / evidence
307full continuous74.070897921839 >29174.001066339761 (+.069831582078), but default97.418212924816 <29197.494555208792 (-.076342283976), whole89.488856959474 <89.515554793257. Frozen gate does not freeze default utility action; conditional mass matters. ONE fixed small default auxiliary CE, coefficient.05, not grid. Original154 prioronly/fresh307fullbody-readout-attentionjointonephase; no307trainedweights/extra stages.

## Explored / true control
Same55490params/init/forward/alloldlosses/.3ramp5/3e-4/seed/80-14/Q50 as completed307full lambda0; historicalcontrol not concurrentlyrerun. Before originalobjective clearsaux_logits, retainfinaldirectcategorical logitgraph. Anchor=weightedmean_DEFAULT CE(finalq, bucketize ACTUAL trainz) *qmean/3, add.05anchor. Labels are actualtraininglabels, no distillation/teacher/pseudo/validationstatistics. Do not addCEtofinalmixedmass or changegate/decoder. Allnewbody/readout/interaction joint; no frozenhead-onlytraining.

Correct historical statement: old all-row JOINT RPS already givesnumeric mass DEFAULT supervision; CONT-onlyconditionalNLL/directCE omit only directDEFAULT CE. This is not fixing missingallDEFAULT gradient, not a properconditionaldensity NLL correction. It deliberately biases q(y|CONT) toward observedDEFAULT values, may double-count/defaultcollapse or damage CONT. Originaljointdistribution remainsnormalized; trainingregularizer semantics explicitlydifferent. Train-only initialization records how manyDEFAULT labelsfalloutside256edges; out-of-range buckets push edge mass, do not inventgridextension or extraaction. Use actualz even ifcanonicaldefault_z differs; no validationrowclassification at inference.

Prepared/untrained, historyresearch pending. Ordinarysourcefunction/frozen/allgroupfinitegrad/atomicstate/ownwholePERMbest144665RAMassembly; DEFAULT/CONT metrics onlyafterwholebest, no stratumselection. Single-seed warn-onlyattention nondeterminism. Independentdiskreload onlyqualification/delivery.

## Gain
Unproved. Comparewhole/DEFAULT/CONT versus307full andretained291; adoptonlywholepositive andactual8assembly. Current8TOTAL80.695620593940/default5/oldOOFdeliveryunchanged. StrictTOTAL>80.8 usefulgain beforeotherfolds, fullOOF82unmet. Ifbenefitappears, laterdifferentheadmigration requires freshdocumentedexperiment, noautomaticqueue.

## Unworthy continuation
If defaultgain is paid for by largerCONT/wholeloss, close this.05regularizer; no coef/edge/decodergrid. If masks/noDEFAULT batch, safezeroaux denominatorclamp; no samplefilter. PriorDEFAULT supervision alreadyexists, so no presumedbenefit. No teacher/priorKL fallback.

## Timing
Individual load/initial/train/ownbestrestore/peak andactualoverlap; codeprep/CPUrecord excluded. Max2GPU, onlyexplicitlaunchafteractiveslots. No speedranking with differenthistoricalcontention.

Prepared309, NOT launched: fixed.05 realDEFAULT-label auxiliary CE on finalnewq; samefresh307full55490params/init, original154only/allnewencoder-readoutjointonephase. Corrected explanation: originaljointRPS alreadysupervisesDEFAULT, so onlydirectCEadded/semanticbias notlikelihoodfix. Historyresearch pending; current308occupies2GPU. No automaticqueue/coefscan/otherheadmigration.

Research completed: [DEFAULT anchor CE primary/history](../cross_domain_research_20261001/PERM_DEFAULT_ANCHOR_CE_RESEARCH_20261002.md). Original labels are realtruth log10 and not uniformlyreplaced. Grid uses nonDEFAULT trainingmin/max centers, internalmidpoints, outeredges?inf; DEFAULT actualz/canonicaldefault_z expectedbin0, no separate.01numericcenter. DirectCE only pushes tailmass, not sub-bin DEFAULT variation. Initializationrecords bins/counts/firstfiniteedge/canonicalbin and outsideedges(typically0due infiniteedges); no gridextension or labelrewrite. ExistingjointRPS DEFAULT supervision and semanticbias retained. No equivalentcurrentraw directDEFAULTCE found in searchedhistory; not globalnovelty.

Explicit309 launched PID87620; original154 only,55490params/samefresh307fullinit andforward, allnewbody/readout/attentionjointonephase. Fixed.05DEFAULT auxiliaryCE, actualtrainbins[0]/counts[399064]; no gridextension/decoderchange. Overlaps308additive, max2GPU. Gainunproved; pending wholebest/CONT-defaulttradeoff/actual8/time.

V309 final PERM89.506715700712, gain+0.101150843794, actualTOTAL80.647878213322; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop6/20, trainingseconds331.898410/peakMiB1813.562012. No otherfolds.

Final309 PERM89.506715700712; versus307full{'PERM': 0.01785874123898168, 'default_PERM': 0.06280395301871522, 'continuous_PERM': -0.06953340475654102}, versusretained291{'PERM': -0.008839092544718596, 'default_PERM': -0.013538330957288736, 'continuous_PERM': 0.0002981773214827399}; adopted=False. WholePERM-best/stop6/20,144665RAM/frozenexact, no independentdiskreload or fullOOF. DEFAULT-only true-label regularizer notproperconditionalNLL correction; checkCONTtradeoff, no coeff/edge/decodergrid. Individual331.898s/peak1813.562MiB; historicalcontrol notconcurrent/timefairnessnotclaimed. OnePERMaddedphase, currentdelivery intact, nootherfolds.

Direction decision afterFINAL: localoverall+.017858741239/default+.062803953019 butCONT-.069533404757 consumes307continuousgain. Absolute89.506715700712<29189.515554793257; noadoption. Do not automaticallyrunSWdefaultCE transfer despite lowpriorityresearch suggestion. Preserve scopedpositive evidence/control, no coefficient/grid/decoder scans. Nextprioritize continuous depthrepresentation with allnewbackbone/readoutjointonephase; researchpending.
