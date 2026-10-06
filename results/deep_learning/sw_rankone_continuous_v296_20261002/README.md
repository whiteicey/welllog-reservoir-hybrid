# SW first-layer rank-one continuous supervision v296

## Focus
Migrate historically positive BatchEnsemble inputdiversity to current protected categoricalSW. This is NOT a new algorithm: historical4member206 fullOOF80.909990/positive+.695075 vs single206 but fifth79.454121; TabM32 width also had localpositivegain. Research provenance: ../cross_domain_research_20261001/TABM_CONTINUOUS_ENSEMBLE_RESEARCH_20261002.md, historical source/results boundaries as documented there.

Only first207->48 newstem becomes four rank1 inputs/output gains, shared W/b; gain1+.1normal seed20261296. Existing newpoint sparse selectors/GLU/readout shared acrossmembers. Finalnewcomponent probability mean (not logits); one sharedstate correction frommean hidden, original104 prior frozen. Allnewbody/rankgains/readout/state jointlytrain. All4members belong to ONE new model/stage, one checkpoint; no old pretrainedensemble/teacher/labelretrieval/posthocmemberchoice. No294trainedweights, oneaddedSWphase/max8.

## Explored and verification
Prepared/untrained, held295/294slots. Same37535params/init/data/Q/seed/eval/prior/share.25ramp5/conditionalRPS1/calibratedBCE/statebound4, only directcontinuousaux differs: mean of4individual hardsoftCE vs hardsoftCE ofmean probability. MixedCE/RPS/evaluation always identical mean prediction; member supervision is already present in oldBatchEnsemble, so this is a matched transfer ablation under current likelihood/protection, not claim of novelty. Separate rankgain gradient group proves actuallearning during ordinarytraining. Original9curves/206+relative/ownSWlabels/max80/pat14/min10/b64/64steps/cos80/max2GPU retained.

## Gain
Unproved. Primary matched delta independent-pooled; absolute referenceSW28974.1537264289993 and continuous25217.88486813991017. SW294point overall74.151817144575 butcontinuous17.380759647270, so it is not a newcontinuouschampion or loadedteacher. Finaldefault/continuous strata and actual144665assembly are required, not defaultmajority gains disguised ascontinuousgain. Fifthstrict80.8meaningfulgain before remainingfolds, fullOOF82 unmet.

## Not worth continuing
No result yet. If independent losespooled or both failretained withoutcontinuousgain, close this firstlayer4member setting; no automatick8/16/alllayer expansion. No conclusion that allTabM is invalid; historyalreadyshowed positive evidence. ConR/RNC alternatives remainresearch only.

## Timing and memory
Fournewmemberactivations/logits costmore despite only1020extra parameters. NoGPUpreflight: directnormaltraining admission4GiB free andmax2jobs, monitor actualpeak; SW1024window/b64 unchanged (no paddingchange). Record individualprep/initial/train/finalrestoreseconds separatelyfromwholepairedwall andactualoverlap; codeprep/CPUassemblyexcluded. Expected memory is a hypothesis, not measured untiltraining.

Implementation initialization is predeclared 1+0.1normal for both rank-one gains, rather than the research note's smaller input-only +/-0.02 sketch. It is an unproved fixed transfer choice, not a validation-selected amplitude; no amplitude sweep. All initial final predictions remain original104 because alpha0 and state correction0.


V296 independent final SW74.153530676552, actualTOTAL80.623597664932; learned gain+0.031777986783, vs211+0.005908053925. Six optionalphases, default5; new37535rankone_sharedencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V296independent finalSW74.153530676552/best17/stop31/peak2424.831MiB, versus289-.000195752447. Strata{"SW252": {"default": 98.64493054166336, "continuous": 17.88486813991017, "overall": 74.14922820781207}, "SW289": {"default": 98.72311025198368, "continuous": 17.72012680313723, "overall": 74.1537264289993}, "SW296independent": {"default": 98.83016481677804, "continuous": 17.473587002886575, "overall": 74.15353067655211}}. Full144665/RAMbest/frozenverified, notdiskreload. PooledstartedPID21508, pairedgainpending; noadoption/currentcombo80.691306 unchanged. Newrankgains group actualgradient.00184643, allnewbody/readout/state jointlytrained.

V296 pooled final SW74.153750372820, actualTOTAL80.623674558626; learned gain+0.031997683051, vs211+0.006127750193. Six optionalphases, default5; new37535rankone_sharedencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.


## Paired completion
Independent SW74.153530676552, pooled74.153750372820; independent-minus-pooled-0.000219696268. Pooled-minus289+0.000023943821. Continuous scores independent17.473587002887, pooled17.469171752586, below25217.88486813991017. No continuous breakthrough or adoption; close this independent-v-pooled auxiliaryCE setting, not historical TabM family. Original104 retained; allnew encoder/rankgains/readouts actuallytrained. Checkpoints retained temporarily as completed matched controls until current bank comparison closes; no automatic extra stages/folds. Paired wall1533.245s, individual independent687.077s/pooled762.144s, partial overlap/gap prevents speed ranking. Actual144665 assembly/own-best RAM replay, no independent disk reload.
