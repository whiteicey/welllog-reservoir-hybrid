# V264 PERM training-well robust risk

## Focus
Stop repeated SWencoder replacements without gains. Test a different question: does continuousPERM training overemphasize easy/longwells and miss generalization across wells? This is a hypothesis, not established diagnosis. Source inspiration GroupDRO [paper](https://arxiv.org/abs/1911.08731), [officialcode](https://github.com/kohpangwei/group_DRO); local stateless entropic group-risk adaptation, not full original reproduction.

## Explored and validation
Prepared/untrained. Same12609parameter248 effective mass/background architecture, newinitialization from immutable154, no248repairedweights. SameownconditionalnormalizedNLL/BCE/jointRPS1/seed/80-14/1e-4/SAM0. Change ONLY half of conditionalNLL: globaloriginalrow-weighted conditionalrisk.5 +pertrainingwell Q-normalized risk.5. well_mean uses mean eligiblewell risks; robust uses 2*log(mean(exp(wellrisk/2))). >=8 continuous rows eligibleperbatch; no eligible fallback originalrisk. Same grouping/forward/initboth, no parameters for weights. All labels real trainingPERM, Q50 unchanged, training anchorpopulation/windows unchanged, no points removed. IntegerwellID only loss aggregation, no modelinput. Robust softweights derived from trainingloss; stateless no EMA/crossstep state. No automatic abnormal/noisylabel determination. Earlystop/selection by officialPERM/actual144665. OnePERMaddition/sixalone/sevenwith retainedSW252<=8.

## Gain
Unproved. Predefined robust-minus-well_mean samecapacity and absolute retainedPERM24889.413465595388; original15489.405564856919. Balancedwell control differs248 original rowrisk; do not conflate groupbalancing and robustness. Fifthstrict80.8/fullOOF82 unmet. Need restoredRAM/frozen/fullpopulation score; independentdiskreload if qualified.

## Not worth continuing
The robustgroup may favor small/hardtrainingwells and hurt average officialscore, which is rowweighted. Temperature2 andfraction.5 fixed; close if no paired/absolutegain, no grid or validationderived groupweights. SourceGroupDRO evidence concerns othertasks, does not promise transfer or prove wellshift here. Continuedwidth/decoder/RNC/SWreplacement scans held.

## Timing
Record individual and actualconcurrent wall/peak; startup/data/initialeval/finalrestore included, prep/CPUassemblyexcluded. No serialspeedup claims. max2GPU.

V264 liveverified PIDs47772/62424: wellmeanepoch4/best89.410577810895, robuste5/best89.412670737129, both below retained24889.413465595388. Differentepoch snapshot not final pairedgain; continued80-14/max2. Same12609params/789.904MiB peak, finitegroupstatistics, trainwellaggregation only. V263 negativeweights removed aftercomposition/SHAmanifest, original104/current252 intact. Atlas1026experiments/195deadends/217unresolved, incompleteancestry explicit.

V264 final PERM89.415397483681, gain+0.009832626762, actualTOTAL80.615916837361; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop8/22, trainingseconds222.600644/peakMiB789.903809. No otherfolds.

V264 final PERM89.414811070835, gain+0.009246213917, actualTOTAL80.615711592865; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop10/24, trainingseconds219.909478/peakMiB789.903809. No otherfolds.

V264 FINAL wellmeanPERM89.415397483681/best8/natural22, robust89.414811070835/best10/natural24, robustminuscontrol-0.000586412845, wellmeanvs248+0.001931888293. Actual144665/RAM/frozenexact/no diskreload. Individual222.600644/219.909478s/789.904MiB, pairwall228.643s. Retainbothpositive/control models. Actual264wellmean+252+151 sevenphase TOTAL80.625533220240, delta+0.000676160903 MICROnotmaterial/no808/nootherfolds/default5unchanged. OldPLAN obsolete26849 newparameters corrected by PLAN_ERRATA actual12609 without mutation. V265 firststepfailedonly, V266 freshdiffAPIfix launched onePORphase/copylateblock+151head/control-vrex; gainunproved.
