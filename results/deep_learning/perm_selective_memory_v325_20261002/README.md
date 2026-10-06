# PERM content-selective bidirectional memory v325

## Focus / hypothesis and controls
Earlier point sparse encoder291 and tiny rational shape312 improve PERM but pointwise transformations have no learned sequential state. Add one compact48 memory after first GLU before second selector, jointly train all freshbody/shape/memory/readout from original154 checkpoint only. Do not reuse trained291/312 repair weights. Point-content gate vs valid-windowmean gate same43626active(+7056memory), same initialization. Value=SiLU(Wv h), g=1/64+(1/4-1/64)*sigmoid(Wg LN(h)); recurrence state=(1-g)*previous+g*value. Shared forward/back parameters and averaged state, zero-output bounded .25 residual. Initialgate1/16, both initial predictive functions unchanged.

## Explored / validation
Implemented parallel associative affine prefixscan,8tensorstages for256, using oldA/B simultaneously; no serial256 Pythonloop/cumproddivision/releasedCUDAweights. Float32 recurrence under disabledautocast, invalidpositions A=B=0 resetmemory, bidirectional flipmask. O(LlogL) work, not authorMamba O(L) performance. Nominalmax80/pat14/min10/b64x64steps, originalQ50/loss/alpha.3ramp5/seed20261010/9curves206 and documentedside contract, original decoder. Samecapacity pair isolatesgateinput scope; historical312 lacks7056 so cannot prove totalmemorycausality from historicalscore.

## Gains
Prepared only, no actualgain. Baseline312rational PERM89.515804404879/CONT74.002888404013; optional8TOTAL80.697807943123 unchanged. Ordinary initial evaluation/source/frozen/bestreplay checks partoftraining, no pertrial numericalpreflight/review. Sixstandalonephases,oneextraPERM,maxeightassembled,strictfifthTOTAL>80.8 beforeotherfolds/fullOOF>82 unmet.

## Closed / do not extend
EarlierfixedTCN/pooling/windowstatselector and input-independent FFT state-space do not test contentselective recurrentforgetting. Precise historicalaudit research incoming; not firstsequencelearningclaim. If matchedwhole/CONTgain doesnotpersist, closegatescope no timescale/rank/width/LR grid. Retain only positive evidence; single-seed microgainnotrobustness.

## Timing and unfinished
Unlaunched while324occupies2GPU. Explicitlaunch onlyafterfreshactualGPUcheck; no automaticqueue. Record eachprocess wall/peak/stagger and same-work pair total; prep/initial/train/bestrestore included, implementation/CPUfinalization excluded. No isolatedspeedclaim; memoryestimate not measured.

Researchaudit: [input-selective memory research](../cross_domain_research_20261001/PERM_INPUT_SELECTIVE_MEMORY_RESEARCH_20261002.md), primaryMamba arxiv2312.00752/authorselective_scan source. ThisEMA simplifiedmemory is not fullMamba or publishedbenchmark reproduction; olderfixedtauFFT/TCN/selector/binconv differ. Initialzerooutput delaysgate/valuegradients untiloutputlearns. Syntaxcheckpassed; trainingnotyetlaunched.

V325 selective explicitlylaunched PID76132 afterfreshglobalGPUprocesscheck showedonlyPOR324referencePID65712. Ownjoint43626 active; windowcontrolunlaunched/noautoqueue. V324adaptive naturalstop43/best29 POR78.027986511084, RAM144665replay/frozenexact andrecordactualassembly completed, below302; pairedfinal awaitsreference. SW326preparednotlaunched.

V325 bothinitialPERM89.405566626894/43626active, selectivePID76132/windowPID66616 live. Windowexplicitlylaunched afterfreshCIM confirmedPOR324terminal; max2GPU, no automaticqueue/otherfolds. Currentresultsnotfinalgain.

V325 final PERM89.506957270992, gain+0.101392414073, actualTOTAL80.647962762920; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds258.757945/peakMiB1007.933105. No otherfolds.

V325 final PERM89.509398334584, gain+0.103833477665, actualTOTAL80.648817135177; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds253.656007/peakMiB1007.933594. No otherfolds.

## Final closure
Bothnaturalstop19/best5: selective/window PERM89.506957270992/89.509398334584, CONT74.005909160716/74.008989668121. Pairedwhole-0.002441063592/CONT-0.003080507405, absolute312whole-0.008847133886/-0.006406070294. BothCONTtiny+0.003020756703/+0.006101264108 versus312 but wholeloss; no adoption. Memoryvalue/update/output actualparameterdeltas nonzero, all144665 RAMbestreplay/frozenexact, notindependentdiskreload. Close this contentselective gate memory setting, no timescale/width/rank/LR grid; not allMamba/state-spaceineffective. Individual258.757945/253.656007s, totalpair297.245296s/stagger37.707976s, peak1007.933105/1007.933594MiB; initialPOR324tail/CPUoverlapdisclosed, notisolatedspeedclaim. Bestrecipe80.697807943123/default5/oldsubmission unchanged, strict808/fullOOF82unmet.
