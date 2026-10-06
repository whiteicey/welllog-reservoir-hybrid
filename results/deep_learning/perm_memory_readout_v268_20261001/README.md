# V268 PERM learned prototype memory readout

## Focus
Replace repeated coadaptation/domain penalties with a conditional latent-dictionary readout; retain frozen154function and positive264equalwell learning mechanism. Prototype memory consists solely of randomly initialized trainablekeys/values, no sample/label retrieval or teacher. Research/source relation is conceptual attention/memory, not fullHopfield reproduction.

## Explored and validation
Prepared/untrained. Same12609mass/background from154frozen128contextfeatures. Newbounded.25 hidden residual: query128->64 +32keys64 +32values32 +zero32->128 projection (15552params) vs pointMLP128->60->128 (15548). Total28161vs28157, fourparameter difference explicitly near-capacity not exact; no dummyparams. Same zeroinitial feature residual, seedmass20261010/newbranch20261268/sampling/optimizers; fresh154/no248/264repairweights. Both originalpredictors fullyfrozen. Jointnewrepresentation3e-4/mass1e-4 onePERMphase, halfrow/halfequaltrainingwell normalizedconditionalNLL+jointRPS1, originalQ/64x64/80-14/SAM0 unchanged. No hiddenaugmentation/entropicrisk/RNC or teacher labels. Zero finalprojection delays earlyquery/key/value and MLPfirstlayer gradients; masszerooutput further delays firststeprepresentation learning. Record actualgroupgradient secondstep, not claim allparameters immediately active. Attentionq layernormalized/dotkeys/sqrt64/softmax, learnedvalue32, no tokenlabels/IDs. Uniformmemorycontrol omitted because query/key lackgradient/onlyconstantresidual, misleading samecount.

## Gain
Unproved. Predefined memory-minus-MLP near-capacity/init/loss/budget and absolute retained264PERM89.415397483681/currentoptionalTOTAL80.625533220240. Both alternativeonePERMaddition from154, notstacked264weights. Sixalone/sevenwithSW252/eightonlyiffuturePORpositive; noqualification/fullOOF claim. Initial.001uniformfloor changes154 function but matchespairedinitial; learnedgain separate. All144665/RAMbest/frozen checks then diskreload ifqualification.

## Not worth continuing
SW235 memory had no stronggain; this targets differentPERMhidden and effectivewellmean loss, not proofmemory generally improves. Input-dependent prototypes may still lack needed rawinformation or overfit trainingclusters. If memory losesMLP/retained264, close32token64query variant, no tokenwidth/temperature grid. Superior storedcount match not proofequalfunction capacity, and no Hopfield superiority claim.

## Timing
Individual and actualconcurrentwall/peak inclstartup/data/initialeval/finalrestore; excludesprep/CPUassembly. Max2GPU, hold267actualterminal. No serialspeedup claim.

V268 liveverified PIDs88004/68720 bothepoch6: MLPbest89.413337894012/memorybest89.413778421887, bothe5 best, below retained26489.415397483681. Snapshotmemory advantage+.000440527875 not finalgain; continue80-14. Actual28157/28161trainable/790.642and790.361MiB peak, sameinitial89.404398334646, finitegroupstats/losses. Source154fullyfrozen, memory no labels/samplestore. V267negativeweights deleted afterfull144665composition/SHAmanifest, positives264/154 retained. Atlas1038experiments/199deadends/221unresolved, incompleteancestryexplicit. Current7TOTAL80.625533220240/default5/delivery unchanged/no808/nootherfolds.

V268 final PERM89.415840060079, gain+0.010275203160, actualTOTAL80.616071739100; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop9/23, trainingseconds227.018359/peakMiB790.641602. No otherfolds.

V268 final PERM89.415277247011, gain+0.009712390092, actualTOTAL80.615874754526; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop10/24, trainingseconds236.025189/peakMiB790.360840. No otherfolds.

V268 FINAL MLP PERM89.415840060079; memory-minus-MLP-0.000562813068. Actual seven-stage TOTAL80.625688121979, previous7delta+0.000154901739, MICROnotmaterial. Full144665 finite/official composition/RAM replay/frozen proof, no independent diskreload. Both checkpoints retained as positive/control; default5 and delivery unchanged/no808/nootherfolds. Representation gradient small, no memory-width sweep. V269 two SW jobs running. User requests larger-potential direction: next PERM raw-feature multiscale representation directly coupled to distribution readout, original154 retained/one joint additional phase; prepare while max2 occupied.
