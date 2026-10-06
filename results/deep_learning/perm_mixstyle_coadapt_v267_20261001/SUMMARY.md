# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# V267 PERM joint readout with conservative hidden-style augmentation

## Focus
Test different crosswell mechanism [MixStyle paper/officialcode](https://github.com/KaiyangZhou/mixstyle-release). Training-only featurestatistics exchange, not probabilitytilt itself. Hidden amplitude may represent truephysics; no assumption it is nuisance.

## Explored and validation
Prepared/untrained. Original154checkpoint only/no264repairweights. Freeze originalSOURCE/expertcompactheads/routes; jointly update154contexthead1e-5 and12609newmass/background1e-4 in ONE PERMphase. Contextheadexpand prehook receives128hidden; validtoken mean/std detached; partners chosen differentTRAININGwell, p.25 uniformstrength<=.1, invalidtokenszero. Control computes same stats/RNG but returns originalhidden; mixstyle changesexpand input and mass branch. evalalwaysidentity/no IDs required, frozen source retains9curve/206+63 contract. OriginalQ/sampling/64x64/80-14/min10/SAM0, conditionalNLL halfrow/halfQnormalizedequaltrainingwell +jointRPS1 retained264 effective mechanism. DifferentiableGaussian16finiteCDF checkpoint512chunks; no ?infinite-gradient. No entropicrisk/RNC/beta penalty. No new normalization fitting/heldoutlabels/teacher/extra data. PartnerIDs only matching, not inferencefeatures. Newmass initial.001uniformmix nonidentity/scoreseparate. Existing154contextinit preserved, coadaptation can change it late

atistics may encode actual permeability or physicalamplitude; mixing can harm labels even when bounded. No universal label-preservation claim. If officialpaired/absolute gain absent, close this p.25/strength.1 preexpand design, no augmentationstrength/layer sweep. LatePOR266 currentlydegraded but differenttarget/update, not broadbackbone invalidation.

## Timing
Record individual/actualconcurrentwall/peak inclstartup/data/initialeval/finalrestore; excludesprep/CPUassembly. Max2GPU, holduntil266terminal.

V267 liveverified PIDs84480/36084 bothepoch3: control89.370909727960/mixstyle89.368278460947, bestinitial89.404398334646/no learnedgain yet, below retained26489.415397483681. Same95329trainable(82720oldcontext+12609newmass), peak786.465MiB, bothgradgroupsnonzero/finite losses, traininggroupstats31/27, no extraGPU. Continue80-14/max2. Originalsourcefrozen, evalstyleidentity. V266negative best/last deleted afterfull144665assembly/SHAmanifest, original151/current264preserved. Atlas1035experiments/198deadends/220unresolved, incompleteancestryexplicit. Currentoptional7TOTAL80.625533220240/default5/delivery unchanged; no808/nootherfolds.

V267 final PERM89.404398334646, gain-0.001166522273, actualTOTAL80.612067135198; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds241.300178/peakMiB786.465332. No otherfolds.

V267 final PERM89.404398334646, gain-0.001166522273, actualTOTAL80.612067135198; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds237.493554/peakMiB786.465332. No otherfolds.

V267 FINAL control/mixstyle PERM89.404398334646 both/best0/natural14stop/no learnedgain, below26489.415397483681. Full144665assembly/RAMbest/frozenSOURCE checks/no diskreload. Individual241.300178/237.493554s/786.465MiB, pairwall246.882s inclstartup/data/initialeval/finalrestore/excludesprep. Closecontext+mass coadapt and conservativeStyle here, no layer/strength grid. V268 launched frozen154 newrepresentationMLP28157/memory28161 incl12609mass, bound.25/zero feature residual/sameinitial89.404398334646, onePERMphase/equalwell learning/max2/80-14. Memory is purelatent parameters/no sample-labelretrieval. Gainsunproved; current7TOTAL80.625533220240/default5/delivery unchanged/nootherfolds. Research docs/deep_learning/research/perm_learned_memory_readout_20261001.md notesnear-capacity and delayedupstreamgradients.

No-gain checkpoints removed afteractualcomposition; CHECKPOINT_DELETION retainsSHA/path/reason. Currentpositive/active weights untouched.


