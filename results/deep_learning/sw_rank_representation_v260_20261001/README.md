# v260 continuous representation supervision, fixed original default-state branch

## Focus
Inspired by Rank-N-Contrast author loss https://github.com/kaiwenzha/Rank-N-Contrast/blob/main/loss.py and paper https://arxiv.org/abs/2210.01189. Local variant applies pairwise target-distance order contrast to one-view existing128hidden, no separate projector, only real continuousSW training points. Original paper uses two augmentedviews; this is a mechanism transfer, not full reproduction. Rank denominators include all target distances >=currentpair, ties included; efficient sorted suffixsum implementation. Feature negativeL2/temperature2, own physicalSW target distances.

## Explored and validation
Prepared. Same251 continuousbody/mass architecture858522trainableparams/init/loss/optimizer botharms, original104 atom network immutable. Entire continuous branch3e-5 plus new12609mass/background1e-4, SAM0/evalNoDropKey. CE ownhalfhard-halftolerance plus auxiliary coefficient0 or0.1 (wholeloss/3 convention). No teacher/extra data/pseudolabels/validation loss. Main likelihood and auxiliary train jointly ONE SWphase, starts104only/no251trainedweights. Sampling trainingcentral rows with rawindex%32==0, duplicate rowsremoved, max8pertrainingwell/max256total; metadata used only for quota, no wellID feature. Botharms compute same sampler/auxiliary diagnostics, no added stochasticRNG. Original9curves/Q50/fold5/256bins/800actions/80-14/min10 intact. Final RAMbest/frozenatom/all144665actualcomposition required; no diskreload/bitwise trainingrerun claim (known CUDA attention/interpolation backward warn_only).

## Gain
Unproved. Predefined RNC-minus-CE samearchitecture/parameterupdates/budget, absolute retainedSW25274.149228207812. RNC no extra parameters;858522 updates includes unused continuouscopygate rows which have no loss butweightdecay, original atom gate remains unchanged. No original positive models/delivery overwritten. Sixalone/sevenwith retainedPERM248/total<=8.

## Not worth continuing
251 body likelihood coadapt had no learnedgain. This adds explicit continuous geometry to same update scope, new hypothesis. If RNC auxiliary improves but officialcontinuous score does not, close weight0.1/temp2 transfer; no temperature/weight grid. Do not assume featuregeometry fragmentation/noisylabels are established problems. Research alternatives Ti-MAE/TS2Vec not simultaneously compounded.

## Timing
Record individual and actual concurrentprocesscreation-to-finalreportwallclock, peakMiB; startup/model/data/initialeval included, preparation/CPUassemblyexcluded; no serialspeedup claim. 256^2 auxiliary bounded, never8192^2.

V259 both arms failed before first optimizer step: torch argsort stable overload requires dim as keyword. No trained gain; immutable source/log/error kept. V260 new source fixes only argsort(dim=-1,stable=True), resets both arms from original104, identical hypothesis/config.

V260 live verified: PIDs74868/77992, both completed epoch1 with finite loss and RNC70 sampledpoints; ~1496.63MiB peak each. Epoch1 CE73.943206247637/RNC73.929819934377, both below identicalinitial74.121701580675; no gain yet, continue80/14. V258 negative checkpoints removed with SHA manifest after actualcomposition; sources/config/results/predictions retained. Atlas batch built1014experiments/191deadends/213unresolved; unresolved ancestry remains explicit.

V259 ce final SW74.121701580675, actualTOTAL80.612457481375; learned gain+0.000000000000, vs211-0.025921041953. Six optionalphases, default5; same-architecture continuous-representation auxiliary0 versus0.1 one-stage study.

V259 rnc final SW74.121701580675, actualTOTAL80.612457481375; learned gain+0.000000000000, vs211-0.025921041953. Six optionalphases, default5; same-architecture continuous-representation auxiliary0 versus0.1 one-stage study.

V260 terminal early12 both/best0 initializationSW74.121701580675/actualTOTAL80.612457481375. RNC-minus-CE restoredbest0. All144665 composition/RAMreplay/frozenatom verified; no independentdiskreload. Individual223.858171/226.733864s, peak1496.630371MiB, concurrent wall232.331s inclstartup/data/initialeval/finalrestore/excludesprep. Close fullcontinuousbody update for this loss, not allRNC/backbones. V261 launched bounded adapter+mass matched CE/RNC, original104 entirelyfrozen/oneSWaddition/max2/80-14; gains unproved.

Closed weights removed per retention policy; CHECKPOINT_DELETION.json preserves paths/SHA/reasons. Original104 and current252 positive retained.
