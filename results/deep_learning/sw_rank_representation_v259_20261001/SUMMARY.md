# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v259 continuous representation supervision, fixed original default-state branch

## Focus
Inspired by Rank-N-Contrast author loss https://github.com/kaiwenzha/Rank-N-Contrast/blob/main/loss.py and paper https://arxiv.org/abs/2210.01189. Local variant applies pairwise target-distance order contrast to one-view existing128hidden, no separate projector, only real continuousSW training points. Original paper uses two augmentedviews; this is a mechanism transfer, not full reproduction. Rank denominators include all target distances >=currentpair, ties included; efficient sorted suffixsum implementation. Feature negativeL2/temperature2, own physicalSW target distances.

## Explored and validation
Prepared. Same251 continuousbody/mass architecture858522trainableparams/init/loss/optimizer botharms, original104 atom network immutable. Entire continuous branch3e-5 plus new12609mass/background1e-4, SAM0/evalNoDropKey. CE ownhalfhard-halftolerance plus auxiliary coefficient0 or0.1 (wholeloss/3 convention). No teacher/extra data/pseudolabels/validation loss. Main likelihood and auxiliary train jointly ONE SWphase, starts104only/no251trainedweights. Sampling trainingcentral rows with rawindex%32==0, duplicate rowsremoved, max8pertrainingwell/max256total; metadata used only for quota, no wellID feature. Botharms compute same sampler/auxiliary diagnostics, no added stochasticRNG. Original9cu

full reproduction. Rank denominators include all target distances >=currentpair, ties included; efficient sorted suffixsum implementation. Feature negativeL2/temperature2, own physicalSW target distances.

## Explored and validation
Prepared. Same251 continuousbody/mass architecture858522trainableparams/init/loss/optimizer botharms, original104 atom network immutable. Entire continuous branch3e-5 plus new12609mass/background1e-4, SAM0/evalNoDropKey. CE ownhalfhard-halftolerance plus auxiliary coefficient0 or0.1 (wholeloss/3 convention). No teacher/extra data/pseudolabels/validation loss. Main likelihood and auxiliary train jointly ONE SWphase, starts104only/no251trainedweights. Sampling trainingcentral rows with rawindex%32==0, duplicate rowsremoved, max8pertrainingwell/max256total; metadata used only for quota, no wellID feature. Botharms compute same sampler/auxiliary diagnostics, no added stochasticRNG. Original9curves/Q50/fold5/256bins/800actions/80-14/min10 intact. Final RAMbest/frozenatom/all144665actualcomposition required; no diskreload/bitwise trainingrerun claim (known CUDA attention/interpolation backward warn_only).

## Gain
Unproved. Predefined RNC-minus-CE samearchitecture/parameterupdates/budget, absolute retainedSW25274.149228207812. RNC no extra parameters;858522 updates includes unused continuouscopygate rows which have no loss butweightdecay, original atom gate remains unchanged. No original positive models/delivery overwritten. Sixalone/sevenwith retainedPERM248/total<=8.

## Not worth continuing
251 body likelihood coadapt had no learnedgain. This adds explicit continuous geometry to same update scope, new hypothesis. If RNC auxiliary improves but officialcontinuous score does not, close weight0.1/temp2 transfer; no temperature/weight grid. Do not assume featuregeometry fragmentation/noisylabels are established problems. Research alternatives Ti-MAE/TS2Vec not simultaneously compounded.

## Timing
Record individual and actual concurrentprocesscreation-to-finalreportwallclock, peakMiB; startup/model/data/initialeval included, preparation/CPUassemblyexcluded; no serialspeedup claim. 256^2 auxiliary bounded, never8192^2.

V259 both arms failed before first optimizer step: torch argsort stable overload requires dim as keyword. No trained gain; immutable source/log/error kept. V260 new source fixes only argsort(dim=-1,stable=True), resets both arms from original104, identical hypothesis/config.

Closed weights removed per retention policy; CHECKPOINT_DELETION.json preserves paths/SHA/reasons. Original104 and current252 positive retained.


