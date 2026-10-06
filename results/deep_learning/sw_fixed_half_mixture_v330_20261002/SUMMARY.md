# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW fixed-half predictive mixture joint training v330

## Focus / control
Freshpositive326first48 body/state/FiLM/ownlatentearly/morph/readout jointly trained from original104frozenprioronly. Onechange: alpha=.5*min(epoch/5,1) vs326 .25*ramp. Same92289params/initialnewweights/input/seed/Q/losscoefficients/decoder. Itchangestraininglikelihood/RPS andsamefinalpredictiveprobability, notposthocvalidation blend; initialalpha0preservesoriginalprior. No326/329repairweights/phasechain, oneSWphase/6standalone/max8. Thisis coadaptationexperiment, notnewbackbonearchitecture; allnewencoder/readoutparameters remainjoint, nothead-only tuning.

## Explored / actual mathematics
[Source audit](../cross_domain_research_20261001/SW_FIXED_HALF_MIXTURE_AUDIT_20261002.md). Original315 learnedallocation range is0..0.5, not0.125..0.375; scalarbest26 approxalpha.045, so differentfromfixed.5trajectory anddoesnot establish.5invalid. MainharddensityCE gradient r*(q-onehot), r=alpha*q_true/((1-alpha)*p_true+alpha*q_true); whenptrue?0,r?1 despitealpha.25. DirectauxcontinuousCE independentofalpha, conditionalRPS coefficient1. Therefore .25doesnot genericallycapcontinuousgradients atquarter orprovelimitingbottleneck. Increasingalpha mayharmgoodprior/defaultBayesaction. Trainingdirectownlabels only, nogradientintofrozenprior/no teacher.

## Gains / fair scope
Launched and training; gain not established. Histo

b64x64/cos80/seed20261010/original9/206+relative/Q50/64train16val fixedfifth unchanged. Savebestthrough27 duringordinarytraining; historicalreference sameactual27 butnotsimultaneouspair, optimizationtrajectoryintentionallydiffers. Fullbestafter27 absolutecandidate not equalworkgain; stopbefore27 actualsmallerworkdisclosed. OwnwholeSWbest, neverpickCONT-onlyepoch. No numericalpreflight/review; ordinarysourceinitial/frozen/bestreplay checks only, syntaxcheck.

## Stop / cost / next
Onlyfixed.5candidate, no alpha/bounds/temp/LR grid or decoder scan. IfwholeorCONT failsbeat326first, keep326 andclosethissetting. Notreuse329width, no extraphase/voting/testlabels/pseudo/externalweights/otherfolds. GPUcap2 andactualcountcheckbeforelaunch; 328/329 terminal before330 launch; no automatic queue. Recordactualmemory/wall/concurrency, prep/initial/train/bestrestore included, implementation/CPUfinalizationexcluded; noisolatedspeedclaim. Current8TOTAL80.699037441177/strict808false/fullOOF82unmet.

## Launch
Launched half PID76420 with zero other GPU jobs; original104-only initialSW74.12175268976877 and92289 joint trainable encoder/readout parameters verified during ordinary training. Fixed target mixture alpha0.5 ramp5, compared with historical326 alpha0.25; no posthoc validation mixing, additional trained repair checkpoint or extra phase. Results pending.

V330 half: SW74.142277545469, actualTOTAL80.619659069053; fresh297encoder/state/readout +21600 own104norm128 latentlatefusion, original104only; no trained321repairweights, oneaddedSWphase. No fullOOF/otherfolds.

## Final outcome
Natural stop27/best13, restored SW74.1422775454687. Same92289-capacity historical326 alpha0.25 versus fixed0.5 at predefined27: whole -0.027265486606; default98.5086222741472 vs98.63988120716091 (-0.131258933014); continuous18.175039918709572 vs17.96344208824835 (+0.211597830461). Whole target best only, not CONT epoch selection. Useful diagnostic tradeoff, no whole gain/adoption; close fixed-half setting without alpha/LR grid. Keep326 current8 TOTAL80.6990374411772 and default5 unchanged; no remainingfolds, fullOOF82 unmet. All joint encoder/readout groups received gradients, all144665 best RAM replay/frozen exact, not independent disk reload. Individual255.737770s/process260.306205s, solo GPU peak1779.212MiB; prep/initial/train/restore included, implementation/CPUfinalization excluded. Historical326 ran concurrently, so wall-times do not establish speed gain. Recorder inherited late-latent-fusion prose is incorrect: actual326 morphology/ownlatentEARLY fresh architecture per CONFIG/source.


