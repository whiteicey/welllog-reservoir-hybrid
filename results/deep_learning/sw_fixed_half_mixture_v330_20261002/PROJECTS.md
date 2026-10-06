# Goal / completed / planned / unfinished
Goal: testtrainingcoadaptationtohighernewdistributionfraction onpositivejointbody. Implemented onehalf candidate/atomictrainer/recorder. PlannedfixedfifthlaunchonlyafterfreeGPUslot. Unfinishedrealgain/finalchecks/fullOOF82. Afterkeepusefulorclosewithoutalphagrid.

## Launch
Launched half PID76420 with zero other GPU jobs; original104-only initialSW74.12175268976877 and92289 joint trainable encoder/readout parameters verified during ordinary training. Fixed target mixture alpha0.5 ramp5, compared with historical326 alpha0.25; no posthoc validation mixing, additional trained repair checkpoint or extra phase. Results pending.

V330 half: SW74.142277545469, actualTOTAL80.619659069053; fresh297encoder/state/readout +21600 own104norm128 latentlatefusion, original104only; no trained321repairweights, oneaddedSWphase. No fullOOF/otherfolds.

## Final outcome
Natural stop27/best13, restored SW74.1422775454687. Same92289-capacity historical326 alpha0.25 versus fixed0.5 at predefined27: whole -0.027265486606; default98.5086222741472 vs98.63988120716091 (-0.131258933014); continuous18.175039918709572 vs17.96344208824835 (+0.211597830461). Whole target best only, not CONT epoch selection. Useful diagnostic tradeoff, no whole gain/adoption; close fixed-half setting without alpha/LR grid. Keep326 current8 TOTAL80.6990374411772 and default5 unchanged; no remainingfolds, fullOOF82 unmet. All joint encoder/readout groups received gradients, all144665 best RAM replay/frozen exact, not independent disk reload. Individual255.737770s/process260.306205s, solo GPU peak1779.212MiB; prep/initial/train/restore included, implementation/CPUfinalization excluded. Historical326 ran concurrently, so wall-times do not establish speed gain. Recorder inherited late-latent-fusion prose is incorrect: actual326 morphology/ownlatentEARLY fresh architecture per CONFIG/source.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
