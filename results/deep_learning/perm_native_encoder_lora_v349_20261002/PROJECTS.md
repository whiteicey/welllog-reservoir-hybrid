# V349 PERM native encoder rank4

## Focus and hypothesis
Keep original154 full warm predictive function; jointly update two true expert enc2.expand rank4 modules (2048) and original154 context readout (82720). Zero B initialization, original parameter freeze, native gradient-enabled forward/capture. No trained repair checkpoint, native Gaussian16 and NLL/BCE/jointRPS1 and decoder unchanged. One added PERM phase; 6 phases with original POR/SW, at most8 with retained alternative POR/SW.

## Explored and verification
Prepared; no score yet. Native forward reused from166, but complete154 head restored unlike166. Historical270 warm head-only and late joint both best0/natural14; this is a near negative, not first encoder adaptation. No new head-only job under latest rule. Historical head-only comparison is not concurrent or capacity matched. Original input/split/Q50,80/14/min10,b64,64steps; solo preferred/max2. Initialization and group gradient readback are built into normal training, no separate preflight/review. Report both whole-selected CONT and DEFAULT; LoRA parameter norms at restored best.

## Gain
Unproven. Source154 PERM89.405564856919/CONT74.096911943817; retained336 PERM89.518389398791/CONT74.009637083197; current eight-stage TOTAL80.704705566183 unchanged. Fifth strict80.8 plus useful gain required for other folds; full OOF82 unmet.

## Do not extend
If no whole/CONT improvement, close rank4 enc2/nativewarm setting, no rank/location/LR sweep. 270/162/166/272 near negatives acknowledged; not all backbone adaptation invalid. No repair chain, teachers, labels from validation in training, external voting, or old task restart.

## Timing
Record individual preprocessing/warmup/training/restoration seconds and process wallclock/peak separately. Solo execution, no paired wallclock claims. RAM best replay is not independent diskreload.

Next: explicit launch, read terminal results, update atlas and decide retention.

Actual original154 checkpoint two enc2.expand weight tensors are128x128x1, CPU source shape readback. V348 assertion expected128to256 from inaccurate research; failed before model.cuda/optimizer/first batch, no score. Preserve348sources/logs/status;349 corrects only dimensions/parameter count, not hypothesis.

349 runtime snapshot: initialized exact native154 score89.405564856919; real enc2LoRA/readout gradient groups recorded in GRADIENT_SNAPSHOT, A/B current norms separate, livePID88616, gainunproven.849.643MiB peak so far, solo, no OOM/no otherfolds.

V349 final PERM89.405564856919, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds119.130165/peakMiB849.642578. No otherfolds.

V349 terminal: natural14/best0 PERM89.405564856919, CONT74.09691194381553, exactnative154predictions changedrows0. No gain/adoption; whole vs336-0.11282454187171709. Encoder/readout gradientsnonzero, terminal currentBnorms changed but best0 restoreszeroB, notgradientblock. Solo119.130s/849.643MiB inclprep/warmup/restoration; RAM144665replay/frozenexact notdiskreload. Closedenc2rank4/nativewarmsetting/no sweep, fullOOF82 andfifth808 unmet; current341TOTAL80.704705566183 unchanged.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
